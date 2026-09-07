#!/usr/bin/env python
"""Package a run_MD results tree for hand-off (e.g. to a cluster).

Default ("essentials") mode packages exactly what production MD needs per
system — preprocessing/{npt.gro, npt.cpt, topol_ions.top, *.itp} plus
MD/md.mdp and manifest.json — skipping the bulky intermediates (solvated/
ions/em/nvt structures, minimization step files, trajectories, logs) that
dominate a full results tree. --full packages everything except GROMACS
#backup# files.

Usage (from run_MD/):

    python scripts/package_preprocessing.py ../haddock_out/results --dry-run
    python scripts/package_preprocessing.py ../haddock_out/results -o prepd.tar.gz
    python scripts/package_preprocessing.py ../haddock_out/results \
        --manifest-root /mnt/home/me/results -o prepd.tar.gz

--manifest-root rewrites each system's output_dir in the packaged
manifest.json to <root>/<system>, making the archive valid wherever it is
untarred and directly consumable by slurm/launch_experiments.py (whose
manifest paths otherwise still point at the machine the pipeline ran on).

Compression is gzip through pigz when available (parallel, much faster on
large archives), else plain gzip; --compression none writes an uncompressed
tar. GROMACS #backup# files are always excluded.
"""

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

ESSENTIAL_FILES = ("preprocessing/npt.gro", "preprocessing/npt.cpt",
                   "preprocessing/topol_ions.top")
ESSENTIAL_MDP = "MD/md.mdp"


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0


def is_backup(path: str) -> bool:
    name = os.path.basename(path)
    return name.startswith("#") and name.endswith("#")


def select(root: str, full: bool):
    """[(abs_path, arcname), ...] for every system under root. manifest.json is
    deliberately NOT in this list — add_manifest() adds it separately (and
    rewrites output_dirs when --manifest-root is given), so it must not also
    appear here or the archive would carry two copies."""
    systems = sorted(d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d, "preprocessing")))
    if not systems:
        sys.exit(f"no system directories (*/preprocessing) under {root}")
    if not os.path.isfile(os.path.join(root, "manifest.json")):
        sys.exit(f"no manifest.json under {root} — rerun preprocess "
                 "(write_manifest) before packaging")

    out, missing = [], {}
    for sysdir in systems:
        base = os.path.join(root, sysdir)
        if full:
            for dirpath, _dirs, files in os.walk(base):
                if is_backup(dirpath):
                    continue
                for f in files:
                    p = os.path.join(dirpath, f)
                    if not is_backup(p) and os.path.basename(p) != "manifest.json":
                        out.append((p, os.path.relpath(p, root)))
        else:
            want = list(ESSENTIAL_FILES)
            want.append(ESSENTIAL_MDP)
            want += [os.path.join("preprocessing", f)
                    for f in sorted(os.listdir(os.path.join(base, "preprocessing")))
                    if f.endswith(".itp")]
            for rel in want:
                p = os.path.join(base, rel)
                if os.path.exists(p):
                    out.append((p, os.path.join(sysdir, rel)))
                else:
                    missing.setdefault(sysdir, []).append(rel)

    return systems, out, missing


def add_manifest(tf, root: str, manifest_root: str):
    src = os.path.join(root, "manifest.json")
    if not manifest_root:
        tf.add(src, arcname="manifest.json", recursive=False)
        return
    with open(src) as fh:
        manifest = json.load(fh)
    for name, entry in manifest.items():
        entry["output_dir"] = os.path.join(manifest_root, name)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(manifest, tmp, indent=2)
    try:
        tf.add(tmp.name, arcname="manifest.json", recursive=False)
    finally:
        os.unlink(tmp.name)


def write_archive(out: str, root: str, files, manifest_root: str, compress: str):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if compress == "none":
        tf = tarfile.open(out, "w")
        proc = None
    elif shutil.which("pigz"):
        fh = open(out, "wb")
        proc = subprocess.Popen(["pigz"], stdin=subprocess.PIPE, stdout=fh)
        tf = tarfile.open(fileobj=proc.stdin, mode="w|")
        print(f"(compressing with pigz: {shutil.which('pigz')})")
    else:
        proc = None
        tf = tarfile.open(out, "w:gz")

    try:
        add_manifest(tf, root, manifest_root)
        for i, (path, arcname) in enumerate(files, 1):
            tf.add(path, arcname=arcname, recursive=False)
            if i % 500 == 0:
                print(f"  {i}/{len(files)} files...", flush=True)
    finally:
        tf.close()
        if proc is not None:
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                sys.exit(f"pigz failed with exit code {proc.returncode}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_root", help="results dir holding */preprocessing "
                                         "and manifest.json (e.g. ../haddock_out/results)")
    ap.add_argument("-o", "--output",
                    help="archive to write (default: <root>_essentials.tar.gz "
                         "or <root>_full.tar.gz next to the results dir)")
    ap.add_argument("--full", action="store_true",
                    help="package everything (except GROMACS #backup# files), "
                         "not just the production essentials")
    ap.add_argument("--manifest-root", metavar="DIR",
                    help="rewrite each system's output_dir in the packaged "
                         "manifest.json to DIR/<system> (valid wherever the "
                         "archive is untarred)")
    ap.add_argument("--compression", choices=("gz", "none"), default="gz",
                    help="gz (default; pigz if available) or none (plain tar)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be packaged; write nothing")
    args = ap.parse_args()

    root = os.path.abspath(args.results_root)
    if not os.path.isdir(root):
        sys.exit(f"{root}: no such directory")

    mode = "full" if args.full else "essentials"
    default = f"{root.rstrip('/')}_{mode}.tar.gz" if args.compression == "gz" \
        else f"{root.rstrip('/')}_{mode}.tar"
    out = os.path.abspath(args.output or default)

    systems, files, missing = select(root, args.full)
    total = sum(os.path.getsize(p) for p, _ in files)
    print(f"{root}: {len(systems)} systems, {len(files) + 1} files "
          f"({mode}), {human(total)} before compression")

    if missing and not args.full:
        for sysdir, rels in missing.items():
            print(f"MISSING  {sysdir}: {', '.join(rels)}", file=sys.stderr)
        sys.exit(f"{len(missing)} system(s) incomplete — rerun preprocessing "
                 "or use --full")
    if missing and args.full:
        for sysdir, rels in missing.items():
            print(f"WARN     {sysdir}: missing {', '.join(rels)}", file=sys.stderr)

    if args.dry_run:
        print("dry-run: nothing written")
        return

    print(f"writing {out}")
    write_archive(out, root, files, args.manifest_root, args.compression)
    print(f"done: {out} ({human(os.path.getsize(out))})")


if __name__ == "__main__":
    main()
