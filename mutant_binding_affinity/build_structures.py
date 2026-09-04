#!/usr/bin/env python3
"""
build_structures.py — Materialise AB-Bind mutant structures and emit the
structures list that run_MD consumes.

For every (pdb_id, mutant) row of the mutants TSV: remap AB-Bind H/L antibody
labels to real chain letters via that structure's chain_map, apply the
mutation with PyMOL (mutate_structure.py), and write the mutant PDB to
<mutants_dir>/<pdb_id>_<tag>.pdb. WT systems point straight at the original
structure. The output is a structures file the MD pipeline runs unchanged:

    base_dir: .
    structures:
      - name: 1bj1fv_WT
        path: data/structures/1bj1fv.pdb
      - name: 1bj1fv_V-F17A
        path: mutants/1bj1fv_V-F17A.pdb

base_dir "." anchors run_MD's outputs at this directory, so results land in
mutant_binding_affinity/results/{name}/ by default.

Existing mutant files are kept as-is, so the script is resumable — delete a
mutant PDB to force its regeneration. Deletion mutants are skipped with a
warning. Needs the PyMOL interpreter (pymol_python_bin in abbind.yaml).

Usage:
    python build_structures.py abbind.yaml [--dry-run]
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MUTATE_SCRIPT = HERE / "mutate_structure.py"

WT_TOKENS = {"", "wt", "wildtype", "wild-type"}


def mutation_tag(mut_str: str) -> str:
    """Filesystem-safe tag: 'B:D32A,A:S30N' -> 'B-D32A_A-S30N'."""
    tag = mut_str.strip().replace(":", "-").replace(",", "_").replace(" ", "")
    return re.sub(r"[^A-Za-z0-9._-]", "", tag)


def has_deletion(mut_str: str) -> bool:
    return "delta" in mut_str.lower() or "Δ" in mut_str


def remap(mut_str: str, chain_map: dict) -> str:
    """Rewrite the chain letters of a mutation string through chain_map."""
    out = []
    for token in mut_str.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            ch, rest = token.split(":", 1)
            out.append(f"{chain_map.get(ch.strip(), ch.strip())}:{rest}")
        else:
            out.append(token)
    return ",".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="abbind.yaml (relative paths resolve against it)")
    ap.add_argument("-o", "--out", default=None,
                    help="output structures YAML (default: the config's structures_out)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the planned systems and mutations; write nothing")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent
    abbind_structures = cfg["structures"]
    mutants_dir = base / cfg.get("mutants_dir", "mutants")
    pymol_python = cfg.get("pymol_python_bin", "/opt/conda/envs/pymol/bin/python")
    out_path = Path(args.out).resolve() if args.out else (base / cfg["structures_out"])

    systems: dict[str, dict] = {}
    wt_seen: set[str] = set()
    n_mutated = n_kept = 0

    def add(name: str, pdb_file: Path) -> None:
        if name in systems:
            print(f"WARNING: duplicate system {name!r} — keeping the first",
                  file=sys.stderr)
            return
        # path relative to the structures file's own directory
        systems[name] = {"name": name,
                         "path": os.path.relpath(pdb_file, out_path.parent)}

    with open(base / cfg["mutants_tsv"], newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pdb = row["pdb_id"].strip().lower()
            mut_str = row["mutant"].strip()

            spec = abbind_structures.get(pdb)
            if spec is None:
                print(f"WARNING: skipping {pdb} {mut_str!r} — no 'structures' "
                      f"entry in abbind.yaml", file=sys.stderr)
                continue
            if isinstance(spec, str):
                spec = {"file": spec}
            pdb_file = base / spec["file"]
            if not pdb_file.exists():
                print(f"WARNING: skipping {pdb} {mut_str!r} — {pdb_file} not found",
                      file=sys.stderr)
                continue

            # every structure gets a WT system pointing at the original file
            if pdb not in wt_seen:
                wt_seen.add(pdb)
                add(f"{pdb}_WT", pdb_file)

            if mut_str.lower() in WT_TOKENS:
                continue
            if has_deletion(mut_str):
                print(f"WARNING: skipping {pdb} {mut_str!r} — deletions are not "
                      f"supported", file=sys.stderr)
                continue

            mapped = remap(mut_str, spec.get("chain_map", {}) or {})
            tag = mutation_tag(mapped)
            if not tag:
                print(f"WARNING: skipping {pdb} {mut_str!r} — mutation produces "
                      f"an empty tag", file=sys.stderr)
                continue
            name = f"{pdb}_{tag}"
            mutant_file = mutants_dir / f"{name}.pdb"

            if mutant_file.exists():
                n_kept += 1
            elif args.dry_run:
                n_mutated += 1
                print(f"would mutate  {name}: {mapped}")
                add(name, mutant_file)
                continue
            else:
                mutants_dir.mkdir(parents=True, exist_ok=True)
                proc = subprocess.run(
                    [pymol_python, str(MUTATE_SCRIPT),
                     str(pdb_file), str(mutant_file), mapped],
                    capture_output=True, text=True,
                )
                if proc.returncode != 0 or not mutant_file.exists():
                    sys.exit(f"ERROR: mutation failed for {name} ({mapped}):\n"
                             f"{proc.stdout}\n{proc.stderr}")
                n_mutated += 1

            add(name, mutant_file)

    if not systems:
        sys.exit("No systems produced — check the mutants TSV and abbind.yaml.")

    if args.dry_run:
        n_wt = sum(1 for s in systems if s.endswith("_WT"))
        print(f"\n[ dry-run ] {len(systems)} systems: {n_wt} WT + "
              f"{len(systems) - n_wt} mutants ({n_mutated} to mutate, "
              f"{n_kept} already present) — nothing written")
        return

    header = (
        f"# AB-Bind structures list — generated by build_structures.py\n"
        f"# from {cfg['mutants_tsv']} + {Path(args.config).name}\n"
        f"# (mutant PDBs in {cfg.get('mutants_dir', 'mutants')}/, created with PyMOL).\n"
        f"# Run the MD pipeline with:\n"
        f"#   cd ../run_MD && snakemake --configfile config.yaml \\\n"
        f"#       --config structures_file={os.path.relpath(out_path, (HERE / '../run_MD').resolve())}\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(header)
        yaml.safe_dump({"base_dir": ".", "structures": list(systems.values())},
                       f, sort_keys=False, default_flow_style=False)

    n_wt = sum(1 for s in systems if s.endswith("_WT"))
    print(f"Wrote {len(systems)} systems ({n_wt} WT + {len(systems) - n_wt} "
          f"mutants: {n_mutated} mutated this run, {n_kept} already present) "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
