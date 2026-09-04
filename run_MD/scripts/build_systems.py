#!/usr/bin/env python3
"""
build_systems.py — Expand the structures list into the pipeline's system
manifest (one entry per MD run).

The pipeline input is a structures file (`structures_file` in config.yaml,
default structures.yaml) with a top-level `base_dir` plus a `structures` list:

    base_dir: /data/md              # anchor for all outputs
    output_dir: runs                # optional — replaces "results" below
    structures:
      - name: 1bj1fv                # system name (path segment under outputs)
        path: data/1bj1fv.pdb       # input PDB
        output_dir: special/x       # optional — per-structure output dir
        chains: [A, B]              # optional — default: every chain
        md_ns: 500                  # optional — overrides the global default

Relative `path`/`base_dir` resolve against the structures file's directory.
Each system's output dir is, in order of precedence:

  1. the entry's `output_dir` (relative -> under base_dir)
  2. {base_dir}/{output_dir or "results"}/{name}/

Importable via load_structures()/build_systems(), or run standalone to
validate a structures file and dump its manifest:

    python scripts/build_systems.py structures.yaml [--config config.yaml]
"""

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import yaml

# System names become path segments under the output dirs — keep them
# filesystem-safe.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _md_len_set(v) -> bool:
    """A length value counts as set unless absent, null or empty."""
    return v is not None and str(v).strip().lower() not in ("", "null")


@lru_cache(maxsize=None)
def _chains_in_pdb(pdb_file: Path) -> tuple[str, ...]:
    """Ordered-unique chain ids present in ATOM/HETATM records, sorted."""
    chains: list[str] = []
    with open(pdb_file) as f:
        for line in f:
            if line[:6].strip() in ("ATOM", "HETATM"):
                ch = line[21]
                if ch not in chains:
                    chains.append(ch)
    return tuple(sorted(chains))


def load_structures(config: dict) -> dict:
    """
    The structures document: {'base_dir', 'output_dir'?, 'structures',
    'anchor'} — either inline in config.yaml (anchored at the current
    directory) or from `structures_file` (anchored at that file's directory).
    """
    inline = "structures" in config
    if inline:
        doc = dict(config)
        anchor = Path.cwd()
    else:
        path = Path(config.get("structures_file", "structures.yaml"))
        if not path.exists():
            raise FileNotFoundError(
                f"structures file not found: {path} (config key 'structures_file')"
            )
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        anchor = path.resolve().parent

    if not isinstance(doc.get("structures"), list) or not doc["structures"]:
        raise ValueError("'structures' must be a non-empty list of "
                         "{name, path, ...} entries")
    if not doc.get("base_dir"):
        raise ValueError("structures file must set a top-level 'base_dir'")
    return {"doc": doc, "anchor": anchor}


def _entry_length(name: str, spec: dict,
                  default_md_ns, default_md_steps) -> tuple[str, object]:
    """Effective (kind, value) production length: entry override wins over the
    global default. Exactly one side must supply one."""
    e_ns, e_steps = spec.get("md_ns"), spec.get("md_steps")
    if _md_len_set(e_ns) and _md_len_set(e_steps):
        raise ValueError(
            f"structures entry {name!r}: set at most one of 'md_ns' / "
            f"'md_steps'"
        )
    if _md_len_set(e_ns):
        return "md_ns", e_ns
    if _md_len_set(e_steps):
        return "md_steps", e_steps
    if _md_len_set(default_md_ns):
        return "md_ns", default_md_ns
    if _md_len_set(default_md_steps):
        return "md_steps", default_md_steps
    raise ValueError(
        f"structures entry {name!r}: no production-MD length — set "
        f"'md_ns'/'md_steps' on the entry or a global default in config.yaml"
    )


def _entry_chains(name: str, spec: dict, pdb_file: Path) -> list[str]:
    chains = spec.get("chains")
    if chains is None:
        return list(_chains_in_pdb(pdb_file))
    if isinstance(chains, str):
        chains = list(chains)
    if not isinstance(chains, list):
        raise ValueError(f"structures entry {name!r}: 'chains' must be a list")
    for ch in chains:
        if not isinstance(ch, str) or len(ch) != 1:
            raise ValueError(
                f"structures entry {name!r}: chain ids must be single "
                f"characters, got {ch!r}"
            )
    return sorted(set(chains))


def build_systems(loaded: dict,
                  default_md_ns=None, default_md_steps=None) -> tuple[dict, Path]:
    """
    Expand a loaded structures document into (systems, results_root).

    systems      {name: {name, pdb_file, chains, md_ns, md_steps, output_dir}}
    results_root default root for per-system outputs and the manifest
                 ({base_dir}/{output_dir or "results"})
    """
    doc, anchor = loaded["doc"], loaded["anchor"]
    structures = doc["structures"]

    if _md_len_set(default_md_ns) and _md_len_set(default_md_steps):
        raise ValueError(
            "set at most one of 'md_ns' / 'md_steps' as the global default"
        )
    if not doc.get("base_dir"):
        raise ValueError("structures file must set a top-level 'base_dir'")

    base_dir = Path(doc["base_dir"])
    if not base_dir.is_absolute():
        base_dir = anchor / base_dir

    top_output = doc.get("output_dir")
    if top_output:
        top_output = Path(top_output)
        results_root = (top_output if top_output.is_absolute()
                        else base_dir / top_output)
    else:
        results_root = base_dir / "results"

    systems: dict = {}
    for idx, spec in enumerate(structures):
        if not isinstance(spec, dict):
            raise ValueError(
                f"structures entry #{idx + 1}: expected a mapping with 'name' "
                f"and 'path' keys, got {spec!r}"
            )
        name = str(spec.get("name", "")).strip()
        if not name:
            raise ValueError(f"structures entry #{idx + 1}: missing 'name'")
        if not spec.get("path"):
            raise ValueError(f"structures entry {name!r}: missing 'path'")
        if not NAME_RE.match(name):
            raise ValueError(
                f"structures entry {name!r}: name must match {NAME_RE.pattern}"
            )
        if name in systems:
            print(f"WARNING: duplicate system {name!r} — keeping the first",
                  file=sys.stderr)
            continue

        pdb_file = Path(spec["path"])
        if not pdb_file.is_absolute():
            pdb_file = anchor / pdb_file
        if not pdb_file.exists():
            raise FileNotFoundError(
                f"structures entry {name!r}: {pdb_file} not found"
            )

        kind, value = _entry_length(name, spec, default_md_ns, default_md_steps)

        out_dir = spec.get("output_dir")
        if out_dir:
            out_dir = Path(out_dir)
            if not out_dir.is_absolute():
                out_dir = base_dir / out_dir
        else:
            out_dir = results_root / name

        systems[name] = {
            "name":       name,
            "pdb_file":   str(pdb_file),
            "chains":     _entry_chains(name, spec, pdb_file),
            "md_ns":      value if kind == "md_ns" else None,
            "md_steps":   value if kind == "md_steps" else None,
            "output_dir": str(out_dir),
        }

    return systems, results_root


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structures", nargs="?", default="structures.yaml",
                    help="structures YAML (default: structures.yaml)")
    ap.add_argument("--config", default=None,
                    help="optional config.yaml to source the global md_ns/md_steps "
                         "default from")
    ap.add_argument("-o", "--out", default=None,
                    help="optional manifest JSON output path")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}

    with open(args.structures) as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc.get("structures"), list):
        sys.exit(f"{args.structures}: no top-level 'structures:' list")

    systems, results_root = build_systems({"doc": doc, "anchor": Path.cwd()},
                                           default_md_ns=cfg.get("md_ns"),
                                           default_md_steps=cfg.get("md_steps"))

    print(f"Total systems: {len(systems)}  (results root: {results_root})")
    for name, entry in systems.items():
        length = (f"md_ns={entry['md_ns']}" if entry["md_ns"] is not None
                  else f"md_steps={entry['md_steps']}")
        print(f"  {name}: {length}, chains {entry['chains']} -> {entry['output_dir']}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(systems, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
