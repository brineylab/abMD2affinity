#!/usr/bin/env python3
"""
build_manifest.py — Build the system manifest from the mutants TSV
(columns: pdb_id, mutant) plus the per-PDB `structures` map from config.yaml.

Each `structures` entry gives the input PDB file and, optionally, a `chain_map`
that remaps AB-Bind antibody labels (H/L) to the real chain letters in that
structure (e.g. 1dqj: {H: B, L: A}). Antigen chains in a mutation string are
already real chain letters and pass through unchanged.

    structures:
      1dqj: {file: data/structures/1dqj_fv.pdb, chain_map: {H: B, L: A}}
      1jrh: {file: data/structures/1jrh_fv.pdb}

A mutant field of "" or "WT" (case-insensitive) marks a wild-type system; one
WT system is auto-generated per PDB in addition to any explicit WT rows.

Importable via build_systems(...), or run standalone to write a JSON manifest.
"""

import csv
import re
import sys
from pathlib import Path

WT_TOKENS = {"", "wt", "wildtype", "wild-type"}


def mutation_tag(mut_str: str) -> str:
    tag = mut_str.strip().replace(":", "-").replace(",", "_").replace(" ", "")
    return re.sub(r"[^A-Za-z0-9._-]", "", tag)


def has_deletion(mut_str: str) -> bool:
    return "delta" in mut_str.lower() or "Δ" in mut_str


def _chains_in_pdb(pdb_file: Path) -> list[str]:
    """Ordered-unique chain ids present in ATOM/HETATM records, sorted."""
    chains: list[str] = []
    with open(pdb_file) as f:
        for line in f:
            if line[:6].strip() in ("ATOM", "HETATM"):
                ch = line[21]
                if ch not in chains:
                    chains.append(ch)
    return sorted(chains)


def _remap(mut_str: str, chain_map: dict) -> str:
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


def build_systems(mutants_tsv, structures: dict, auto_wt: bool = True) -> dict:
    """
    Parse a (pdb_id, mutant) TSV against the per-PDB `structures` map and return
    a dict of system definitions keyed by '{pdb}_{mutation_tag}'.

    When auto_wt is True, one WT system is auto-generated per PDB in addition to
    any explicit WT rows in the TSV.
    """
    mutants_tsv = Path(mutants_tsv)
    systems: dict = {}
    wt_seen: set = set()

    with open(mutants_tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pdb = row["pdb_id"].strip().lower()
            mut_str = row["mutant"].strip()

            spec = structures.get(pdb)
            if spec is None:
                print(f"WARNING: skipping {pdb} {mut_str!r} — no 'structures' entry in config",
                      file=sys.stderr)
                continue
            if isinstance(spec, str):
                spec = {"file": spec}

            pdb_file = Path(spec["file"])
            if not pdb_file.exists():
                print(f"WARNING: skipping {pdb} {mut_str!r} — {pdb_file} not found",
                      file=sys.stderr)
                continue
            if mut_str and has_deletion(mut_str):
                print(f"WARNING: skipping {pdb} {mut_str!r} — deletions are not supported",
                      file=sys.stderr)
                continue

            chain_map = spec.get("chain_map", {}) or {}
            chains = _chains_in_pdb(pdb_file)
            is_wt = mut_str.lower() in WT_TOKENS
            wt_tag = f"WT_{''.join(chains)}"

            entry = {
                "pdb":                pdb,
                "mutation_str_input": None if is_wt else mut_str,
                "mutation_str_pdb":   None if is_wt else _remap(mut_str, chain_map),
                "mutation_tag":       wt_tag if is_wt else mutation_tag(mut_str),
                "chain_map":          chain_map,
                "chains_to_extract":  chains,
                "pdb_file":           str(pdb_file),
                "is_wt":              is_wt,
            }
            systems.setdefault(f"{pdb}_{entry['mutation_tag']}", entry)

            # One auto-generated WT system per PDB.
            if auto_wt and pdb not in wt_seen:
                wt_seen.add(pdb)
                systems.setdefault(f"{pdb}_{wt_tag}", {
                    **entry,
                    "mutation_str_input": None,
                    "mutation_str_pdb":   None,
                    "mutation_tag":       wt_tag,
                    "is_wt":              True,
                })

    return systems


def main():
    import argparse
    import json

    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mutants_tsv")
    parser.add_argument("config", help="config.yaml providing the 'structures' map")
    parser.add_argument("-o", "--out", default="manifest.json")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    systems = build_systems(args.mutants_tsv, cfg["structures"])

    print(f"Total systems: {len(systems)}")
    for pdb in sorted({v["pdb"] for v in systems.values()}):
        count = sum(1 for v in systems.values() if v["pdb"] == pdb)
        print(f"  {pdb}: {count} systems")

    with open(args.out, "w") as f:
        json.dump(systems, f, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
