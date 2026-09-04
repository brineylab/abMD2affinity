#!/usr/bin/env python3
"""
build_mutants_tsv.py — Derive the (pdb_id, mutant) TSV from the raw AB-Bind
dataset (data/PRO-25-393-s002.tsv), one block of rows per entry in the
`structures` map of abbind_scripts/abbind.yaml.

Each structures key is a pdb_id (e.g. 1dqjfv, 1dqjfab); its first 4 characters
are the dataset code looked up in the raw TSV's "#PDB" column. So a fab/fv
pair (1dqjfv + 1dqjfab) both draw the 1dqj mutants, giving two independent
systems.

Mutation strings are copied through unchanged (AB-Bind notation: H/L for the
antibody, real letters for antigen); the H/L -> real chain remapping happens
in build_structures.py via each structure's chain_map.

Usage:
    python abbind_scripts/build_mutants_tsv.py abbind_scripts/abbind.yaml
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def mutants_by_code(raw_tsv: Path) -> dict[str, list[str]]:
    """dataset code (lowercased '#PDB') -> ordered-unique list of mutation strings."""
    out: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with open(raw_tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            code = (row.get("#PDB") or "").strip().lower()
            mut = (row.get("Mutation") or "").strip()
            key = (code, mut)
            if code and key not in seen:
                seen.add(key)
                out[code].append(mut)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="abbind_scripts/abbind.yaml")
    ap.add_argument("-o", "--out", default=None,
                    help="output TSV (default: the config's mutants_tsv)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    structures = cfg["structures"]
    out_path = args.out or cfg["mutants_tsv"]

    by_code = mutants_by_code(Path(cfg["raw_tsv"]))

    rows: list[tuple[str, str]] = []
    for pdb_id in structures:
        code = pdb_id[:4].lower()
        muts = by_code.get(code)
        if not muts:
            print(f"WARNING: no rows in raw TSV for dataset code {code!r} "
                  f"(pdb_id {pdb_id!r})", file=sys.stderr)
            continue
        for mut in muts:
            rows.append((pdb_id, mut))

    if not rows:
        sys.exit("No rows produced — check the structures map and dataset codes.")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["pdb_id", "mutant"])
        w.writerows(rows)

    by_pdb: dict[str, int] = defaultdict(int)
    for pdb_id, _ in rows:
        by_pdb[pdb_id] += 1
    print(f"Wrote {len(rows)} rows for {len(by_pdb)} pdb_ids -> {out_path}")
    for pdb_id in sorted(by_pdb):
        print(f"  {pdb_id}: {by_pdb[pdb_id]} mutants")


if __name__ == "__main__":
    main()
