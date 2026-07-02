#!/usr/bin/env python3
"""
extract_chains.py — Extract specified chains from a PDB file.

Usage:
    python extract_chains.py input.pdb output.pdb A B C
"""

import sys
from pathlib import Path


def extract_chains(input_pdb: str, output_pdb: str, chains: list[str]) -> None:
    chains_set = set(chains)
    kept = 0
    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)

    with open(input_pdb) as fin, open(output_pdb, "w") as fout:
        for line in fin:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM", "ANISOU"):
                chain_id = line[21]
                if chain_id not in chains_set:
                    continue
                # Keep only blank or 'A' alternate conformations; normalize 'A' → blank
                altloc = line[16] if len(line) > 16 else " "
                if altloc not in (" ", "A"):
                    continue
                if altloc == "A":
                    line = line[:16] + " " + line[17:]
                fout.write(line)
                kept += 1
            elif record == "TER":
                chain_id = line[21] if len(line) > 21 else " "
                if chain_id in chains_set:
                    fout.write(line)
            elif record == "END":
                fout.write(line)

    print(f"Extracted {kept} atoms for chains {chains} → {output_pdb}")


def main():
    if len(sys.argv) < 4:
        print("Usage: extract_chains.py input.pdb output.pdb CHAIN [CHAIN ...]")
        sys.exit(1)

    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2]
    chains = sys.argv[3:]

    extract_chains(input_pdb, output_pdb, chains)


if __name__ == "__main__":
    main()
