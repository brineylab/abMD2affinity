#!/usr/bin/env python3
"""
mutate_structure.py — Apply point mutations to a PDB using PyMOL's mutagenesis wizard.

Usage:
    python mutate_structure.py input.pdb output.pdb "A:D217A,B:S30N"

Mutation format: CHAIN:WTRESNUM MUTAA  (e.g. A:D217A = chain A, Asp 217 → Ala)
Multiple mutations separated by commas.
"""

import os
import re
import sys
from pathlib import Path

THREE_LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def parse_mutations(mut_str: str) -> list[tuple[str, str, str, str]]:
    """
    Parse 'A:D217A,B:S30N' → [('A', 'D', '217', 'A'), ('B', 'S', '30', 'N')]
    Handles insertion codes (e.g. D60a → resnum='60a').
    """
    mutations = []
    for token in mut_str.strip().split(","):
        token = token.strip()
        if not token:
            continue
        # CHAIN:WTRESIDAAINS  e.g. A:D217A or A:D60aA
        m = re.match(r"([A-Za-z0-9]):([A-Z])([0-9]+[a-z]?)([A-Z])$", token)
        if not m:
            raise ValueError(f"Cannot parse mutation token: {token!r}")
        chain, wt, resnum, mut_aa = m.groups()
        mutations.append((chain, wt, resnum, mut_aa))
    return mutations


def apply_mutations(input_pdb: str, output_pdb: str, mut_str: str) -> None:
    import pymol
    from pymol import cmd

    mutations = parse_mutations(mut_str)
    if not mutations:
        raise ValueError("No mutations parsed from: " + repr(mut_str))

    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)

    pymol.finish_launching(["pymol", "-cq"])
    cmd.reinitialize()
    cmd.load(input_pdb, "mol")

    for chain, wt, resnum, mut_aa in mutations:
        mut_three = THREE_LETTER.get(mut_aa)
        if not mut_three:
            print(f"  WARNING: unknown AA code {mut_aa!r}, skipping {chain}:{wt}{resnum}{mut_aa}")
            continue

        selection = f"chain {chain} and resi {resnum}"

        # Warn if WT residue in structure doesn't match AB-Bind annotation
        import pymol as _pymol
        _pymol.stored.resn_check = []
        cmd.iterate(f"chain {chain} and resi {resnum} and name CA",
                    "stored.resn_check.append(resn)")
        if _pymol.stored.resn_check:
            actual = _pymol.stored.resn_check[0]
            expected = THREE_LETTER.get(wt, wt)
            if actual != expected:
                print(f"  WARNING: AB-Bind expects {expected} at {chain}:{resnum} "
                      f"but structure has {actual}; mutating anyway")

        print(f"  Mutating {chain}:{wt}{resnum} → {mut_three}")

        cmd.wizard("mutagenesis")
        cmd.get_wizard().set_mode(mut_three)
        cmd.get_wizard().do_select(selection)
        cmd.frame(1)
        cmd.get_wizard().apply()
        cmd.set_wizard()

    cmd.save(output_pdb, "mol")
    cmd.delete("all")
    print(f"  Saved mutant: {output_pdb}")


def main():
    if len(sys.argv) < 4:
        print("Usage: mutate_structure.py input.pdb output.pdb 'CHAIN:WTRESIDMUT,...'")
        sys.exit(1)

    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2]
    mut_str = sys.argv[3]

    try:
        apply_mutations(input_pdb, output_pdb, mut_str)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
