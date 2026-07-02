#!/usr/bin/env python3
"""
minimize_openmm.py — GPU-accelerated vacuum energy minimization via OpenMM.

Cleans the input PDB/CIF with PDBFixer (adds missing heavy atoms, removes heterogens),
then minimizes with AMBER14 on CUDA before passing to GROMACS.

Usage:
    python minimize_openmm.py <input.pdb|input.cif> <output.pdb> [--gpu <id>]
"""

import argparse
import sys
from pathlib import Path


def minimize(input_path: str, output_path: str, gpu_id: int = 0) -> None:
    from pdbfixer import PDBFixer
    from openmm import LangevinMiddleIntegrator, Platform
    from openmm.app import (
        ForceField,
        Modeller,
        PDBFile,
        Simulation,
        HBonds,
    )
    from openmm.unit import kelvin, picosecond, picoseconds, kilojoules_per_mole

    print(f"[minimize_openmm] Input:  {input_path}")
    print(f"[minimize_openmm] Output: {output_path}")
    print(f"[minimize_openmm] GPU:    {gpu_id}")

    # --- PDBFixer: add missing heavy atoms, standardise residues ---
    fixer = PDBFixer(filename=input_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    # --- Force field: AMBER14 (matches downstream GROMACS ff) ---
    ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")

    modeller = Modeller(fixer.topology, fixer.positions)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=__import__("openmm.app", fromlist=["NoCutoff"]).NoCutoff,
        constraints=HBonds,
        hydrogenMass=None,
    )

    # Dummy integrator — only needed to build Simulation, EM doesn't use it
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 2 * picoseconds)

    platform = Platform.getPlatformByName("CUDA")
    properties = {"CudaDeviceIndex": str(gpu_id), "CudaPrecision": "mixed"}

    sim = Simulation(modeller.topology, system, integrator, platform, properties)
    sim.context.setPositions(modeller.positions)

    state = sim.context.getState(getEnergy=True)
    print(f"[minimize_openmm] Initial potential energy: {state.getPotentialEnergy().value_in_unit(kilojoules_per_mole):.2f} kJ/mol")

    BATCH = 100       # iterations per reporting window
    MAX_BATCHES = 500 # hard cap: 50 000 iterations total
    DELTA_TOL = 0.5   # kJ/mol — declare convergence when batch-to-batch |ΔE| < this

    prev_energy = None
    converged = False
    for batch in range(1, MAX_BATCHES + 1):
        sim.minimizeEnergy(tolerance=10.0, maxIterations=BATCH)
        state = sim.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        print(f"[minimize_openmm] iter {batch * BATCH:6d}: {energy:.2f} kJ/mol", flush=True)
        if prev_energy is not None and abs(prev_energy - energy) < DELTA_TOL:
            print(f"[minimize_openmm] Converged at iter {batch * BATCH} "
                  f"(|ΔE| = {abs(prev_energy - energy):.4f} kJ/mol)")
            converged = True
            break
        prev_energy = energy
    if not converged:
        print(f"[minimize_openmm] WARNING: did not converge within {MAX_BATCHES * BATCH} iterations. "
              f"Final energy: {energy:.2f} kJ/mol")

    state = sim.context.getState(getEnergy=True, getPositions=True)
    print(f"[minimize_openmm] Final potential energy:   {state.getPotentialEnergy().value_in_unit(kilojoules_per_mole):.2f} kJ/mol")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        PDBFile.writeFile(sim.topology, state.getPositions(), f)

    print(f"[minimize_openmm] Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenMM vacuum energy minimization")
    parser.add_argument("input", help="Input CIF file")
    parser.add_argument("output", help="Output PDB file")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    try:
        minimize(args.input, args.output, gpu_id=args.gpu)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
