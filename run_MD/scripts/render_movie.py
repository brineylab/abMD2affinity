#!/usr/bin/env python3
"""
render_movie.py — Render each frame of a multi-state PDB trajectory to a PNG
with PyMOL (headless / offscreen ray tracing).

Protein is drawn as cartoon, anything non-protein still present (glycans,
ligands) as sticks — water and ions are expected to have been stripped upstream
by trjconv. Frames are written as <outdir>/frame_0001.png ... for ffmpeg to
encode into a movie.

Usage:
    python render_movie.py trajectory.pdb outdir [--width 800] [--height 600]
"""

import argparse
import sys
from pathlib import Path


def render(traj_pdb: str, outdir: str, width: int, height: int) -> None:
    import pymol
    from pymol import cmd

    Path(outdir).mkdir(parents=True, exist_ok=True)

    pymol.finish_launching(["pymol", "-cq"])
    cmd.reinitialize()
    cmd.load(traj_pdb, "mov")

    n = cmd.count_states("mov")
    if n == 0:
        raise ValueError(f"No states loaded from {traj_pdb}")
    print(f"[render_movie] {n} frames from {traj_pdb}")

    cmd.hide("everything")
    cmd.show("cartoon", "polymer.protein")
    cmd.show("sticks", "not polymer.protein")   # glycans / ligands, if any
    cmd.util.cbc()                               # color by chain
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("cartoon_transparency", 0.0)
    cmd.viewport(width, height)
    cmd.orient()                                 # fixed view from frame 1

    for i in range(1, n + 1):
        cmd.set("state", i)
        cmd.png(f"{outdir}/frame_{i:04d}.png",
                width=width, height=height, ray=1, dpi=150)
        if i % 25 == 0 or i == n:
            print(f"[render_movie] rendered {i}/{n}", flush=True)

    cmd.delete("all")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traj_pdb")
    ap.add_argument("outdir")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()
    try:
        render(args.traj_pdb, args.outdir, args.width, args.height)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
