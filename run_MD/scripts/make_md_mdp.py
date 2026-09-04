#!/usr/bin/env python3
"""
make_md_mdp.py — Write a production-MD .mdp with a configured run length.

Copies a base .mdp (mdp/md.mdp) verbatim and replaces its `nsteps` according
to --ns (nanoseconds, converted via the base file's `dt`) or --steps (a raw
step count). Exactly one of the two is required.

    python scripts/make_md_mdp.py mdp/md.mdp results/params/md.mdp --ns 100
    python scripts/make_md_mdp.py mdp/md.mdp results/params/md.mdp --steps 500000
"""

import argparse
import re
import sys
from pathlib import Path


def read_mdp_params(path: Path) -> dict[str, str]:
    """key -> value for every `key = value` line (';' starts a comment)."""
    params: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            stripped = line.split(";", 1)[0].strip()
            if "=" in stripped:
                key, val = stripped.split("=", 1)
                params[key.strip().lower()] = val.strip()
    return params


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_mdp", help="base .mdp (all other parameters copied)")
    ap.add_argument("out_mdp", help="output .mdp path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ns", type=float,
                   help="run length in nanoseconds (nsteps = ns * 1000 / dt)")
    g.add_argument("--steps", type=int, help="run length in integration steps")
    args = ap.parse_args()

    base = Path(args.base_mdp)
    params = read_mdp_params(base)

    if args.steps is not None:
        nsteps, length = args.steps, None
    else:
        if "dt" not in params:
            sys.exit(f"{base}: no 'dt' found — needed to convert --ns to steps")
        dt = float(params["dt"])
        nsteps = round(args.ns * 1000.0 / dt)
        length = args.ns

    nsteps_line = (
        f"nsteps                  = {nsteps}"
        + (f"   ; {length:g} ns at dt={dt:g}" if length is not None else "")
        + "\n"
    )

    lines = base.read_text().splitlines(keepends=True)
    out_lines, replaced = [], False
    for line in lines:
        stripped = line.split(";", 1)[0].strip()
        if re.match(r"^nsteps\s*=", stripped, flags=re.IGNORECASE):
            out_lines.append(nsteps_line)
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        out_lines.append(nsteps_line)

    out = Path(args.out_mdp)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(out_lines))

    dt_note = f" (dt={params.get('dt', '?')} ps)" if length is not None else ""
    print(f"Wrote {out}: nsteps = {nsteps}{dt_note}")


if __name__ == "__main__":
    main()
