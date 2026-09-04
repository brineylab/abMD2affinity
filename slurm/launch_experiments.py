#!/usr/bin/env python
"""Submit production-MD jobs (slurm/run_md.sh), one per prepped system,
keeping at most --cap jobs in the Slurm queue at a time.

    python slurm/launch_experiments.py [systems.txt] \
        [--cap 98] [--interval 300] [--dry-run]

Run this on the LOGIN/submit node (it only needs `sbatch`/`squeue` + the
Python stdlib — not the container). The work list is every system that has
finished preprocessing (results/preprocessing/<system>/npt.gro), so it stays
in sync with whatever the pipeline has actually produced. To submit only a
subset, pass a file with one system name per line.

For each system, once a queue slot frees up, it submits slurm/run_md.sh for
it. A marker file results/launch/submitted_<system>.log is written at
submission and is the "already launched, skip it" flag, so re-running resumes
where it left off (delete a marker to force a re-submit).

The script polls squeue every --interval seconds, topping the queue back up to
--cap, until every system has been submitted. Only jobs whose name starts with
the 'abmd_' prefix count toward the cap, so unrelated jobs of yours don't
crowd it.

--dry-run lists what it would submit (builds no markers, submits nothing).
"""

import argparse
import getpass
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUN_SCRIPT = os.path.join(HERE, "run_md.sh")
MARKER_DIR = os.path.join(REPO, "results", "launch")
PRE_DIR = os.path.join(REPO, "results", "preprocessing")
JOB_PREFIX = "abmd_"   # only jobs whose name starts with this count toward --cap


def prepped_systems() -> list[str]:
    """Systems with a finished preprocessing dir (npt.gro present)."""
    out = []
    if os.path.isdir(PRE_DIR):
        for name in sorted(os.listdir(PRE_DIR)):
            if os.path.isfile(os.path.join(PRE_DIR, name, "npt.gro")):
                out.append(name)
    return out


def named_systems(path) -> list[str]:
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def queued_count() -> int:
    """Number of my PENDING/RUNNING/CONFIGURING jobs named with JOB_PREFIX."""
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h",
         "-t", "PENDING,RUNNING,CONFIGURING", "-o", "%j"],
        check=True, stdout=subprocess.PIPE, universal_newlines=True,
    ).stdout
    return sum(1 for name in out.split() if name.startswith(JOB_PREFIX))


def submit(system: str) -> str:
    cmd = ["sbatch", f"--job-name={JOB_PREFIX}{system}", RUN_SCRIPT, system]
    out = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True,
    ).stdout.strip()
    return out.split()[-1]  # "Submitted batch job 12345"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("systems_path", nargs="?", default=None,
                    help="optional file with one system name per line "
                         "(default: every prepped system under "
                         "results/preprocessing/)")
    ap.add_argument("--cap", type=int, default=98,
                    help="max jobs in the queue at once (default: 98)")
    ap.add_argument("--interval", type=int, default=300,
                    help="seconds between queue-refill polls (default: 300)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be submitted; submit nothing")
    args = ap.parse_args()

    # GPU env for enroot's NVIDIA hook. These must be present in each job's
    # environment at *container start* so the hook injects the CUDA driver;
    # setting them inside run_md.sh would be too late (that runs inside the
    # already-started container). sbatch inherits this process's environment
    # (--export=ALL), so setting them here propagates to every job we submit.
    # setdefault: respect an explicit override from the calling shell.
    os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
    os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

    systems = (named_systems(args.systems_path) if args.systems_path
               else prepped_systems())
    if not systems:
        sys.exit("No systems found — run the pipeline's preprocessing first "
                 f"(nothing with npt.gro under {PRE_DIR}).")
    os.makedirs(MARKER_DIR, exist_ok=True)

    # Work list: systems not already submitted and whose prepped inputs exist.
    todo = []
    for system in systems:
        if os.path.exists(os.path.join(MARKER_DIR, f"submitted_{system}.log")):
            print(f"skip     {system} (already submitted)")
            continue
        npt = os.path.join(PRE_DIR, system, "npt.gro")
        if not os.path.exists(npt):
            print(f"MISSING  {system}: no {npt} — not prepped, skipping",
                  file=sys.stderr)
            continue
        todo.append(system)

    if args.dry_run:
        for system in todo:
            print(f"submit   {system} (dry-run)")
        print(f"\n{len(todo)} systems would be submitted (cap={args.cap})")
        return

    print(f"{len(todo)} systems to submit; cap={args.cap}, "
          f"interval={args.interval}s")

    while todo:
        free = args.cap - queued_count()
        while free > 0 and todo:
            system = todo.pop(0)
            job_id = submit(system)
            with open(os.path.join(MARKER_DIR, f"submitted_{system}.log"), "w") as fh:
                fh.write(f"system: {system}\n")
                fh.write(f"submitted: {datetime.now().isoformat()}\n")
                fh.write(f"job: {job_id}\n")
            print(f"submit   {system} -> job {job_id}  ({len(todo)} left)")
            free -= 1
        if todo:
            print(f"[{datetime.now():%H:%M:%S}] queue at "
                  f"{queued_count()}/{args.cap}; sleeping {args.interval}s, "
                  f"{len(todo)} left")
            time.sleep(args.interval)

    print("all systems submitted")


if __name__ == "__main__":
    main()
