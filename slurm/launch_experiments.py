#!/usr/bin/env python
"""Submit production-MD + movie jobs (slurm/run_md.sh), one per row in a mutants
TSV, keeping at most --cap jobs in the Slurm queue at a time.

    python slurm/launch_experiments.py [data/completed_mutants.tsv] \
        [--cap 80] [--interval 60] [--dry-run]

Run this on the LOGIN/submit node (it only needs `sbatch`/`squeue` + the Python
stdlib — not the container). For each TSV row it derives the run name
'<pdb>_<tag>' (tag = the same ':'->'-', ','->'_' sanitisation the Snakefile /
build_manifest use), then, once a queue slot frees up, submits slurm/run_md.sh
for it. A marker file results/launch/submitted_<run>.log is written at
submission and is the "already launched, skip it" flag, so re-running resumes
where it left off (delete a marker to force a re-submit).

The script polls squeue every --interval seconds, topping the queue back up to
--cap, until every row has been submitted. Only jobs whose name starts with the
'abmd_' prefix count toward the cap, so unrelated jobs of yours don't crowd it.

--dry-run lists what it would submit (builds no markers, submits nothing).
"""

import argparse
import csv
import getpass
import os
import re
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


def mutation_tag(mut: str) -> str:
    """Same sanitisation as build_manifest.mutation_tag (dir-name form)."""
    tag = mut.strip().replace(":", "-").replace(",", "_").replace(" ", "")
    return re.sub(r"[^A-Za-z0-9._-]", "", tag)


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def queued_count() -> int:
    """Number of my PENDING/RUNNING/CONFIGURING jobs named with JOB_PREFIX."""
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h",
         "-t", "PENDING,RUNNING,CONFIGURING", "-o", "%j"],
        check=True, stdout=subprocess.PIPE, universal_newlines=True,
    ).stdout
    return sum(1 for name in out.split() if name.startswith(JOB_PREFIX))


def submit(pdb: str, tag: str) -> str:
    run = f"{pdb}_{tag}"
    cmd = ["sbatch", f"--job-name={JOB_PREFIX}{run}", RUN_SCRIPT, pdb, tag]
    out = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True,
    ).stdout.strip()
    return out.split()[-1]  # "Submitted batch job 12345"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments_path", nargs="?",
                    default="data/completed_mutants.tsv",
                    help="mutants TSV (default: data/completed_mutants.tsv)")
    ap.add_argument("--cap", type=int, default=80,
                    help="max jobs in the queue at once (default: 80)")
    ap.add_argument("--interval", type=int, default=60,
                    help="seconds between queue-refill polls (default: 60)")
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

    rows = read_rows(args.experiments_path)
    os.makedirs(MARKER_DIR, exist_ok=True)

    # Work list: rows not already submitted and whose prepped inputs exist.
    todo = []
    for row in rows:
        pdb = row["pdb_id"].strip()
        tag = mutation_tag(row["mutant"])
        run = f"{pdb}_{tag}"
        if os.path.exists(os.path.join(MARKER_DIR, f"submitted_{run}.log")):
            print(f"skip     {run} (already submitted)")
            continue
        npt = os.path.join(PRE_DIR, pdb, run, "npt.gro")
        if not os.path.exists(npt):
            print(f"MISSING  {run}: no {npt} — not prepped, skipping",
                  file=sys.stderr)
            continue
        todo.append((pdb, tag, run))

    if args.dry_run:
        for pdb, tag, run in todo:
            print(f"submit   {run} (dry-run)")
        print(f"\n{len(todo)} systems would be submitted (cap={args.cap})")
        return

    print(f"{len(todo)} systems to submit; cap={args.cap}, "
          f"interval={args.interval}s")

    while todo:
        free = args.cap - queued_count()
        while free > 0 and todo:
            pdb, tag, run = todo.pop(0)
            job_id = submit(pdb, tag)
            with open(os.path.join(MARKER_DIR, f"submitted_{run}.log"), "w") as fh:
                fh.write(f"run: {run}\n")
                fh.write(f"submitted: {datetime.now().isoformat()}\n")
                fh.write(f"job: {job_id}\n")
                fh.write(f"pdb: {pdb}\n")
                fh.write(f"tag: {tag}\n")
            print(f"submit   {run} -> job {job_id}  ({len(todo)} left)")
            free -= 1
        if todo:
            print(f"[{datetime.now():%H:%M:%S}] queue at "
                  f"{queued_count()}/{args.cap}; sleeping {args.interval}s, "
                  f"{len(todo)} left")
            time.sleep(args.interval)

    print("all systems submitted")


if __name__ == "__main__":
    main()
