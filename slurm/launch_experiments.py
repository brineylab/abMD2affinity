#!/usr/bin/env python
"""Submit production-MD jobs, one per row in a mutants TSV, keeping at most
--cap jobs in the Slurm queue at once. Run on the login/submit node.

Default (fresh): submit a 100 ns run (slurm/run_md.sh) for each row whose
npt.gro exists, skipping rows already marked in results/launch/. One pass, then
exit.

--extend: continue each system's existing trajectory (in object storage) up to
--target-ns via slurm/run_md_extend.sh, babysitting the queue and resubmitting
until every system writes its results/launch_extend/done/<run>.log marker.
Needs s5cmd + the storage keys for the pre-check; systems whose 100 ns run
finished (md.gro present) are submitted before partial ones.

    python slurm/launch_experiments.py [data/completed_mutants.tsv] [--cap N]
    python slurm/launch_experiments.py --extend [--target-ns 500] [--completed-only]

--dry-run reports what it would do and submits nothing.
"""

import argparse
import csv
import getpass
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUN_SCRIPT = os.path.join(HERE, "run_md.sh")
RUN_EXTEND_SCRIPT = os.path.join(HERE, "run_md_extend.sh")
MARKER_DIR = os.path.join(REPO, "results", "launch")
PRE_DIR = os.path.join(REPO, "results", "preprocessing")
DONE_DIR = os.path.join(REPO, "results", "launch_extend", "done")
JOB_PREFIX = "abmd_"   # only jobs named with this count toward --cap

# Object storage. The extend pre-check only lists (read-only), so default to the
# externally reachable endpoint; override with OBJ_ENDPOINT. Unreachable =>
# pre-check is skipped and every system is submitted (jobs rediscover at runtime).
OBJ_ENDPOINT = os.environ.get("OBJ_ENDPOINT", "https://cwobject.com")
OBJ_BUCKET = os.environ.get("OBJ_BUCKET", "brineylab-us-east")


def mutation_tag(mut):
    """Sanitise a mutation into its dir-name form (matches build_manifest)."""
    tag = mut.strip().replace(":", "-").replace(",", "_").replace(" ", "")
    return re.sub(r"[^A-Za-z0-9._-]", "", tag)


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def dedup_runs(rows):
    """Ordered, deduped '<pdb>_<tag>' run names from the TSV rows."""
    runs, seen = [], set()
    for row in rows:
        run = f"{row['pdb_id'].strip()}_{mutation_tag(row['mutant'])}"
        if run not in seen:
            seen.add(run)
            runs.append(run)
    return runs


def queued_names():
    """My PENDING/RUNNING/CONFIGURING abmd_ job names; empty set if squeue absent."""
    try:
        out = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h",
             "-t", "PENDING,RUNNING,CONFIGURING", "-o", "%j"],
            check=True, stdout=subprocess.PIPE, universal_newlines=True,
        ).stdout
    except FileNotFoundError:
        print("WARN: squeue not found; assuming empty queue", file=sys.stderr)
        return set()
    return {n for n in out.split() if n.startswith(JOB_PREFIX)}


def submit(run, target_ns=None):
    """sbatch run_md.sh (fresh) or run_md_extend.sh (extend); return the job id."""
    pdb, tag = run.split("_", 1)
    cmd = ["sbatch", f"--job-name={JOB_PREFIX}{run}"]
    cmd += [RUN_SCRIPT, pdb, tag] if target_ns is None \
        else [RUN_EXTEND_SCRIPT, pdb, tag, str(target_ns)]
    out = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True,
    ).stdout.strip()
    return out.split()[-1]  # "Submitted batch job 12345"


def is_done(run):
    return os.path.exists(os.path.join(DONE_DIR, f"{run}.log"))


# --- extend-mode storage pre-check -----------------------------------------

def _s5_ls(pattern):
    """`s5cmd ls`: lines on success, [] for an empty glob, None if unreachable
    (an empty glob and a real outage both exit non-zero, so we split on stderr)."""
    try:
        out = subprocess.run(
            ["s5cmd", "--endpoint-url", OBJ_ENDPOINT, "ls", pattern],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        )
    except FileNotFoundError:
        print("WARN: s5cmd not found; skipping storage pre-check", file=sys.stderr)
        return None
    if out.returncode != 0:
        if "no object found" in out.stderr:
            return []
        print("WARN: object-storage list failed; skipping storage pre-check",
              file=sys.stderr)
        return None
    return out.stdout.splitlines()


def storage_state():
    """(latest, completed) or None if storage is unreachable.
      latest    : {sys: highest jobid with an md.cpt} — run to resume from.
      completed : set of sys with an md.gro (100 ns finished normally); None if
                  the md.gro listing was inconclusive (treat completion as unknown).
    """
    base = f"s3://{OBJ_BUCKET}/{getpass.getuser()}"
    cpt = _s5_ls(f"{base}/abmd_*/md.cpt")
    if cpt is None:
        return None
    latest = defaultdict(lambda: -1)
    rx_cpt = re.compile(r"abmd_(?P<sys>.+)_(?P<jobid>\d+)/md\.cpt\s*$")
    for line in cpt:
        m = rx_cpt.search(line.strip())
        if m and int(m.group("jobid")) > latest[m.group("sys")]:
            latest[m.group("sys")] = int(m.group("jobid"))

    gro = _s5_ls(f"{base}/abmd_*/md.gro")
    if gro is None:
        completed = None
    else:
        rx_gro = re.compile(r"abmd_(?P<sys>.+)_\d+/md\.gro\s*$")
        completed = {m.group("sys") for line in gro
                     if (m := rx_gro.search(line.strip()))}
    return dict(latest), completed


# --- modes ------------------------------------------------------------------

def run_fresh(args, runs):
    """Submit a fresh 100 ns run_md.sh per system, once, up to --cap in flight."""
    os.makedirs(MARKER_DIR, exist_ok=True)
    todo = []
    for run in runs:
        pdb = run.split("_", 1)[0]
        if os.path.exists(os.path.join(MARKER_DIR, f"submitted_{run}.log")):
            print(f"skip     {run} (already submitted)")
            continue
        npt = os.path.join(PRE_DIR, pdb, run, "npt.gro")
        if not os.path.exists(npt):
            print(f"MISSING  {run}: no {npt} — not prepped, skipping", file=sys.stderr)
            continue
        todo.append(run)

    if args.dry_run:
        for run in todo:
            print(f"submit   {run} (dry-run)")
        print(f"\n{len(todo)} systems would be submitted (cap={args.cap})")
        return

    print(f"{len(todo)} systems to submit; cap={args.cap}, interval={args.interval}s")
    while todo:
        free = args.cap - len(queued_names())
        while free > 0 and todo:
            run = todo.pop(0)
            job_id = submit(run)
            with open(os.path.join(MARKER_DIR, f"submitted_{run}.log"), "w") as fh:
                fh.write(f"run: {run}\nsubmitted: {datetime.now().isoformat()}\n"
                         f"job: {job_id}\n")
            print(f"submit   {run} -> job {job_id}  ({len(todo)} left)")
            free -= 1
        if todo:
            print(f"[{datetime.now():%H:%M:%S}] queue full; sleeping {args.interval}s, "
                  f"{len(todo)} left")
            time.sleep(args.interval)
    print("all systems submitted")


def run_extend(args, runs):
    """Extend each system to --target-ns, resubmitting until its done marker
    appears (or --max-attempts is hit). Completed 100 ns runs go first."""
    os.makedirs(DONE_DIR, exist_ok=True)
    state = None if args.no_check_storage else storage_state()
    latest, completed = (None, None) if state is None else state
    active = queued_names()

    def in_queue(run):
        return f"{JOB_PREFIX}{run}" in active

    ready, partial = [], []
    for run in runs:
        if is_done(run):
            print(f"done     {run} (already at {args.target_ns} ns)")
            continue
        if latest is not None and run not in latest:
            print(f"MISSING  {run}: no md.cpt in storage — skipping", file=sys.stderr)
            continue
        src = f" (from abmd_{run}_{latest[run]})" if latest else ""
        queued = " [in queue]" if in_queue(run) else ""
        if completed is None or run in completed:   # unknown completion => ready
            print(f"extend   {run}{src}{queued}")
            ready.append(run)
        elif args.completed_only:
            print(f"PARTIAL  {run}{src}: 100 ns not finished, skipping", file=sys.stderr)
        else:
            print(f"extend   {run}{src} [partial]{queued}")
            partial.append(run)

    eligible = ready + partial   # prefer completed runs
    if args.dry_run:
        n_queued = sum(1 for r in eligible if in_queue(r))
        print(f"\n{len(eligible)} systems would be extended to {args.target_ns} ns "
              f"(cap={args.cap}); {n_queued} already in queue")
        return
    if not eligible:
        print("nothing to extend")
        return

    print(f"{len(eligible)} systems to extend to {args.target_ns} ns; cap={args.cap}, "
          f"interval={args.interval}s{', single wave' if args.once else ''}")
    attempts = defaultdict(int)
    while True:
        active = queued_names()
        pending = [r for r in eligible if not is_done(r)
                   and f"{JOB_PREFIX}{r}" not in active
                   and attempts[r] < args.max_attempts]
        free = args.cap - len(active)
        while free > 0 and pending:
            run = pending.pop(0)
            job_id = submit(run, args.target_ns)
            attempts[run] += 1
            active.add(f"{JOB_PREFIX}{run}")
            tail = f" (attempt {attempts[run]})" if attempts[run] > 1 else ""
            print(f"submit   {run} -> job {job_id}{tail}")
            free -= 1

        if args.once:
            print("single wave submitted (--once); exiting")
            return

        remaining = [r for r in eligible if not is_done(r)]
        if not remaining:
            print(f"all systems reached {args.target_ns} ns")
            return
        stuck = [r for r in remaining if attempts[r] >= args.max_attempts
                 and f"{JOB_PREFIX}{r}" not in active]
        if len(stuck) == len(remaining):
            print(f"stopping: {len(stuck)} systems not done after {args.max_attempts} "
                  f"attempts: {', '.join(stuck)}", file=sys.stderr)
            return
        print(f"[{datetime.now():%H:%M:%S}] queue at {len(active)}/{args.cap}; "
              f"{len(remaining)} not yet at {args.target_ns} ns; sleeping {args.interval}s")
        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments_path", nargs="?", default="data/completed_mutants.tsv",
                    help="mutants TSV (default: data/completed_mutants.tsv)")
    ap.add_argument("--cap", type=int, default=98, help="max queued jobs (default: 98)")
    ap.add_argument("--interval", type=int, default=300,
                    help="seconds between queue-refill polls (default: 300)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be submitted; submit nothing")
    ext = ap.add_argument_group("extend mode")
    ext.add_argument("--extend", action="store_true",
                     help="continue existing trajectories to --target-ns instead of "
                          "starting fresh 100 ns runs")
    ext.add_argument("--target-ns", type=int, default=500,
                     help="total length to reach, ns (default: 500)")
    ext.add_argument("--max-attempts", type=int, default=1,
                     help="max resubmits per system before giving up (default: 1)")
    ext.add_argument("--once", action="store_true",
                     help="submit a single wave and exit")
    ext.add_argument("--completed-only", action="store_true",
                     help="skip systems whose 100 ns run didn't finish (no md.gro)")
    ext.add_argument("--no-check-storage", action="store_true",
                     help="don't pre-list storage; submit and let jobs discover")
    args = ap.parse_args()

    # GPU env for enroot's NVIDIA hook — must exist at container start; sbatch
    # (--export=ALL) carries these into each job. setdefault: respect overrides.
    os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
    os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

    runs = dedup_runs(read_rows(args.experiments_path))
    (run_extend if args.extend else run_fresh)(args, runs)


if __name__ == "__main__":
    main()
