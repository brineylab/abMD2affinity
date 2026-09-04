#!/usr/bin/env python
"""Extend every completed system's production MD up to a target length (default
500 ns) by continuing from the trajectory already in object storage.

    python slurm/launch_experiments_extend.py [systems.txt] \
        [--target-ns 500] [--cap 80] [--interval 60] [--once] [--dry-run]

This is the "superpowered" sibling of launch_experiments.py: instead of
starting a fresh production run from npt.gro, it finds each system's existing
trajectory in object storage (uploaded there by run_md.sh's sync_job_dir) and
submits slurm/run_md_extend.sh, which downloads it, raises the step count, and
resumes mdrun with -cpi -append so the result is one continuous <target> ns
run. The original launch_experiments.py / run_md.sh are left untouched.

Run on the LOGIN/submit node (needs sbatch/squeue, the Python stdlib, and —
for the storage pre-check — s5cmd with the object-storage keys from ~/.env
sourced into your shell). The work list defaults to every system that has
finished preprocessing (results/preprocessing/<system>/npt.gro); pass a file
with one system name per line to submit a subset. For each system it:

  * skips it if results/launch_extend/done/<system>.log exists (already at target);
  * skips it if a job named abmd_<system> is already in the queue (in flight);
  * skips it (with a warning) if no md.cpt for it exists in object storage
    (nothing to continue — it was never produced); use --no-check-storage to
    submit anyway and let the job discover it;
  * otherwise submits run_md_extend.sh once a queue slot (<= --cap) frees up,
    PREFERRING systems whose production run has completed. Completion is
    judged by md.gro (the final confout, written only when mdrun reaches
    nsteps) existing in some prefix; completed systems are submitted before
    partial ones. Pass --completed-only to skip partial (no-md.gro) runs
    entirely.

Note: the resume source is still the furthest-along md.cpt (highest jobid),
completed or not — that's what lets a partially-extended run pick up where it
left off; the md.gro check only orders which systems go first.

A 400 ns extension usually will not finish in one job's wall time. run_md_extend
uploads its checkpoint periodically and, on completion, writes the done marker;
this launcher keeps polling and RE-SUBMITS any eligible system that is neither
done nor currently queued (up to --max-attempts), and each resumed job picks up
the furthest-along checkpoint from storage. It exits when every eligible system
is done. Pass --once to submit a single wave and exit (e.g. for cron/manual
babysitting); --dry-run reports what it would do and submits nothing.
"""

import argparse
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
RUN_SCRIPT = os.path.join(HERE, "run_md_extend.sh")
DONE_DIR = os.path.join(REPO, "results", "launch_extend", "done")
PRE_DIR = os.path.join(REPO, "results", "preprocessing")
JOB_PREFIX = "abmd_"   # extend jobs share this name shape with run_md.sh so
                       # run_md_extend's storage discovery (abmd_<sys>_*) finds
                       # both the original run and its own partials.

# object-storage coordinates; override via env. Two endpoints hit the same
# store: http://cwlota.com is cluster-internal (what ~/.env's sync_job_dir uses
# to upload from compute nodes) and https://cwobject.com is externally
# reachable (works from the prep/login box). This launcher's pre-check only ever
# lists, so it defaults to the external endpoint; set OBJ_ENDPOINT to override
# (e.g. http://cwlota.com when running on the cluster submit node). If the
# endpoint is unreachable the pre-check is skipped gracefully and every system
# is submitted anyway (the job re-discovers the trajectory at runtime).
OBJ_ENDPOINT = os.environ.get("OBJ_ENDPOINT", "https://cwobject.com")
OBJ_BUCKET = os.environ.get("OBJ_BUCKET", "brineylab-us-east")


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


def queued_names():
    """Names of my PENDING/RUNNING/CONFIGURING jobs that start with JOB_PREFIX.

    Returns None (with a warning) if squeue isn't available — e.g. a --dry-run
    on a non-cluster box — so the queue pre-check degrades gracefully instead
    of crashing. Callers treat None as "queue unknown".
    """
    try:
        out = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h",
             "-t", "PENDING,RUNNING,CONFIGURING", "-o", "%j"],
            check=True, stdout=subprocess.PIPE, universal_newlines=True,
        ).stdout
    except FileNotFoundError:
        print("WARN: squeue not found; skipping queue check", file=sys.stderr)
        return None
    return {n for n in out.split() if n.startswith(JOB_PREFIX)}


def _s5_ls(pattern):
    """Run `s5cmd ls <pattern>`; return its lines, or None if storage is
    unreachable. An empty glob ('no object found') returns [] — s5cmd exits
    non-zero both for that and for a real outage, and only the former means
    "reachable, nothing there", so we must not confuse the two.
    """
    try:
        out = subprocess.run(
            ["s5cmd", "--endpoint-url", OBJ_ENDPOINT, "ls", pattern],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except FileNotFoundError:
        print("WARN: s5cmd not found; skipping storage pre-check", file=sys.stderr)
        return None
    if out.returncode != 0:
        err = out.stderr.strip()
        if "no object found" in err:
            return []
        print(f"WARN: object-storage list failed; skipping storage pre-check\n"
              f"      {err.splitlines()[-1] if err else ''}",
              file=sys.stderr)
        return None
    return out.stdout.splitlines()


def storage_state():
    """Inspect object storage. Returns (latest, completed) or None if unreachable.

      latest    : {'<system>': highest jobid whose prefix has an md.cpt} —
                  the furthest-along run to resume from.
      completed : set of systems whose production run finished normally, i.e.
                  some prefix has an md.gro (the final confout, written only
                  when mdrun reaches nsteps; a killed/requeued run has md.cpt
                  but no md.gro). None if the md.gro listing itself was
                  inconclusive, in which case callers treat completion as
                  unknown (assume ready).
    """
    base = f"s3://{OBJ_BUCKET}/{getpass.getuser()}"

    cpt_lines = _s5_ls(f"{base}/abmd_*/md.cpt")
    if cpt_lines is None:
        return None
    latest = defaultdict(lambda: -1)
    rx_cpt = re.compile(r"abmd_(?P<sys>.+)_(?P<jobid>\d+)/md\.cpt\s*$")
    for line in cpt_lines:
        m = rx_cpt.search(line.strip())
        if m:
            j = int(m.group("jobid"))
            if j > latest[m.group("sys")]:
                latest[m.group("sys")] = j

    gro_lines = _s5_ls(f"{base}/abmd_*/md.gro")
    if gro_lines is None:
        completed = None
    else:
        completed = set()
        rx_gro = re.compile(r"abmd_(?P<sys>.+)_\d+/md\.gro\s*$")
        for line in gro_lines:
            m = rx_gro.search(line.strip())
            if m:
                completed.add(m.group("sys"))
    return dict(latest), completed


def submit(system: str, target_ns: int) -> str:
    cmd = ["sbatch", f"--job-name={JOB_PREFIX}{system}",
           RUN_SCRIPT, system, str(target_ns)]
    out = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True,
    ).stdout.strip()
    return out.split()[-1]  # "Submitted batch job 12345"


def is_done(system: str) -> bool:
    return os.path.exists(os.path.join(DONE_DIR, f"{system}.log"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("systems_path", nargs="?", default=None,
                    help="optional file with one system name per line "
                         "(default: every prepped system under "
                         "results/preprocessing/)")
    ap.add_argument("--target-ns", type=int, default=500,
                    help="total trajectory length to reach, ns (default: 500)")
    ap.add_argument("--cap", type=int, default=80,
                    help="max jobs in the queue at once (default: 80)")
    ap.add_argument("--interval", type=int, default=60,
                    help="seconds between queue-refill polls (default: 60)")
    ap.add_argument("--max-attempts", type=int, default=6,
                    help="max resubmits per system before giving up (default: 6)")
    ap.add_argument("--once", action="store_true",
                    help="submit a single wave (no babysitting) and exit")
    ap.add_argument("--completed-only", action="store_true",
                    help="only extend systems whose production run finished "
                         "(md.gro present); skip partial runs entirely instead "
                         "of deprioritising them")
    ap.add_argument("--no-check-storage", action="store_true",
                    help="don't pre-list object storage; submit and let the job "
                         "discover the trajectory (or fail if none)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be submitted; submit nothing")
    args = ap.parse_args()

    # GPU env for enroot's NVIDIA hook — must be present at container start, so
    # set here (sbatch --export=ALL carries them into each job). Same as
    # launch_experiments.py. setdefault: respect an explicit shell override.
    os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
    os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

    systems = (named_systems(args.systems_path) if args.systems_path
              else prepped_systems())
    if not systems:
        sys.exit(f"No systems found (nothing with npt.gro under {PRE_DIR}).")
    os.makedirs(DONE_DIR, exist_ok=True)

    # runs we could extend, in list order (deduped)
    runs, seen = [], set()
    for system in systems:
        if system not in seen:
            seen.add(system)
            runs.append(system)

    # object-storage pre-check: which systems have a trajectory, and which have
    # a *completed* production run (md.gro present).
    state = None if args.no_check_storage else storage_state()
    latest, completed = (None, None) if state is None else state

    # queue pre-check: a system whose abmd_<system> job is already
    # PENDING/RUNNING is in flight and must not be resubmitted. The babysitting
    # loop below already enforces this per poll; snapshotting it here just lets
    # the report and --dry-run show what will actually be skipped. Kept in
    # `eligible` regardless so the loop can still resume it if it dies before
    # reaching the target.
    active = queued_names()          # set of abmd_* job names, or None if unknown

    def in_queue(run):
        return active is not None and f"{JOB_PREFIX}{run}" in active

    ready, partial = [], []      # completed production runs first, then partials
    for run in runs:
        if is_done(run):
            print(f"done     {run} (already at {args.target_ns} ns)")
            continue
        if latest is not None and run not in latest:
            print(f"MISSING  {run}: no md.cpt in object storage — not produced, "
                  f"skipping", file=sys.stderr)
            continue
        src = f" (from abmd_{run}_{latest[run]})" if latest else ""
        # unknown completion (no-check / md.gro list inconclusive) => assume ready
        is_complete = completed is None or run in completed
        queued = " [already in queue — won't resubmit]" if in_queue(run) else ""
        if is_complete:
            print(f"extend   {run}{src}{queued}")
            ready.append(run)
        elif args.completed_only:
            print(f"PARTIAL  {run}{src}: production run not finished (no md.gro), "
                  f"skipping (--completed-only)", file=sys.stderr)
        else:
            print(f"extend   {run}{src} [partial run — no md.gro]{queued}")
            partial.append(run)

    eligible = ready + partial   # prefer completed runs: submit them first

    n_queued = sum(1 for r in eligible if in_queue(r))
    if args.dry_run:
        extra = f", {len(partial)} of them from partial runs" if partial else ""
        queued_note = (f"; {n_queued} already in the queue (skipped), "
                       f"{len(eligible) - n_queued} new submission(s)"
                       if n_queued else "")
        print(f"\n{len(eligible)} systems would be extended to {args.target_ns} ns"
              f"{extra} (cap={args.cap}){queued_note}")
        return

    if not eligible:
        print("nothing to extend")
        return

    print(f"{len(eligible)} systems to extend to {args.target_ns} ns; "
          f"cap={args.cap}, interval={args.interval}s"
          f"{', single wave' if args.once else ''}")

    attempts = defaultdict(int)
    while True:
        active = queued_names() or set()   # None (squeue missing) -> treat as empty
        pending = [r for r in eligible
                   if not is_done(r) and f"{JOB_PREFIX}{r}" not in active
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

        # done when every eligible system has its done marker (or hit the cap
        # of attempts). Anything not done and not queued gets picked up next poll.
        remaining = [r for r in eligible if not is_done(r)]
        stuck = [r for r in remaining if attempts[r] >= args.max_attempts
                 and f"{JOB_PREFIX}{r}" not in active]
        if not remaining:
            print(f"all systems reached {args.target_ns} ns")
            return
        if len(stuck) == len(remaining):
            print(f"stopping: {len(stuck)} systems not done after "
                  f"{args.max_attempts} attempts: {', '.join(stuck)}",
                  file=sys.stderr)
            return

        n_active = len(active)
        print(f"[{datetime.now():%H:%M:%S}] queue at {n_active}/{args.cap}; "
              f"{len(remaining)} not yet at {args.target_ns} ns; "
              f"sleeping {args.interval}s")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
