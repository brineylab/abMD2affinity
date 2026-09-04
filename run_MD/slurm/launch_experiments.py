#!/usr/bin/env python
"""Submit production-MD jobs, one per system in the pipeline's manifest.json,
keeping at most --cap jobs in the Slurm queue at once. Run on the login/submit
node.

The manifest is written by the pipeline itself (rule write_manifest) and maps
every system to its output dir, so this launcher stays in sync with whatever
the pipeline has produced.

Default (fresh): submit a production run (slurm/run_md.sh) for every system
whose preprocessing is done ({output_dir}/preprocessing/npt.gro), skipping
systems already marked ({output_dir}/launch.submitted). One pass, then exit.

--extend: continue each system's existing trajectory (in object storage) up to
--target-ns via slurm/run_md_extend.sh, babysitting the queue and resubmitting
until every system writes its {output_dir}/extend.done marker. Needs s5cmd +
the storage keys for the pre-check; systems whose production run finished
(md.gro present) are submitted before partial ones.

    python slurm/launch_experiments.py <manifest.json> [--cap N]
    python slurm/launch_experiments.py <manifest.json> --extend [--target-ns 500]

--dry-run reports what it would do and submits nothing.
"""

import argparse
import getpass
import json
import os
import re
import struct
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUN_SCRIPT = os.path.join(HERE, "run_md.sh")
RUN_EXTEND_SCRIPT = os.path.join(HERE, "run_md_extend.sh")
JOB_PREFIX = "abmd_"   # only jobs named with this count toward --cap

# Object storage. The extend pre-check only lists (read-only), so default to the
# externally reachable endpoint; override with OBJ_ENDPOINT. Unreachable =>
# pre-check is skipped and every system is submitted (jobs rediscover at runtime).
OBJ_ENDPOINT = os.environ.get("OBJ_ENDPOINT", "https://cwobject.com")
OBJ_BUCKET = os.environ.get("OBJ_BUCKET", "brineylab-us-east")


def load_manifest(path):
    """{system: {output_dir, ...}} from the pipeline's manifest.json."""
    with open(path) as fh:
        manifest = json.load(fh)
    out = {}
    for name, entry in manifest.items():
        out[name] = entry["output_dir"]
    return out


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


def submit(system, out_dir, target_ns=None):
    """sbatch run_md.sh (fresh) or run_md_extend.sh (extend); return the job id."""
    cmd = ["sbatch", f"--job-name={JOB_PREFIX}{system}"]
    cmd += [RUN_SCRIPT, system, out_dir] if target_ns is None \
        else [RUN_EXTEND_SCRIPT, system, out_dir, str(target_ns)]
    out = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True,
    ).stdout.strip()
    return out.split()[-1]  # "Submitted batch job 12345"


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


def _cpt_time_ns(data):
    """Simulation time (ns) parsed straight from GROMACS checkpoint bytes, or
    None. A .cpt stores the step as a big-endian int64 immediately followed by
    the time `t` as a big-endian double; we don't need gmx to read them. Scan
    (rather than hardcode an offset — it shifts with the version-string length)
    for the first (step, t) pair that looks like real MD: a large step count and
    a per-step dt of ~2 fs (accept 0.1 fs .. 20 fs)."""
    for off in range(0, len(data) - 16):
        step = struct.unpack_from(">q", data, off)[0]
        if step < 1000 or step > 10**13:
            continue
        t = struct.unpack_from(">d", data, off + 8)[0]
        if 0 < t <= 2e9 and 1e-4 <= t / step <= 2e-2:
            return t / 1000.0
    return None


def checkpoint_ns(src_prefix):
    """Current simulation time (ns) of the md.cpt under s3 prefix `src_prefix`,
    or None if it can't be read (s5cmd missing, download/parse failure). Reads
    the checkpoint bytes directly (no gmx needed — the launcher runs on the
    submit node, where gmx lives only inside the container). Best-effort, used
    only to enrich --dry-run output. Downloads the small cpt to a temp file."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".cpt", delete=False)
    tmp.close()
    try:
        cp = subprocess.run(
            ["s5cmd", "--endpoint-url", OBJ_ENDPOINT, "cp",
             f"{src_prefix}/md.cpt", tmp.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        )
        if cp.returncode != 0:
            return None
        with open(tmp.name, "rb") as fh:
            return _cpt_time_ns(fh.read())
    except (FileNotFoundError, OSError):
        return None
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def storage_state():
    """(latest, completed) or None if storage is unreachable.
      latest    : {sys: highest jobid with an md.cpt} — run to resume from.
      completed : set of sys with an md.gro (production run finished normally);
                  None if the md.gro listing was inconclusive (treat completion
                  as unknown).
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

def run_fresh(args, systems):
    """Submit a fresh production run_md.sh per system, once, up to --cap in flight."""
    todo = []
    for system, out_dir in systems.items():
        marker = os.path.join(out_dir, "launch.submitted")
        if os.path.exists(marker):
            print(f"skip     {system} (already submitted)")
            continue
        npt = os.path.join(out_dir, "preprocessing", "npt.gro")
        if not os.path.exists(npt):
            print(f"MISSING  {system}: no {npt} — not prepped, skipping",
                  file=sys.stderr)
            continue
        todo.append((system, out_dir))

    if args.dry_run:
        for system, _ in todo:
            print(f"submit   {system} (dry-run)")
        print(f"\n{len(todo)} systems would be submitted (cap={args.cap})")
        return

    print(f"{len(todo)} systems to submit; cap={args.cap}, interval={args.interval}s")
    while todo:
        free = args.cap - len(queued_names())
        while free > 0 and todo:
            system, out_dir = todo.pop(0)
            job_id = submit(system, out_dir)
            with open(os.path.join(out_dir, "launch.submitted"), "w") as fh:
                fh.write(f"system: {system}\nsubmitted: {datetime.now().isoformat()}\n"
                         f"job: {job_id}\n")
            print(f"submit   {system} -> job {job_id}  ({len(todo)} left)")
            free -= 1
        if todo:
            print(f"[{datetime.now():%H:%M:%S}] queue full; sleeping {args.interval}s, "
                  f"{len(todo)} left")
            time.sleep(args.interval)
    print("all systems submitted")


def run_extend(args, systems):
    """Extend each system to --target-ns, resubmitting until its done marker
    appears (or --max-attempts is hit). Completed production runs go first."""
    state = None if args.no_check_storage else storage_state()
    latest, completed = (None, None) if state is None else state
    active = queued_names()

    def in_queue(system):
        return f"{JOB_PREFIX}{system}" in active

    def is_done(system):
        return os.path.exists(os.path.join(systems[system], "extend.done"))

    ready, partial = [], []
    for system in systems:
        if is_done(system):
            print(f"done     {system} (already at {args.target_ns} ns)")
            continue
        if latest is not None and system not in latest:
            print(f"MISSING  {system}: no md.cpt in storage — skipping",
                  file=sys.stderr)
            continue
        src = f" (from abmd_{system}_{latest[system]})" if latest else ""
        queued = " [in queue]" if in_queue(system) else ""
        if completed is None or system in completed:   # unknown completion => ready
            if not args.dry_run:   # dry-run prints a richer per-run block below
                print(f"extend   {system}{src}{queued}")
            ready.append(system)
        elif args.completed_only:
            print(f"PARTIAL  {system}{src}: production run not finished, skipping",
                  file=sys.stderr)
        else:
            if not args.dry_run:
                print(f"extend   {system}{src} [partial]{queued}")
            partial.append(system)

    eligible = ready + partial   # prefer completed runs
    if args.dry_run:
        print()
        partial_set = set(partial)
        remaining_total = 0.0
        n_unknown = 0
        for system in eligible:
            src = f"abmd_{system}_{latest[system]}" if latest else "?"
            queued = " [in queue]" if in_queue(system) else ""
            tag = " [partial]" if system in partial_set else ""
            cur = checkpoint_ns(f"s3://{OBJ_BUCKET}/{getpass.getuser()}/{src}") \
                if latest else None
            if cur is None:
                print(f"extend   {system}: from {src}, current length unknown"
                      f" -> {args.target_ns} ns{tag}{queued}")
                n_unknown += 1
            else:
                togo = max(0.0, args.target_ns - cur)
                remaining_total += togo
                print(f"extend   {system}: from {src} at {cur:.1f} ns"
                      f" -> {args.target_ns} ns ({togo:.1f} ns to go){tag}{queued}")
        n_queued = sum(1 for s in eligible if in_queue(s))
        note = f", {n_unknown} with unknown current length" if n_unknown else ""
        print(f"\n{len(eligible)} systems would be extended to {args.target_ns} ns "
              f"(cap={args.cap}); {n_queued} already in queue{note}")
        print(f"~{remaining_total:.0f} ns total remaining across "
              f"{len(eligible) - n_unknown} readable systems")
        return
    if not eligible:
        print("nothing to extend")
        return

    print(f"{len(eligible)} systems to extend to {args.target_ns} ns; cap={args.cap}, "
          f"interval={args.interval}s{', single wave' if args.once else ''}")
    attempts = defaultdict(int)
    while True:
        active = queued_names()
        pending = [s for s in eligible if not is_done(s)
                   and f"{JOB_PREFIX}{s}" not in active
                   and attempts[s] < args.max_attempts]
        free = args.cap - len(active)
        while free > 0 and pending:
            system = pending.pop(0)
            job_id = submit(system, systems[system], args.target_ns)
            attempts[system] += 1
            active.add(f"{JOB_PREFIX}{system}")
            tail = f" (attempt {attempts[system]})" if attempts[system] > 1 else ""
            print(f"submit   {system} -> job {job_id}{tail}")
            free -= 1

        if args.once:
            print("single wave submitted (--once); exiting")
            return

        remaining = [s for s in eligible if not is_done(s)]
        if not remaining:
            print(f"all systems reached {args.target_ns} ns")
            return
        stuck = [s for s in remaining if attempts[s] >= args.max_attempts
                 and f"{JOB_PREFIX}{s}" not in active]
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
    ap.add_argument("manifest", nargs="?", default="manifest.json",
                    help="manifest.json written by the pipeline's write_manifest "
                         "rule (default: manifest.json)")
    ap.add_argument("--cap", type=int, default=98, help="max queued jobs (default: 98)")
    ap.add_argument("--interval", type=int, default=300,
                    help="seconds between queue-refill polls (default: 300)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be submitted; submit nothing")
    ext = ap.add_argument_group("extend mode")
    ext.add_argument("--extend", action="store_true",
                     help="continue existing trajectories to --target-ns instead of "
                          "starting fresh production runs")
    ext.add_argument("--target-ns", type=int, default=500,
                     help="total length to reach, ns (default: 500)")
    ext.add_argument("--max-attempts", type=int, default=1,
                     help="max resubmits per system before giving up (default: 1)")
    ext.add_argument("--once", action="store_true",
                     help="submit a single wave and exit")
    ext.add_argument("--completed-only", action="store_true",
                     help="skip systems whose production run didn't finish (no md.gro)")
    ext.add_argument("--no-check-storage", action="store_true",
                     help="don't pre-list storage; submit and let jobs discover")
    args = ap.parse_args()

    # GPU env for enroot's NVIDIA hook — must exist at container start; sbatch
    # (--export=ALL) carries these into each job. setdefault: respect overrides.
    os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
    os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

    try:
        systems = load_manifest(args.manifest)
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"cannot read manifest {args.manifest}: {e}")
    if not systems:
        sys.exit(f"{args.manifest} lists no systems.")
    (run_extend if args.extend else run_fresh)(args, systems)


if __name__ == "__main__":
    main()
