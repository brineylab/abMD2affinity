#!/usr/bin/env python
"""Build a config per experiments row and launch any run not already launched.

    python launch_experiments.py [experiments.tsv] [--dry-run]

For each row: write results/configs/<run>.yaml if missing, then sbatch train.sh
if results/submitted_<run>.log is missing. That log (written at submission) is
the "already launched, skip it" marker and records the run name, submission
time, job id, and the full config that was submitted.

Pass --dry-run to build the configs but stop short of launching: no sbatch
call and no submit log, so the runs aren't marked as launched.
"""

import argparse
import os
import subprocess
from datetime import datetime

import make_config

BASE_CONFIG = "150M_defaults.yaml"
RESULTS_DIR = "results"

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("experiments_path", nargs="?", default="experiments.tsv",
                help="experiments file (default: experiments.tsv)")
ap.add_argument("--dry-run", action="store_true",
                help="Build configs but don't submit jobs or write submit logs.")
args = ap.parse_args()
experiments_path = args.experiments_path
dry_run = args.dry_run

rows = make_config._read_rows(experiments_path)

for row in rows:
    run_name = row["run_name"]
    cfg = os.path.join(RESULTS_DIR, "configs", f"{run_name}.yaml")
    submit_log = os.path.join(RESULTS_DIR, f"submitted_{run_name}.log")

    # build the config if it isn't there yet
    if not os.path.exists(cfg):
        os.makedirs(os.path.dirname(cfg), exist_ok=True)
        with open(cfg, "w") as out:
            out.write(make_config.render(BASE_CONFIG, experiments_path, run_name))
        print(f"build  {run_name} -> {cfg}")

    # skip runs we've already launched (the submit log is the marker)
    if os.path.exists(submit_log):
        print(f"skip   {run_name}")
        continue

    sbatch_cmd = ["sbatch", f"--job-name=ablm_{run_name}",
                  "train.sh", os.path.abspath(cfg)]

    if dry_run:
        print(f"submit {run_name} (dry-run): {' '.join(sbatch_cmd)}")
        continue

    # submit, then write the submit log so we don't relaunch it next time
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = subprocess.run(
        sbatch_cmd, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
    ).stdout.strip()
    job_id = out.split()[-1]  # "Submitted batch job 12345"

    with open(cfg) as fh:
        cfg_text = fh.read()
    with open(submit_log, "w") as fh:
        fh.write(f"run_name: {run_name}\n")
        fh.write(f"submitted: {datetime.now().isoformat()}\n")
        fh.write(f"job: {job_id}\n")
        fh.write(f"config: {cfg}\n")
        fh.write("---\n")
        fh.write(cfg_text)
    print(f"submit {run_name} -> job {job_id}")
