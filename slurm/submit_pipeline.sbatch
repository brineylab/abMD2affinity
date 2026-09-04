#!/bin/bash
#
# submit_pipeline.sbatch — run the MD2affinity pipeline via Snakemake's Slurm
# executor plugin (snakemake-executor-plugin-slurm).
#
# Model: this sbatch launches a lightweight Snakemake *controller* (1 CPU, small
# memory). The controller does NOT run any science itself — it submits each rule
# instance as its own Slurm job with `sbatch` and waits. `--jobs 80` caps the
# number of concurrently submitted/running jobs at 80. The GPU-bound rules
# (openmm_minimize, production_md) each request 1 GPU via `--gres=gpu:1`.
#
# This is the setup for fanning production MD across a cluster (e.g. running
# production-only against a subset of systems via --config structures_file=...).
#
# SITE-SPECIFIC — adjust before use:
#   * SBATCH --partition below (controller queue) and slurm_partition= (job queue)
#   * runtime / mem_mb defaults per rule (minutes / MB)
#   * The GPU gres string if your cluster names GPUs differently (e.g. gpu:a100:1)
#   * Compute nodes must have gmx / pdb2pqr / the python envs available. If the
#     pipeline tools only exist in the container, add per-job containerisation
#     (e.g. Snakemake `--use-apptainer` with a `container:` directive, or module
#     loads) — the Slurm executor runs each rule's shell on a bare compute node.
#
# Usage: sbatch slurm/submit_pipeline.sbatch [extra snakemake targets/args...]
#   e.g. production-only against a specific structures list:
#     sbatch slurm/submit_pipeline.sbatch --config structures_file=structures_abbind.yaml

#SBATCH --job-name=md2affinity-ctl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

snakemake \
    --configfile config.yaml \
    --executor slurm \
    --jobs 80 \
    --default-resources \
        slurm_partition="gpu" \
        runtime=1440 \
        mem_mb=16000 \
    --set-resources \
        openmm_minimize:slurm_extra="'--gres=gpu:1'" \
        production_md:slurm_extra="'--gres=gpu:1'" \
        production_md:runtime=2880 \
    --rerun-incomplete \
    --keep-going \
    "$@"
