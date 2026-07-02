#!/bin/bash
#
# submit_pipeline.sbatch — run the full MD2affinity pre-processing + MD
# pipeline as a single Slurm job on one multi-GPU node.
#
# Snakemake (7.x, no executor-plugin support) acts as the in-job scheduler:
# one Slurm allocation runs a Snakemake controller that fans work out across
# the CPUs/GPUs granted to this job, round-robining GPU-bound rules
# (openmm_minimize, production_md) across the 4 GPU resource pools declared
# in the Snakefile. There is no per-rule Slurm submission.
#
# Adjust --partition/--gres/--cpus-per-task/--mem/--time for your cluster,
# and keep --gres=gpu:N in sync with n_gpus in config.yaml (default 4).
#
# Usage: sbatch slurm/submit_pipeline.sbatch [extra snakemake args...]

#SBATCH --job-name=md2affinity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

snakemake \
    --configfile config.yaml \
    --cores "${SLURM_CPUS_PER_TASK:-32}" \
    --resources gpu0=1 gpu1=1 gpu2=1 gpu3=1 \
    --rerun-incomplete \
    --keep-going \
    "$@"
