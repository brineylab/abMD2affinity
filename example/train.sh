#!/bin/bash

# --- job ---
#SBATCH --job-name=train_data_loader
#SBATCH --partition=hpc-mid

# --- resources ---
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# --- logs ---
#SBATCH --output=/mnt/home/%u/logs/%x_%A.out
#SBATCH --error=/mnt/home/%u/logs/%x_%A.err

# --- container ---
#SBATCH --container-image=/mnt/data/containers/deeplearning_v2026-05-26.sqsh
#SBATCH --container-mounts=/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp
#SBATCH --no-container-mount-home

# --- notifications ---
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user='slack:USER-ID' # TODO - fill with your slack UID to get notifications

set -euo pipefail

# per-run config, built and passed in by launch_experiments.py
CONFIG_FILE="${1}"

# source env file
# if you've copied the example .env file,
# this will create the work dir (JOB_WORK_DIR), set object storage / wandb keys, etc
source /mnt/home/${SLURM_JOB_USER}/.env

# cd into the work dir (on local /tmp) so training artifacts land there
cd "$JOB_WORK_DIR"

# unique port to avoid collisions if multiple jobs run concurrently
export MAIN_PROCESS_PORT=$((29500 + (${SLURM_JOB_ID:-0} + ${SLURM_ARRAY_TASK_ID:-0}) % 1000))

# train
accelerate launch \
    --multi_gpu \
    --mixed_precision bf16 \
    --main_process_port "$MAIN_PROCESS_PORT" \
    "${SLURM_SUBMIT_DIR}/pretraining.py" \
    --config "${CONFIG_FILE}"

# transfer to object storage, then delete the files locally
# function defined in the example .env file
sync_job_dir
