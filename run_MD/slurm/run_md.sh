#!/bin/bash
#
# run_md.sh — production MD for ONE system, run inside the pipeline
# container. Submitted once per system by slurm/launch_experiments.py.
# Mirrors the Snakefile `production_md` rule, but as a standalone per-job
# script (no Snakemake on the compute node).
#
# Args (passed by launch_experiments.py):
#   $1  system name     e.g. 1bj1fv_WT — for job naming/logs only
#   $2  output dir      the system's directory from the manifest: contains
#                      preprocessing/ (npt.gro, npt.cpt, topol_ions.top) and
#                      MD/ (md.mdp written by the pipeline's write_md_mdp rule)
#
# grompp runs *in* the prepped dir so topol_ions.top's #includes resolve; the
# md.tpr it emits is fully self-contained, so mdrun then runs entirely on
# local /tmp scratch. sync_job_dir (from ~/.env) uploads the finished run to
# object storage and clears scratch.
#
# GPU note: the CUDA driver is injected into the container by enroot's NVIDIA
# hook, which needs NVIDIA_VISIBLE_DEVICES + NVIDIA_DRIVER_CAPABILITIES set *at
# container start*. Exporting them in this script would be too late — the whole
# script runs inside the already-started container. launch_experiments.py sets
# them in the submitting environment so sbatch (--export=ALL) carries them into
# each job; a ~/.config/enroot/environ.d/10-nvidia.env file does the same job-
# independently. Without one of these, mdrun reports "no compatible GPU".

# --- job -------------------------------------------------------------------
#SBATCH --job-name=abmd                 # overridden per-system on submit
#SBATCH --partition=rtxp6000
# --- resources -------------------------------------------------------------
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00                 # size after benchmarking ns/day
# --- logs ------------------------------------------------------------------
#SBATCH --output=/mnt/home/%u/logs/%x_%A.out
#SBATCH --error=/mnt/home/%u/logs/%x_%A.err
# --- container -------------------------------------------------------------
#SBATCH --container-image=/mnt/data/sferrier/containers/abmd2affinity2.sqsh
#SBATCH --container-mounts=/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp
#SBATCH --no-container-mount-home

set -euo pipefail

SYS="${1:?usage: run_md.sh <system> <output_dir>}"
OUTDIR="${2:?usage: run_md.sh <system> <output_dir>}"

PREDIR="${OUTDIR}/preprocessing"
MDP="${OUTDIR}/MD/md.mdp"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [ ! -f "${MDP}" ]; then
    echo "ERROR: ${MDP} not found — generate it first with:" >&2
    echo "  snakemake --configfile config.yaml ${MDP}" >&2
    exit 1
fi

# env file: creates JOB_WORK_DIR on local /tmp, sets object-storage keys, and
# defines sync_job_dir().
source "/mnt/home/${SLURM_JOB_USER}/.env"

echo "[$(date)] ${SYS}: grompp (in prepped dir, self-contained md.tpr -> scratch)"
( cd "${PREDIR}" && gmx grompp \
    -f "${MDP}" \
    -c npt.gro \
    -t npt.cpt \
    -p topol_ions.top \
    -o "${JOB_WORK_DIR}/md.tpr" \
    -maxwarn 0 )

cd "${JOB_WORK_DIR}"

echo "[$(date)] ${SYS}: production mdrun (GPU-resident)"
CPI=""; [ -f md.cpt ] && CPI="-cpi md.cpt"      # resume if this job requeued
gmx mdrun -v -deffnm md -s md.tpr ${CPI} \
    -ntmpi 1 -ntomp "${THREADS}" \
    -nb gpu -pme gpu -bonded gpu -update gpu

# Keep lightweight artifacts on shared storage for quick local inspection
# (the full run — incl. md.xtc — goes to object storage via sync_job_dir).
mkdir -p "${OUTDIR}/MD"
cp md.log "${OUTDIR}/MD/md.log"

echo "[$(date)] ${SYS}: sync_job_dir -> object storage, clear scratch"
sync_job_dir

echo "[$(date)] ${SYS}: done"
