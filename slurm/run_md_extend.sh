#!/bin/bash
#
# run_md_extend.sh — EXTEND one system's production MD up to a target length
# (default 500 ns) by continuing from whatever trajectory already exists in
# object storage. Standalone per-job script, submitted once per system by
# slurm/launch_experiments_extend.py. Does NOT touch the original run_md.sh /
# production_md path — it starts from the finished (or partially finished)
# trajectory that run_md.sh already uploaded.
#
# Args (passed by launch_experiments_extend.py):
#   $1  pdb_id         e.g. 1bj1fv
#   $2  mutation tag   e.g. V-F17A   (the sanitised {mut} used in dir names)
#   $3  target_ns      (optional) total simulation length in ns; default 500
#
# What it does, entirely on local /tmp scratch:
#   1. Discover the most-advanced trajectory for this system in object storage.
#      run_md.sh uploads each run to
#        s3://<bucket>/<user>/abmd_<pdb>_<tag>_<jobid>/{md.tpr,md.cpt,md.xtc,...}
#      and this script uploads its own partials the same way, so the prefix with
#      the highest <jobid> that has an md.cpt is the furthest-along run. We pull
#      md.{tpr,cpt,xtc,edr,log} from it.
#   2. `gmx convert-tpr -until <target_ns*1000> ps` to raise the step count.
#   3. `gmx mdrun -cpi md.cpt -deffnm md -append` continues the SAME output
#      files (append), so the result is one continuous 500 ns md.xtc/md.edr.
#   4. Sync scratch to THIS job's own prefix periodically (so a wall-time kill
#      loses at most one sync interval) and once more at exit. A 400 ns
#      extension will not fit in one job's wall time; when the launcher
#      re-submits, step 1 finds this job's partial (higher jobid) and resumes.
#   5. On reaching the target, write results/launch_extend/done/<sys>.log on
#      shared storage — the launcher's "done, don't resubmit" flag.
#
# GPU note: identical to run_md.sh — the CUDA driver is injected by enroot's
# NVIDIA hook, which needs NVIDIA_VISIBLE_DEVICES + NVIDIA_DRIVER_CAPABILITIES
# set at container start. launch_experiments_extend.py sets them in the
# submitting environment so sbatch (--export=ALL) carries them into each job.

# --- job -------------------------------------------------------------------
#SBATCH --job-name=abmd_extend          # overridden per-system on submit
#SBATCH --partition=rtxp6000
# --- resources -------------------------------------------------------------
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00                 # 400 ns rarely fits — job self-resumes
#SBATCH --signal=B:TERM@180             # 3 min SIGTERM warning -> final sync
# --- logs ------------------------------------------------------------------
#SBATCH --output=/mnt/home/%u/logs/%x_%A.out
#SBATCH --error=/mnt/home/%u/logs/%x_%A.err
# --- container -------------------------------------------------------------
#SBATCH --container-image=/mnt/data/sferrier/containers/abmd2affinity.sqsh
#SBATCH --container-mounts=/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp
#SBATCH --no-container-mount-home

set -euo pipefail

PDB="${1:?usage: run_md_extend.sh <pdb_id> <mutation_tag> [target_ns]}"
TAG="${2:?usage: run_md_extend.sh <pdb_id> <mutation_tag> [target_ns]}"
TARGET_NS="${3:-500}"
SYS="${PDB}_${TAG}"
UNTIL_PS=$(( TARGET_NS * 1000 ))

# object-storage coordinates; override via env. Defaults to http://cwlota.com,
# the cluster-internal endpoint that ~/.env's sync_job_dir uploads to (proven to
# resolve on compute nodes). https://cwobject.com is the same store via the
# externally reachable endpoint — set OBJ_ENDPOINT to it if cwlota.com does not
# resolve from where this job runs.
OBJ_ENDPOINT="${OBJ_ENDPOINT:-http://cwlota.com}"
OBJ_BUCKET="${OBJ_BUCKET:-brineylab-us-east}"

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

# env file: creates JOB_WORK_DIR on local /tmp, sets object-storage keys, and
# defines sync_job_dir() (which also removes scratch, so we don't call it until
# the very end).
source "/mnt/home/${SLURM_JOB_USER}/.env"

USER_PREFIX="s3://${OBJ_BUCKET}/${SLURM_JOB_USER}"
SELF_PREFIX="${USER_PREFIX}/${JOB_DIR}"

s5() { s5cmd --endpoint-url "${OBJ_ENDPOINT}" "$@"; }

# Push current scratch to THIS job's own prefix (checkpoint upload). Same shape
# as sync_job_dir but without deleting scratch — safe to call repeatedly.
push_checkpoint() {
    echo "[$(date)] ${SYS}: sync scratch -> ${SELF_PREFIX}/"
    s5 sync --exclude "*/.*" --exclude ".*" "${JOB_WORK_DIR}/" "${SELF_PREFIX}/" \
        || echo "[$(date)] ${SYS}: WARN checkpoint sync failed (will retry)"
}

cd "${JOB_WORK_DIR}"

# --- 1. discover the furthest-along existing trajectory --------------------
# Every run of this system (original run_md.sh run + any prior extend attempts)
# lives at abmd_<sys>_<jobid>/. The largest jobid that has an md.cpt is the
# furthest along. Skip our own (brand-new, empty) prefix.
echo "[$(date)] ${SYS}: locating latest trajectory under ${USER_PREFIX}/"
BEST_DIR=""; BEST_ID=-1
while read -r line; do
    [ -n "$line" ] || continue
    key="${line##* }"                       # last field = full s3://.../md.cpt
    dir="${key%/md.cpt}"; dir="${dir##*/}"  # -> abmd_<sys>_<jobid>
    id="${dir##*_}"
    case "$id" in ''|*[!0-9]*) continue ;; esac
    [ "$dir" = "$JOB_DIR" ] && continue
    if [ "$id" -gt "$BEST_ID" ]; then BEST_ID="$id"; BEST_DIR="$dir"; fi
done < <(s5 ls "${USER_PREFIX}/abmd_${SYS}_*/md.cpt" 2>/dev/null || true)

if [ -z "$BEST_DIR" ]; then
    echo "[$(date)] ${SYS}: ERROR no md.cpt found under ${USER_PREFIX}/abmd_${SYS}_* — nothing to continue" >&2
    exit 1
fi
SRC="${USER_PREFIX}/${BEST_DIR}"
echo "[$(date)] ${SYS}: continuing from ${SRC}/"

# Pull the files needed to extend + append. md.tpr/md.cpt are required;
# md.xtc/md.edr/md.log let mdrun append into one continuous trajectory.
s5 cp "${SRC}/md.tpr" .
s5 cp "${SRC}/md.cpt" .
for f in md.xtc md.edr md.log; do
    s5 cp "${SRC}/${f}" . 2>/dev/null || echo "[$(date)] ${SYS}: note ${f} absent in source"
done

# --- 2. raise the step count to the target length --------------------------
echo "[$(date)] ${SYS}: convert-tpr -> ${TARGET_NS} ns (${UNTIL_PS} ps)"
gmx convert-tpr -s md.tpr -until "${UNTIL_PS}" -o md_ext.tpr
mv -f md_ext.tpr md.tpr

# --- 3. resume mdrun, appending into the existing md.* files ---------------
# Periodic checkpoint uploader + exit trap so a wall-time SIGTERM still flushes
# the latest checkpoint before the node is reclaimed.
( while true; do sleep 1800; push_checkpoint; done ) &
SYNC_PID=$!
trap 'kill "${SYNC_PID}" 2>/dev/null || true; push_checkpoint' EXIT

echo "[$(date)] ${SYS}: production mdrun (GPU-resident) -> ${TARGET_NS} ns"
set +e
gmx mdrun -v -deffnm md -s md.tpr -cpi md.cpt -append \
    -ntmpi 1 -ntomp "${THREADS}" \
    -nb gpu -pme gpu -bonded gpu -update gpu
RC=$?
set -e

# stop the periodic uploader; the EXIT trap does the final push_checkpoint
kill "${SYNC_PID}" 2>/dev/null || true
trap - EXIT
push_checkpoint

# Keep a lightweight copy of the log on shared storage for quick inspection.
mkdir -p "${PROJECT_DIR}/results/launch_extend/logs"
cp -f md.log "${PROJECT_DIR}/results/launch_extend/logs/${SYS}.md.log" || true

if [ "$RC" -ne 0 ]; then
    echo "[$(date)] ${SYS}: mdrun exited ${RC} (likely wall-time / requeue); partial synced to ${SELF_PREFIX}/, re-submit to resume" >&2
    exit "$RC"
fi

# --- 4. reached target: mark done on shared storage ------------------------
mkdir -p "${PROJECT_DIR}/results/launch_extend/done"
{
    echo "sys: ${SYS}"
    echo "target_ns: ${TARGET_NS}"
    echo "finished: $(date -Iseconds)"
    echo "final_prefix: ${SELF_PREFIX}/"
    echo "resumed_from: ${SRC}/"
} > "${PROJECT_DIR}/results/launch_extend/done/${SYS}.log"

echo "[$(date)] ${SYS}: reached ${TARGET_NS} ns; full trajectory at ${SELF_PREFIX}/"
