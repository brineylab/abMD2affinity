#!/bin/bash
# run_test_1ns.sh — smoke-test the pipeline: the structures listed in
# structures.yaml (one system by default), 1 ns production MD, inside the
# pipeline container.
#
# GPU is wired into rootless podman by hand (no nvidia-container-toolkit here):
# bind the host driver .so's + nvidia device nodes, then ldconfig in-container.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-abmd2affinity:cuda129}"
CORES="${CORES:-8}"

GPU_DEVS=(--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm
          --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset)
GPU_LIBS=()
for f in /usr/lib/x86_64-linux-gnu/libcuda.so* \
         /usr/lib/x86_64-linux-gnu/libnvidia-ml.so* \
         /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so*; do
    [ -e "$f" ] && GPU_LIBS+=(-v "$f:$f:ro")
done

exec podman run --rm \
    "${GPU_DEVS[@]}" "${GPU_LIBS[@]}" \
    -v "$REPO":/work:z -w /work \
    "$IMAGE" bash -lc "
        ldconfig 2>/dev/null || true
        snakemake --configfile config.yaml \
            --config md_ns=1 \
            --cores $CORES --resources gpu=1 \
            --keep-going --rerun-incomplete --printshellcmds \
            all
    "
