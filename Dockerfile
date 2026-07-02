# =============================================================================
# Container for the abMD2affinity pipeline (Snakemake).
# CUDA 12.9 GPU build.
#
# Everything the pipeline invokes, in one image (three isolated conda envs):
#   md      GROMACS (CUDA 12.9)  — pdb2gmx / editconf / solvate / genion /
#                                  grompp / mdrun(GPU)
#           pdb2pqr              — PROPKA protonation at pH 7
#           snakemake            — workflow manager (+ slurm executor plugin)
#           python               — extract_chains.py (stdlib only)
#   openmm  OpenMM + pdbfixer    — minimize_openmm.py (vacuum GPU minimize)
#   pymol   PyMOL (open-source)  — mutate_structure.py (mutagenesis wizard)
#
# The three envs are kept separate so the OpenMM/PyMOL Qt/CUDA stacks don't
# fight the GROMACS solve. The pipeline already supports per-tool interpreters
# (gmx_python_bin / openmm_python_bin / pymol_python_bin / pdb2pqr_bin in
# config.yaml), so each rule calls into the right env.
#
# The NVIDIA *driver* is NOT installed — the container runtime's nvidia hook
# (enroot / nvidia-container-toolkit) injects the host driver at runtime. The
# conda packages carry the CUDA 12.9 runtime libraries.
# =============================================================================
FROM docker.io/condaforge/miniforge3:24.11.3-2

# 12.9 (not 12.8): conda-forge's CUDA GROMACS builds are pinned to cuda 12.6 or
# >=12.9 — there is no nompi_cuda GROMACS for 12.8. Driver 580 supports 12.9.
ARG CUDA_VERSION=12.9

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# OpenGL/runtime bits PyMOL needs for headless (offscreen) rendering on nodes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libglu1-mesa \
        libxrender1 libxext6 libsm6 \
        libgomp1 ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Env "md": GROMACS (CUDA) + pdb2pqr + the workflow stack.
# "gromacs=*=nompi_cuda_*" selects the thread-MPI CUDA build variant.
# CONDA_OVERRIDE_CUDA lets the solver pick the CUDA build on a driver-less
# build host (build machines usually have no GPU).
# -----------------------------------------------------------------------------
RUN CONDA_OVERRIDE_CUDA=${CUDA_VERSION} mamba create -y -n md -c conda-forge \
        python=3.11 \
        "cuda-version=${CUDA_VERSION}" \
        "gromacs=*=nompi_cuda_*" \
        pdb2pqr \
        "numpy>=1.24.0" \
    && conda clean -afy

# pdb2pqr's console script is named `pdb2pqr30`; expose it as `pdb2pqr` too so
# the config.yaml default (pdb2pqr_bin: "pdb2pqr") resolves either way.
RUN if [ ! -e /opt/conda/envs/md/bin/pdb2pqr ]; then \
        ln -s pdb2pqr30 /opt/conda/envs/md/bin/pdb2pqr; \
    fi

# Snakemake + the SLURM executor plugin live on PyPI (not conda-forge); install
# them into the md env with pip.
RUN /opt/conda/envs/md/bin/pip install --no-cache-dir \
        "snakemake>=8.0.0" \
        "snakemake-executor-plugin-slurm>=0.5.0"

# -----------------------------------------------------------------------------
# Env "openmm": OpenMM + pdbfixer for the vacuum energy minimization
# (minimize_openmm.py). CUDA build so the minimize can run on a GPU.
# -----------------------------------------------------------------------------
RUN CONDA_OVERRIDE_CUDA=${CUDA_VERSION} mamba create -y -n openmm -c conda-forge \
        python=3.11 \
        "cuda-version=${CUDA_VERSION}" \
        openmm \
        pdbfixer \
    && conda clean -afy

# -----------------------------------------------------------------------------
# Env "pymol": open-source PyMOL, isolated from the science Python envs to
# avoid Qt/python solver conflicts (mutate_structure.py imports pymol).
# -----------------------------------------------------------------------------
RUN mamba create -y -n pymol -c conda-forge \
        pymol-open-source \
    && conda clean -afy \
    && ln -s /opt/conda/envs/pymol/bin/pymol /usr/local/bin/pymol

# ffmpeg (into the md env, on PATH) — encodes PyMOL frames into the trajectory
# movie in rule `movie`.
RUN mamba install -y -n md -c conda-forge ffmpeg \
    && conda clean -afy

# Put the md env on PATH so `snakemake`, `gmx`, `pdb2pqr`, `python` resolve.
ENV PATH=/opt/conda/envs/md/bin:/opt/conda/condabin:${PATH} \
    CONDA_DEFAULT_ENV=md \
    # Headless / offscreen defaults so PyMOL works on compute nodes
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg

WORKDIR /work

CMD ["/bin/bash"]
