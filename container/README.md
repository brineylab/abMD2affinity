# container

Dockerfile for the pipeline image: three conda envs — `md` (GROMACS-CUDA,
pdb2pqr, snakemake, python), `openmm` (OpenMM + pdbfixer), `pymol` (PyMOL).

```bash
podman build -t abmd2affinity:cuda129 -f Dockerfile .
```

The image does not bundle the NVIDIA driver; inject it at runtime via the
nvidia-container-toolkit (`--device nvidia.com/gpu=all`) or bind the host
driver libs + device nodes by hand (see `../run_MD/scripts/run_test_1ns.sh`
for a working example).
