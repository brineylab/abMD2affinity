# abMD2affinity

Run molecular dynamics on protein structures and (for the AB-Bind
antibody-antigen dataset) relate mutant trajectories to binding affinity.

| Directory | Contents |
|-----------|----------|
| `run_MD/` | Generic Snakemake MD pipeline: structure prep, solvation, equilibration, production MD — driven by a config file + a structures list |
| `mutant_binding_affinity/` | AB-Bind-specific data and generators: raw dataset, input structures, mutant generation, and the structures list it feeds to `run_MD/` |
| `haddock_out/` | Docked-complex selection from the epiLora HADDOCK3 runs (winners + matched incorrect docks, RMSD-de-redundified) and its `run_MD/` structures list |
| `container/` | Dockerfile for the pipeline image (GROMACS-CUDA, OpenMM, PyMOL, Snakemake) |

See each directory's README.md for usage.
