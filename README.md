# MD pipeline (PDB prep -> MD preprocessing -> production MD)

Generic Snakemake pipeline that takes a list of input structures and runs MD
on each of them: structure preparation, solvation/equilibration, and
production MD. It works on any protein (or protein–ligand/protein–glycan)
complex — the AB-Bind antibody dataset that drove the original version is
now an optional input handled by `abbind_scripts/` (see below).

```
PDB structure
  -> extract chains (Python)                  } PDB prep
  -> vacuum minimize (OpenMM, GPU)
  -> PDB2PQR pH 7 protonation
  -> pdb2gmx AMBER99SB-ILDN
  -> editconf (triclinic, 1.2 nm)
  -> solvate (TIP3P)                            } MD preprocessing
  -> genion (0.15 M NaCl, neutral)
  -> GROMACS EM (steepest descent)
  -> NVT equilibration (100 ps)
  -> NPT equilibration (100 ps)
  -> [sync preprocessing/ to object storage]
  -> production MD (md_ns / md_steps, GPU)     } production MD
  -> trajectory movie (mp4)                     } analysis
```

## Input: a config file + a list of structures

You interact with the pipeline through two files:

1. **`config.yaml`** — parameters: the default production-MD length
   (`md_ns` in nanoseconds, or `md_steps` in integration steps), movie
   options, thread counts, executable paths, object-storage sync destination.
2. **A structures list** (`structures.yaml`, or whatever `structures_file`
   points at) — the structures to run MD on. Each entry is one input
   structure; `name` is the system name used under `results/`:

```yaml
structures:
  - name: 1bj1fv                  # -> default length from config.yaml
    path: data/structures/1bj1fv.pdb

  - name: 1dqj_500ns              # -> same structure, 500 ns instead
    path: data/structures/1dqjfv.pdb
    md_ns: 500                    # per-structure override of md_ns/md_steps

  - name: ab42_binding            # -> only run chains A and B of the file
    path: structures/ab42.pdb
    chains: [A, B]                # optional; default: every chain in the file
```

Entries accept `name` and `path` (both required), `chains` (optional), and
exactly one of `md_ns` / `md_steps` (optional, overrides the global default).
System names become the path segments under `results/`, so they must be
filesystem-safe (`[A-Za-z0-9][A-Za-z0-9._-]*`).

Validate a structures list without running anything:

```bash
python scripts/build_systems.py structures.yaml --config config.yaml
```

## Configuration

Edit `config.yaml`:

| Key | Meaning |
|-----|---------|
| `structures_file` | the structures list (or put a `structures:` map inline in config.yaml) |
| `md_ns` / `md_steps` | default production-MD length — set exactly one (ns, or raw steps); individual structures may override it in the structures list |
| `mdp_md` | base production `.mdp` (all other parameters copied verbatim) |
| `output_dir` | root of the output tree (default `results/`) |
| `make_movie` | render an mp4 per trajectory (default `true`); `movie_skip`, `movie_fps` tune it |
| `cpu_threads` | CPU threads per multi-threaded rule |
| `s5cmd_dest` | object-storage destination for the preprocessing checkpoint sync (blank = skip) |
| `pdb2pqr_bin`, `gmx_python_bin`, `pymol_python_bin`, `openmm_python_bin`, `s5cmd_bin` | executable paths (default to the container env layout) |

## AB-Bind workflow (abbind_scripts/)

To reproduce the original antibody-antigen batch from the AB-Bind dataset,
generate the mutant structures and a structures list for them, then hand that
list to the (unchanged) pipeline:

```bash
# raw dataset -> mutants TSV (one row per pdb_id/mutant)
python abbind_scripts/build_mutants_tsv.py abbind_scripts/abbind.yaml

# mutants TSV -> mutant PDBs (PyMOL applies each mutation, chain_map
# H/L -> real chain letters) + structures_abbind.yaml listing every
# WT + mutant system
python abbind_scripts/build_structures.py abbind_scripts/abbind.yaml

# run the pipeline on it
snakemake --configfile config.yaml \
    --config structures_file=structures_abbind.yaml --cores 32 --resources gpu=4
```

`abbind_scripts/abbind.yaml` holds the dataset paths, the per-PDB `structures`
map (input file + optional `chain_map`), the mutant output directory, and the
PyMOL interpreter. Mutant PDBs are written to `data/abbind_mutants/`
(gitignored) and kept across runs, so the generator is resumable.

## Container

Everything the pipeline invokes lives in one image (`./Dockerfile`), built with
three isolated conda envs — `md` (GROMACS-CUDA, pdb2pqr, snakemake, python),
`openmm` (OpenMM + pdbfixer), `pymol` (PyMOL). `config.yaml` points each rule at
the right env's interpreter.

```bash
podman build -t abmd2affinity:cuda129 -f Dockerfile .
```

The image does **not** bundle the NVIDIA driver. On a host with the
nvidia-container-toolkit, pass `--device nvidia.com/gpu=all`. Where that isn't
available (e.g. rootless podman without CDI), bind the host driver libs and
device nodes by hand — see `scripts/run_test_1ns.sh` for a working example.

## Output layout

```
results/
├── preprocessing/{system}/    # extraction through NPT equilibration
├── MD/{system}/               # production MD only
├── movies/{system}.mp4        # trajectory movies, collected (rule movie)
├── params/{system}/md.mdp     # production .mdp with that system's length
├── preprocessing.synced       # sentinel: preprocessing/ has been synced
└── manifest.json              # parsed system manifest (rule write_manifest)
```

## Object-storage checkpoint

Once every requested system has reached `npt.gro`, rule `sync_preprocessing`
runs `s5cmd sync results/preprocessing/ <s5cmd_dest>`, and every `production_md`
job depends on that sync completing first — so **no production MD starts until
the entire preprocessing batch is archived**. Set `s5cmd_dest` before a real
batch; leave it blank to skip the sync and start production immediately.

## Running

All commands run inside the container. Mount the repo at `/work` and add GPU
access as described above.

```bash
# Dry-run
snakemake --configfile config.yaml -n

# Full run
snakemake --configfile config.yaml --cores 32 --resources gpu=<n_gpus>

# Short smoke test: the structures.yaml systems, 1 ns each
# (wraps the container + GPU wiring)
bash scripts/run_test_1ns.sh

# Short run by hand: override the length without editing config.yaml
snakemake --configfile config.yaml --config md_ns=1 --cores 8 --resources gpu=1
```

GPU-bound rules (`openmm_minimize`, `production_md`) each request 1 GPU
(`resources: gpu=1`); pass `--resources gpu=N` so Snakemake runs up to `N` of
them concurrently. Each process picks up its allocated device via
`CUDA_VISIBLE_DEVICES`.

## Cluster (slurm/)

- `submit_pipeline.sbatch` — run the whole pipeline through Snakemake's Slurm
  executor (one job per rule instance).
- `run_md.sh` / `launch_experiments.py` — standalone production-MD fan-out
  that bypasses Snakemake on compute nodes: the launcher submits one
  `run_md.sh` job per prepped system (`results/preprocessing/<system>/npt.gro`),
  capped at `--cap` jobs in the queue, with marker files for resume.
- `launch_experiments.py --extend` / `run_md_extend.sh` — extend existing
  trajectories in object storage to a target length (default 500 ns) by
  resuming mdrun from the furthest-along checkpoint.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_systems.py` | Parse + validate a structures list into the system manifest |
| `scripts/make_md_mdp.py` | Write the production .mdp with a system's run length (md_ns / md_steps) |
| `scripts/extract_chains.py` | Extract specified chains from a PDB file |
| `scripts/minimize_openmm.py` | Vacuum energy minimization via OpenMM + AMBER14 |
| `scripts/render_movie.py` | Ray-trace trajectory frames with PyMOL for the movie (rule `movie`) |
| `scripts/run_test_1ns.sh` | Run the 1 ns smoke test in the container with GPU wired in |
| `abbind_scripts/build_mutants_tsv.py` | Raw AB-Bind dataset -> mutants TSV |
| `abbind_scripts/build_structures.py` | Mutants TSV -> mutant PDBs (PyMOL) + generic structures list |
| `abbind_scripts/mutate_structure.py` | Apply point mutations with PyMOL's mutagenesis wizard |
