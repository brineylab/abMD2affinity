# MD pipeline (PDB prep -> MD preprocessing -> production MD)

Generic Snakemake pipeline that takes a list of input structures and runs MD
on each of them: structure preparation, solvation/equilibration, and
production MD.

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

Run everything from this directory (`run_MD/`).

## Input: a config file + a structures list

1. **`config.yaml`** — parameters: the object-storage destination
   (`output_dir`), the default production-MD length (`md_ns` in nanoseconds,
   or `md_steps` in integration steps), movie options, thread counts,
   executable paths.
2. **A structures file** (`structures.yaml`, or whatever `structures_file`
   points at) — the structures to run MD on:

```yaml
base_dir: /data/md              # anchor for all outputs
output_dir: runs                # optional — replaces "results" below
structures:
  - name: 1bj1fv                # -> {base_dir}/results/1bj1fv/ (or runs/1bj1fv/)
    path: data/1bj1fv.pdb       # input PDB (relative -> against this file's dir)
  - name: 1dqj_500ns
    path: data/1dqjfv.pdb
    output_dir: long/1dqj       # optional — per-structure output dir (-> base_dir/long/1dqj/)
    md_ns: 500                  # optional — per-structure length override
    chains: [A, B]              # optional — default: every chain in the file
```

Each system's output dir is, in order of precedence: the entry's `output_dir`
(relative -> under `base_dir`), else
`{base_dir}/{output_dir or "results"}/{name}/`. System names must be
filesystem-safe (`[A-Za-z0-9][A-Za-z0-9._-]*`).

Validate a structures list without running anything:

```bash
python scripts/build_systems.py structures.yaml --config config.yaml
```

## Configuration

Edit `config.yaml`:

| Key | Meaning |
|-----|---------|
| `structures_file` | the structures list (or put a `structures:` document inline in config.yaml) |
| `output_dir` | object-storage destination every system's `preprocessing/` tree is synced to before production MD starts; blank = skip |
| `md_ns` / `md_steps` | default production-MD length — set exactly one (ns, or raw steps); individual structures may override it in the structures file |
| `mdp_md` | base production `.mdp` (all other parameters copied verbatim) |
| `make_movie` | render an mp4 per trajectory (default `true`); `movie_skip`, `movie_fps` tune it |
| `cpu_threads` | CPU threads per multi-threaded rule |
| `pdb2pqr_bin`, `gmx_python_bin`, `pymol_python_bin`, `openmm_python_bin`, `s5cmd_bin` | executable paths (default to the container env layout) |

## Output layout

Each system gets its own directory:

```
{base_dir}/results/{name}/
├── preprocessing/    # extraction through NPT equilibration (+ logs/)
├── MD/               # md.mdp, md.tpr, md.gro, md.xtc (+ logs/)
└── movie.mp4         # trajectory movie (rule movie)
```

`{base_dir}/results/manifest.json` records the parsed system list —
{system: path, chains, length, output_dir} — and is what the cluster
launcher consumes.

## Object-storage checkpoint

Once every requested system has reached `npt.gro`, rule `sync_preprocessing`
runs `s5cmd sync {system}/preprocessing/ {output_dir}/{system}/preprocessing/`
for each system, and every `production_md` job depends on that completing
first — so **no production MD starts until the entire preprocessing batch is
archived**. Set `output_dir` for a real batch; leave it blank to skip the
sync and start production immediately.

## Container

Everything the pipeline invokes lives in one image, built from
`../container/Dockerfile` (three conda envs: `md` — GROMACS-CUDA, pdb2pqr,
snakemake; `openmm` — OpenMM + pdbfixer; `pymol` — PyMOL). `config.yaml`
points each rule at the right env's interpreter. The image does not bundle
the NVIDIA driver — on a host with the nvidia-container-toolkit pass
`--device nvidia.com/gpu=all`; otherwise bind the host driver libs and device
nodes by hand (see `scripts/run_test_1ns.sh`).

## Running

All commands run inside the container (mount this directory at `/work`):

```bash
# Dry-run
snakemake --configfile config.yaml -n

# Full run
snakemake --configfile config.yaml --cores 32 --resources gpu=<n_gpus>

# Short smoke test: the structures.yaml systems, 1 ns each
# (wraps the container + GPU wiring)
bash scripts/run_test_1ns.sh

# Short run by hand
snakemake --configfile config.yaml --config md_ns=1 --cores 8 --resources gpu=1

# Single system
snakemake --configfile config.yaml --cores 8 --resources gpu=1 \
    /abs/path/to/{base_dir}/results/{name}/MD/md.gro
```

GPU-bound rules (`openmm_minimize`, `production_md`) each request 1 GPU
(`resources: gpu=1`); pass `--resources gpu=N` so Snakemake runs up to `N` of
them concurrently.

## Cluster (slurm/)

- `submit_pipeline.sbatch` — run the whole pipeline through Snakemake's Slurm
  executor (one job per rule instance).
- `run_md.sh` / `launch_experiments.py` — standalone production-MD fan-out
  that bypasses Snakemake on compute nodes: the launcher reads
  `manifest.json`, submits one `run_md.sh <system> <output_dir>` job per
  prepped system, capped at `--cap` jobs in the queue, with per-system
  `launch.submitted` markers for resume.
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
