# MD2affinity pipeline (PDB prep -> MD preprocessing -> production MD)

Self-contained Snakemake pipeline covering the first two-thirds of the
MD2affinity workflow: structure preparation, solvation/equilibration, and
production MD for antibody-antigen complexes. Trajectory analysis and Spearman
correlation (the final third of the original pipeline) are **not** included
here — see the parent MD2affinity repo for that stage.

```
PDB structure
  -> extract chains (Python)
  -> apply mutation (PyMOL)                    } PDB prep
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
  -> production MD (100 ns, GPU)                } production MD
  -> trajectory movie (mp4)                     } analysis
```

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

## Input

1. **`data/structures/`** — one PDB per system, named `{pdb_id}.pdb`. The
   `pdb_id` is whatever you key the `structures` map on; its first 4 chars are
   the dataset code. To run the Fab and Fv of the same complex as separate
   systems, give each its own file + key, e.g. `1dqjfv.pdb` / `1dqjfab.pdb`.
2. **A mutants TSV** with two columns, `pdb_id` and `mutant`:

   | pdb_id | mutant |
   |--------|--------|
   | 1dqj   | H:D32A |
   | 1dqj   | C:K97A |
   | 1jrh   | I:E45Q |

   `mutant` uses AB-Bind notation: `H`/`L` are the antibody heavy/light chains
   and get remapped to the real chain letters via each PDB's `chain_map`
   (below); any other letter is a real PDB chain letter (antigen) and passes
   through unchanged. Multiple point mutations are comma-separated
   (`H:D32A,L:S30A`). An empty `mutant` or the literal `WT` requests a
   wild-type-only run; one WT system is also auto-generated per PDB unless
   `auto_wt: false`.

   Generate the TSV from the raw AB-Bind dataset — one block of rows per entry
   in the `structures` map, drawing mutants from each key's 4-letter dataset
   code:

   ```bash
   python scripts/build_mutants_tsv.py \
       data/PRO-25-393-s002.tsv config.yaml -o data/mutants.tsv
   ```

There is **no** separate SAbDab metadata file or hand-maintained manifest — the
system list is derived from the mutants TSV crossed with the `structures` map
in `config.yaml`.

## Configuration

Edit `config.yaml`:

| Key | Meaning |
|-----|---------|
| `mutants_tsv` | the (pdb_id, mutant) TSV |
| `structures` | per-PDB map: `file` (input PDB) + optional `chain_map` (H/L -> real chain letters) |
| `output_dir` | root of the output tree (default `results/`) |
| `mdp_md` | production MD `.mdp` (override with `mdp/md_1ns.mdp` for a short test) |
| `auto_wt` | auto-generate a WT system per PDB (default `true`) |
| `make_movie` | render an mp4 per trajectory (default `true`); `movie_skip`, `movie_fps` tune it |
| `n_gpus`, `cpu_threads` | local resource pools |
| `s5cmd_dest` | object-storage destination for the preprocessing checkpoint sync (blank = skip) |
| `gmx_bin`, `pdb2pqr_bin`, `gmx_python_bin`, `pymol_python_bin`, `openmm_python_bin`, `s5cmd_bin` | executable paths (default to the container env layout) |

Chain mapping is only needed where a structure's chains aren't already labelled
`H`/`L`. For example 1dqj's chains are `A`/`B`/`C`, with the heavy chain at `B`
and light at `A`. Its Fv and Fab are two keys sharing the 1dqj dataset:

```yaml
structures:
  1dqjfv:
    file: "data/structures/1dqjfv.pdb"
    chain_map: {H: B, L: A}
  1dqjfab:
    file: "data/structures/1dqjfab.pdb"
    chain_map: {H: B, L: A}
```

## Output layout

```
results/
├── preprocessing/{pdb}/{pdb}_{mutant}/   # extraction through NPT equilibration
├── MD/{pdb}/{pdb}_{mutant}/              # production MD only
├── movies/{pdb}_{mutant}.mp4             # trajectory movies, collected (rule movie)
├── preprocessing.synced                  # sentinel: preprocessing/ has been synced
└── manifest.json                         # parsed system manifest (rule write_manifest)
```

System naming: `{pdb}_{mutation_tag}`, e.g. `1dqj_H-D32A`, `1dqj_WT_ABC`.

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
snakemake --configfile config.yaml --cores 32 --resources gpu0=1

# Short smoke test: one mutant per PDB, 1 ns (wraps the container + GPU wiring)
bash scripts/run_test_1ns.sh
```

GPU-bound rules (`openmm_minimize`, `production_md`) are scheduled round-robin
across the `gpu0..gpu3` resource pools; keep the `--resources gpuN=1` flags in
sync with `n_gpus` in `config.yaml`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_mutants_tsv.py` | Derive the (pdb_id, mutant) TSV from the raw AB-Bind dataset, filtered to available structures |
| `scripts/build_manifest.py` | Parse the mutants TSV x the `structures` map into a system manifest |
| `scripts/extract_chains.py` | Extract specified chains from a PDB file |
| `scripts/mutate_structure.py` | Apply point mutations with PyMOL's mutagenesis wizard |
| `scripts/minimize_openmm.py` | Vacuum energy minimization via OpenMM + AMBER14 |
| `scripts/render_movie.py` | Ray-trace trajectory frames with PyMOL for the movie (rule `movie`) |
| `scripts/run_test_1ns.sh` | Run the 1 ns smoke test in the container with GPU wired in |
