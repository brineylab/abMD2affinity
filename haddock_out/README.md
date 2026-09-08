# haddock_out

Docked antibody-antigen complexes from the epiLora HADDOCK3 runs, selected
for 5 ns MD — three groups:

1. **winners** — every correctly-docked (CAPRI Acceptable+) cluster model
   from both the epiLoRA-constrained (`runs/`) and vanilla
   (`runs_vanilla/`) conditions;
2. **incorrect** — an equal number of incorrectly-docked models drawn
   uniformly at random (seeded);
3. **top-scoring** — per pdb that has any winner, the 50 best-HADDOCK-score
   models (both conditions pooled, any CAPRI quality) not already in the
   test set: what you would run if you trusted the score. In practice these
   are all incorrectly docked — the Acceptable+ poses are already captured
   by the winner set — making this a hard-negative group.

All groups are de-redundified: within each (pdb, ab_run, condition) group,
models are superposed on the antigen (chain A) CA atoms and an antibody
(chain B) CA RMSD <= 2.0 A counts as the same binding mode — only one
representative is kept (best CAPRI quality, then DockQ).

```bash
# (re)generate the selection — reads ../epiLoRA/docking
python prepare_structures.py prep.yaml

# run preprocessing with run_MD (production MD runs elsewhere, e.g. via the
# slurm launcher against haddock_out/results/manifest.json)
cd ../run_MD
snakemake --configfile config.yaml \
    --config structures_file=../haddock_out/intermediate_data/structures.yaml \
    --cores 32 --resources gpu=4 preprocess
```

## Extending a subset to 500 ns

Only a balanced 96-system subset (48 winners, 24 top-scored, 24 incorrect,
spread across PDB ids) is extended to 500 ns; `select_extend_set.py` writes it
out of `results/manifest.json`, each entry carrying `target_ns: 500`:

```bash
python select_extend_set.py            # -> results/manifest_extend500.json

# on the cluster (submit node):
cd ../run_MD
python slurm/launch_experiments.py \
    ../haddock_out/results/manifest_extend500.json \
    --extend --cap 96 --max-attempts 20
```

A 495 ns extension spans several 48 h jobs; each attempt resumes from the
previous job's checkpoint until the system writes `extend.done`.

## Ground-truth complexes at 500 ns

`prepare_ground_truth.py` extracts the crystal Fv + antigen (docked-model
convention: chain A = antigen, chain B = Fv) for every complex whose docking
succeeded — the PDBs with kept winners — from the epiLoRA SAbDab export,
writing `data_gt/{pdb}_gt.pdb`, `intermediate_data/structures_gt.yaml` (a
0.5 ns bootstrap run) and `results_gt/manifest_gt500.json` (`target_ns: 500`):

```bash
python prepare_ground_truth.py

# on the cluster: preprocess, bootstrap production, then extend as above
cd ../run_MD
snakemake --configfile config.yaml \
    --config structures_file=../haddock_out/intermediate_data/structures_gt.yaml \
    --cores 32 --resources gpu=4 preprocess
python slurm/launch_experiments.py ../haddock_out/results_gt/manifest.json
python slurm/launch_experiments.py \
    ../haddock_out/results_gt/manifest_gt500.json \
    --extend --cap 96 --max-attempts 20
```

Layout:

- `prep.yaml` — inputs and parameters (docking root, RMSD threshold, RNG
  seed, top_per_pdb).
- `data/` — the selected models, gunzipped, named
  `{pdb}_ab{run}_{epi|van}_c{clt}m{mdl}_{win|bad|top}.pdb`.
- `intermediate_data/` — the selection audit (`winners.csv`,
  `incorrect_sampled.csv`, `top_scored.csv`, with per-model kept /
  merged-into decisions) and `structures.yaml`, the run_MD input (one entry
  per selected model, `md_ns: 5`).
- `results/` — MD output written by `run_MD/` (gitignored).
