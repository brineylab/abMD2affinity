# mutant_binding_affinity

AB-Bind antibody-antigen dataset: input structures, raw + derived mutation
data, and the generators that turn them into a structures list for `run_MD/`.

```bash
# 1. raw dataset -> mutants TSV (one row per pdb_id/mutant)
python build_mutants_tsv.py abbind.yaml

# 2. mutants TSV -> mutant PDBs (PyMOL, in mutants/) + structures_abbind.yaml
#    (resumable: existing mutant PDBs are kept; --dry-run to preview)
python build_structures.py abbind.yaml [--dry-run]

# 3. run MD on every system (results land in results/{system}/)
cd ../run_MD
snakemake --configfile config.yaml \
    --config structures_file=../mutant_binding_affinity/structures_abbind.yaml
```

Layout:

- `abbind.yaml` — dataset paths, per-PDB structure map (+ optional `chain_map`
  remapping AB-Bind H/L antibody labels to real chain letters), PyMOL
  interpreter. Relative paths resolve against this file.
- `data/` — input data: the raw AB-Bind dataset (`PRO-25-393-s002.tsv`), the
  derived `mutants.tsv` / `completed_mutants.tsv`, and `structures/` (the
  input PDBs).
- `mutants/` — generated mutant PDBs (intermediate; gitignored, resumable).
- `structures_abbind.yaml` — generated structures list (one entry per WT +
  mutant system, mutation tags from real-chain notation, e.g.
  `1dqjfv_B-D32A`). Deletion mutants are skipped with a warning.
- `results/` — MD output written by `run_MD/` (gitignored).

Point `run_MD` at a different structures list for any other set of inputs —
this directory is only needed for AB-Bind work.
