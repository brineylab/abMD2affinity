#!/usr/bin/env python3
"""
prepare_structures.py — Select docked antibody-antigen complexes for 5 ns MD
from the epiLora HADDOCK3 docking runs, and emit everything run_MD needs.

Selection:
  * winners — every cluster model with CAPRI quality Acceptable or better,
    from both the epiLoRA-constrained runs (runs/) and the vanilla runs
    (runs_vanilla/);
  * an equal number of incorrectly-docked models, drawn uniformly at random
    (seeded) from all Incorrect models;
  * top-scoring — per pdb that has any winner, the `top_per_pdb`
    best-HADDOCK-score (most negative first) models with both conditions
    pooled and any CAPRI quality: what you would run if you trusted the
    score.

All sets are de-redundified: within each (pdb, ab_run, condition) group,
models are superposed on the antigen (chain A) CA atoms and an antibody
(chain B) CA RMSD <= rmsd_threshold counts as the same binding mode — only
one representative per binding mode is kept (best CAPRI quality first, then
highest DockQ; the sampled incorrect set keeps its random draw order). The
random incorrect set is topped up until it has exactly as many unique
binding modes as the winner set. Top-scoring models must be new (not already
selected) and non-redundant w.r.t. the whole test set.

Outputs (relative to this script's directory):
  data/{pdb}_ab{run}_{epi|van}_c{clt}m{mdl}_{win|bad|top}.pdb   selected models
  intermediate_data/structures.yaml                          run_MD input (md_ns: 5)
  intermediate_data/winners.csv                              selection audit
  intermediate_data/incorrect_sampled.csv                     selection audit
  intermediate_data/top_scored.csv                            selection audit

Usage:
    python prepare_structures.py prep.yaml
"""

import argparse
import csv
import gzip
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent

QUALITY_RANK = {"High": 0, "Medium": 1, "Acceptable": 2, "Incorrect": 3}
RUNS_DIR = {"epilora": "runs", "vanilla": "runs_vanilla"}
COND_TAG = {"epilora": "epi", "vanilla": "van"}


def model_path(docking_root: Path, row: dict) -> Path:
    return (docking_root / RUNS_DIR[row["constraint"]] / row["pdb_id"]
            / f"ab_{row['ab_run']}_vs_ag" / "haddock_out" / "10_seletopclusts"
            / f"{row['model']}.pdb.gz")


def system_name(row: dict, label: str) -> str:
    clt, mdl = row["model"].replace("cluster_", "").replace("model_", "").split("_")
    return (f"{row['pdb_id']}_ab{row['ab_run']}_{COND_TAG[row['constraint']]}"
            f"_c{clt}m{mdl}_{label}")


def parse_ca(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(receptor CA coords chain A, ligand CA coords chain B) of a model."""
    rec, lig = [], []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16] == " CA ":
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                ch = line[21]
                if ch == "A":
                    rec.append(xyz)
                elif ch == "B":
                    lig.append(xyz)
    if not rec or not lig:
        raise ValueError(f"{path}: no chain A/B CA atoms found")
    return np.array(rec), np.array(lig)


def _fit(mobile: np.ndarray, target: np.ndarray):
    """Rigid transform (R, t) mapping `mobile` onto `target` (Kabsch)."""
    mc, tc = mobile.mean(0), target.mean(0)
    H = (mobile - mc).T @ (target - tc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, tc - mc @ R.T


def ligand_rmsd(a: tuple[np.ndarray, np.ndarray],
                b: tuple[np.ndarray, np.ndarray]) -> float:
    """RMSD of the antibody (chain B) CA atoms of `a` against `b`, after
    superposing `a` onto `b` on the antigen (chain A) CA atoms."""
    for name, x, y in (("receptor", a[0], b[0]), ("ligand", a[1], b[1])):
        if len(x) != len(y):
            raise ValueError(
                f"{name} CA count mismatch ({len(x)} vs {len(y)}) — models are "
                f"not from the same docking run")
    R, t = _fit(a[0], b[0])
    lig = a[1] @ R.T + t
    return float(np.sqrt(np.mean(np.sum((lig - b[1]) ** 2, axis=1))))


def dedupe(rows: list[dict], threshold: float, coords: dict) -> list[dict]:
    """Greedy: keep the first row of each binding mode (<= threshold RMSD to an
    already-kept row of the same group merges into it). Rows must be pre-
    sorted in keep-priority order. Adds 'kept' and 'merged_into' keys."""
    kept: list[dict] = []
    for row in rows:
        row["kept"] = True
        row["merged_into"] = ""
        for k in kept:
            if k["_group"] != row["_group"]:
                continue
            if ligand_rmsd(coords[row["_path"]], coords[k["_path"]]) <= threshold:
                row["kept"] = False
                row["merged_into"] = k["_name"]
                break
        if row["kept"]:
            kept.append(row)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="prep.yaml (relative paths resolve against it)")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent
    docking_root = (base / cfg["docking_root"]).resolve()
    threshold = float(cfg["rmsd_threshold"])
    rng = random.Random(cfg["seed"])

    with open(docking_root / cfg["capri_details"], newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["_group"] = (row["pdb_id"], row["ab_run"], row["constraint"])
        row["_path"] = model_path(docking_root, row)
    winners = [r for r in rows if r["capri_quality"] != "Incorrect"]
    pool = [r for r in rows if r["capri_quality"] == "Incorrect"]
    print(f"CAPRI table: {len(rows)} models — {len(winners)} winners "
          f"(Acceptable+), {len(pool)} incorrect")

    # --- winners: de-redundify, best model of each binding mode first -----
    for row in winners:
        row["_name"] = system_name(row, "win")
    winners.sort(key=lambda r: (QUALITY_RANK.get(r["capri_quality"], 9),
                               -float(r["dockq"])))
    coords: dict[Path, tuple] = {}
    for row in winners:
        coords[row["_path"]] = parse_ca(row["_path"])
    kept_winners = dedupe(winners, threshold, coords)
    n = len(kept_winners)
    print(f"winners: {len(winners)} -> {n} after {threshold} A de-redundancy "
          f"({len(winners) - n} merged)")

    # --- incorrect: uniform random draw, topped up to n unique modes -------
    rng.shuffle(pool)
    incorrect: list[dict] = []
    for row in pool:
        if len(incorrect) >= n:
            break
        row["_name"] = system_name(row, "bad")
        try:
            coords.setdefault(row["_path"], parse_ca(row["_path"]))
        except (ValueError, OSError) as e:
            print(f"WARNING: skipping {row['_path'].name}: {e}", file=sys.stderr)
            continue
        incorrect.append(row)
    kept_incorrect = dedupe(incorrect, threshold, coords)
    while len(kept_incorrect) < n and pool:
        row = pool.pop()
        if row.get("kept"):
            continue
        row["_name"] = system_name(row, "bad")
        try:
            coords.setdefault(row["_path"], parse_ca(row["_path"]))
        except (ValueError, OSError) as e:
            print(f"WARNING: skipping {row['_path'].name}: {e}", file=sys.stderr)
            continue
        incorrect.append(row)
        kept_incorrect = dedupe(incorrect, threshold, coords)
    if len(kept_incorrect) < n:
        sys.exit(f"incorrect pool exhausted at {len(kept_incorrect)} unique "
                 f"modes — cannot match {n} winners")
    print(f"incorrect: sampled {len(incorrect)} -> {len(kept_incorrect)} unique "
          f"binding modes (target {n})")

    # --- top-scoring: per winner pdb, the best-HADDOCK-score models (both
    #     conditions pooled, any CAPRI quality), non-redundant w.r.t. the
    #     whole test set ------------------------------------------------------
    n_top = int(cfg.get("top_per_pdb", 0))
    top_all: list[dict] = []       # every candidate tried, for the audit csv
    kept_top: list[dict] = []
    if n_top:
        selected_ids = {id(r) for r in kept_winners} | {id(r) for r in kept_incorrect}
        selected_by_group: dict[tuple, list[dict]] = defaultdict(list)
        for r in kept_winners + kept_incorrect:
            selected_by_group[r["_group"]].append(r)
        for pdb in sorted({r["pdb_id"] for r in kept_winners}):
            cands = [r for r in rows
                     if r["pdb_id"] == pdb and id(r) not in selected_ids]
            cands.sort(key=lambda r: float(r["haddock_score"])
                       if r["haddock_score"].strip() else float("inf"))
            kept_pdb: list[dict] = []
            for row in cands:
                if len(kept_pdb) >= n_top:
                    break
                row["_name"] = system_name(row, "top")
                row["kept"], row["merged_into"] = True, ""
                try:
                    coords.setdefault(row["_path"], parse_ca(row["_path"]))
                except (ValueError, OSError) as e:
                    print(f"WARNING: skipping {row['_path'].name}: {e}", file=sys.stderr)
                    row["kept"], row["merged_into"] = False, "unreadable"
                    top_all.append(row)
                    continue
                blockers = list(selected_by_group.get(row["_group"], []))
                blockers += [k for k in kept_pdb if k["_group"] == row["_group"]]
                hit = next((k for k in blockers
                            if ligand_rmsd(coords[row["_path"]],
                                           coords[k["_path"]]) <= threshold), None)
                if hit is not None:
                    row["kept"], row["merged_into"] = False, hit["_name"]
                else:
                    kept_pdb.append(row)
                top_all.append(row)
            kept_top += kept_pdb
            n_tried = sum(1 for r in top_all if r["pdb_id"] == pdb)
            print(f"top-scoring {pdb}: kept {len(kept_pdb)} "
                  f"({n_tried - len(kept_pdb)} blocked as redundant/unreadable)")

    # --- write out ----------------------------------------------------------
    data_dir = HERE / "data"
    inter_dir = HERE / "intermediate_data"
    data_dir.mkdir(exist_ok=True)
    inter_dir.mkdir(exist_ok=True)
    for stale in data_dir.glob("*.pdb"):
        stale.unlink()

    def write_pdb(row: dict) -> None:
        out = data_dir / f"{row['_name']}.pdb"
        with gzip.open(row["_path"], "rt") as fh, open(out, "w") as fo:
            fo.write(fh.read())

    def write_csv(path: Path, rows_all: list[dict]) -> None:
        cols = ["name", "pdb_id", "constraint", "ab_run", "model",
                "capri_quality", "dockq", "haddock_score", "kept", "merged_into"]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore",
                               lineterminator="\n")
            w.writeheader()
            for row in rows_all:
                row["name"] = row["_name"]
                w.writerow(row)

    for row in kept_winners + kept_incorrect + kept_top:
        write_pdb(row)
    write_csv(inter_dir / "winners.csv", winners)
    write_csv(inter_dir / "incorrect_sampled.csv", incorrect)
    write_csv(inter_dir / "top_scored.csv", top_all)

    systems = [{"name": row["_name"], "path": f"../data/{row['_name']}.pdb",
                "md_ns": 5} for row in kept_winners + kept_incorrect + kept_top]
    header = (
        "# Docked-complex structures for 5 ns MD — generated by\n"
        f"# haddock_out/prepare_structures.py from\n"
        f"# {docking_root}/{cfg['capri_details']} (rmsd_threshold={threshold} A,\n"
        f"# seed={cfg['seed']}). MD outputs land in haddock_out/results/.\n"
        "# Run the pipeline with:\n"
        "#   cd ../run_MD && snakemake --configfile config.yaml \\\n"
        "#       --config structures_file=../haddock_out/intermediate_data/structures.yaml\n"
    )
    with open(inter_dir / "structures.yaml", "w") as fh:
        fh.write(header)
        yaml.safe_dump({"base_dir": "..", "structures": systems}, fh,
                       sort_keys=False, default_flow_style=False)

    by_grp = defaultdict(int)
    for row in kept_winners:
        by_grp[(row["pdb_id"], row["constraint"])] += 1
    print(f"\nwrote {len(kept_winners)} winners + {len(kept_incorrect)} incorrect "
          f"+ {len(kept_top)} top-scoring -> {data_dir}")
    for (pdb, cond), cnt in sorted(by_grp.items()):
        print(f"  {pdb} {cond}: {cnt} winners kept")


if __name__ == "__main__":
    main()
