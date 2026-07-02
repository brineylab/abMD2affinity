# Snakefile — MD2affinity: PDB prep + MD preprocessing + production MD
#
# Covers:
#   PDB structure
#     -> extract chains (Python)
#     -> apply mutation (PyMOL)                       } PDB prep
#     -> vacuum minimize (OpenMM, GPU)
#     -> PDB2PQR pH 7 protonation
#     -> pdb2gmx AMBER99SB-ILDN
#     -> editconf (triclinic, 1.2 nm)
#     -> solvate (TIP3P)                               } MD preprocessing
#     -> genion (0.15 M NaCl, neutral)
#     -> GROMACS EM (steepest descent)
#     -> NVT equilibration (100 ps)
#     -> NPT equilibration (100 ps)
#     -> production MD (100 ns, GPU)                   } production MD
#
# Does NOT include trajectory analysis / Spearman correlation — see the
# parent MD2affinity repo for that stage.
#
# Input:
#   - config["mutants_tsv"]: TSV with columns pdb_id, mutant.
#   - config["structures"]:  per-PDB map of input PDB file + optional chain_map
#                            remapping H/L to real chain letters (see
#                            scripts/build_manifest.py).
#
# Output layout (under config["output_dir"], default "results/"):
#   preprocessing/{pdb}/{pdb}_{mut}/   — chain extraction through NPT equilibration
#   MD/{pdb}/{pdb}_{mut}/              — production MD only
#
# A sync gate (rule sync_preprocessing) copies the entire preprocessing/ tree
# to object storage via `s5cmd sync` once every requested system has reached
# npt.gro. Every production_md job depends on that sync completing, so no
# production run starts until the full preprocessing batch is archived.
# Set config["s5cmd_dest"] = "" to skip the sync (production starts immediately).
#
# GPU: openmm_minimize and production_md each request 1 GPU (resources: gpu=1).
# Launch under Slurm so each job is allocated its own GPU; GROMACS/OpenMM pick
# up the allocated device via CUDA_VISIBLE_DEVICES.
#
# Run (local):
#   snakemake --configfile config.yaml --cores 32 --resources gpu=<n_gpus>
# Dry-run:
#   snakemake --configfile config.yaml -n
# Single system:
#   snakemake --configfile config.yaml --cores 8 --resources gpu=1 \
#       results/MD/1dqj/1dqj_H-D32A/md.gro
# Slurm:
#   sbatch slurm/submit_pipeline.sbatch

import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from build_manifest import build_systems

configfile: "config.yaml"

OUT    = Path(config["output_dir"])
PRE    = OUT / "preprocessing"
MD     = OUT / "MD"
MOVIES = OUT / "movies"

CPU_THREADS = int(config.get("cpu_threads", 8))

PDB2PQR        = config.get("pdb2pqr_bin", "pdb2pqr")
GMX_PYTHON     = config.get("gmx_python_bin", "python3")
PYMOL_PYTHON   = config.get("pymol_python_bin", "python")
OPENMM_PYTHON  = config.get("openmm_python_bin", "python")
S5CMD          = config.get("s5cmd_bin", "s5cmd")
S5CMD_DEST     = config.get("s5cmd_dest", "")
MDP_MD         = config.get("mdp_md", "mdp/md.mdp")

MAKE_MOVIE     = config.get("make_movie", True)
if isinstance(MAKE_MOVIE, str):
    MAKE_MOVIE = MAKE_MOVIE.strip().lower() not in ("false", "0", "no", "off", "")

_auto_wt = config.get("auto_wt", True)
if isinstance(_auto_wt, str):
    _auto_wt = _auto_wt.strip().lower() not in ("false", "0", "no", "off", "")

SYSTEMS = build_systems(config["mutants_tsv"], config["structures"], auto_wt=_auto_wt)
if not SYSTEMS:
    raise ValueError(
        "No systems parsed — check config['mutants_tsv'] and config['structures']."
    )
# Constrain {mut} to the exact set of valid mutation tags so Snakemake never
# tries to apply rules to stale or unrelated files in the output tree.
_MUTS_PATTERN = "(?:" + "|".join(
    sorted({v["mutation_tag"] for v in SYSTEMS.values()}, key=len, reverse=True)
) + ")"

# pdb_id is the structures-map key (e.g. 1dqjfv, 1dqjfab); no underscores, so
# it never collides with the mutation tag in the {pdb}_{mut} path segment.
wildcard_constraints:
    pdb = "[a-z0-9]+",
    mut = _MUTS_PATTERN,


def sys_key(wc) -> str:
    return f"{wc.pdb}_{wc.mut}"


def sys_data(wc) -> dict:
    return SYSTEMS[sys_key(wc)]


_PDBS = [v["pdb"] for v in SYSTEMS.values()]
_MUTS = [v["mutation_tag"] for v in SYSTEMS.values()]


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

_TARGETS = list(expand(str(MD / "{pdb}" / "{pdb}_{mut}" / "md.gro"),
                       zip, pdb=_PDBS, mut=_MUTS))
if MAKE_MOVIE:
    _TARGETS += list(expand(str(MOVIES / "{pdb}_{mut}.mp4"),
                            zip, pdb=_PDBS, mut=_MUTS))


rule all:
    input:
        _TARGETS,


# ---------------------------------------------------------------------------
# Step 0 — Write the parsed system manifest to disk (for inspection/audit)
# ---------------------------------------------------------------------------

rule write_manifest:
    output:
        json = str(OUT / "manifest.json"),
    run:
        import json
        with open(output.json, "w") as fh:
            json.dump(SYSTEMS, fh, indent=2)


# ---------------------------------------------------------------------------
# Step 1 — Extract the relevant chains from the input PDB (PDB prep)
# ---------------------------------------------------------------------------

rule extract_chains:
    input:
        pdb = lambda wc: sys_data(wc)["pdb_file"],
    output:
        pdb = str(PRE / "{pdb}" / "{pdb}_{mut}" / "00_extracted.pdb"),
    params:
        chains = lambda wc: " ".join(sys_data(wc)["chains_to_extract"]),
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "extract_chains.log"),
    shell:
        """
        {GMX_PYTHON} scripts/extract_chains.py \
            {input.pdb} {output.pdb} {params.chains} \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 2 — Apply mutations with PyMOL (WT systems: copy unchanged) (PDB prep)
# ---------------------------------------------------------------------------

rule mutate:
    input:
        pdb = str(PRE / "{pdb}" / "{pdb}_{mut}" / "00_extracted.pdb"),
    output:
        pdb = str(PRE / "{pdb}" / "{pdb}_{mut}" / "01_mutated.pdb"),
    params:
        mut_str = lambda wc: sys_data(wc).get("mutation_str_pdb") or "",
        is_wt   = lambda wc: "1" if sys_data(wc).get("is_wt", False) else "0",
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "mutate.log"),
    shell:
        """
        if [ "{params.is_wt}" = "1" ] || [ -z "{params.mut_str}" ]; then
            cp {input.pdb} {output.pdb}
            echo "WT — no mutation applied" > {log}
        else
            {PYMOL_PYTHON} scripts/mutate_structure.py \
                {input.pdb} {output.pdb} "{params.mut_str}" \
                > {log} 2>&1
        fi
        """


# ---------------------------------------------------------------------------
# Step 3 — OpenMM vacuum energy minimization, GPU (MD preprocessing)
# ---------------------------------------------------------------------------

rule openmm_minimize:
    input:
        pdb = str(PRE / "{pdb}" / "{pdb}_{mut}" / "01_mutated.pdb"),
    output:
        pdb = str(PRE / "{pdb}" / "{pdb}_{mut}" / "minimized.pdb"),
    resources:
        gpu = 1,
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "openmm_minimize.log"),
    shell:
        """
        {OPENMM_PYTHON} scripts/minimize_openmm.py \
            {input.pdb} {output.pdb} \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 4 — PDB2PQR: PROPKA protonation at pH 7 (MD preprocessing)
# ---------------------------------------------------------------------------

rule pdb2pqr:
    input:
        pdb = str(PRE / "{pdb}" / "{pdb}_{mut}" / "minimized.pdb"),
    output:
        pdb = str(PRE / "{pdb}" / "{pdb}_{mut}" / "protonated.pdb"),
        pqr = str(PRE / "{pdb}" / "{pdb}_{mut}" / "protonated.pqr"),
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "pdb2pqr.log"),
    shell:
        """
        {PDB2PQR} \
            --ff AMBER \
            --with-ph 7.0 \
            --titration-state-method propka \
            --pdb-output {output.pdb} \
            {input.pdb} {output.pqr} \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 5 — GROMACS topology (pdb2gmx, AMBER99SB-ILDN + TIP3P) (MD preprocessing)
# ---------------------------------------------------------------------------

rule pdb2gmx:
    input:
        pdb = str(PRE / "{pdb}" / "{pdb}_{mut}" / "protonated.pdb"),
    output:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "protein.gro"),
        top = str(PRE / "{pdb}" / "{pdb}_{mut}" / "topol_base.top"),
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "pdb2gmx.log"),
    shell:
        """
        ROOT=$(pwd)
        cd $(dirname {output.gro})
        gmx pdb2gmx \
            -f protonated.pdb \
            -o protein.gro \
            -p topol_base.top \
            -ff amber99sb-ildn \
            -water tip3p \
            -ignh \
            >> "$ROOT/{log}" 2>&1
        """


# ---------------------------------------------------------------------------
# Step 6 — Simulation box (triclinic, principal-axis aligned, 1.2 nm padding)
# ---------------------------------------------------------------------------

rule editconf:
    input:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "protein.gro"),
    output:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "boxed.gro"),
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "editconf.log"),
    shell:
        """
        printf 'Protein\n' | gmx editconf \
            -f {input.gro} \
            -o {output.gro} \
            -princ -c -d 1.2 -bt triclinic \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 7 — Solvation (TIP3P water)
# ---------------------------------------------------------------------------

rule solvate:
    input:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "boxed.gro"),
        top = str(PRE / "{pdb}" / "{pdb}_{mut}" / "topol_base.top"),
    output:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "solvated.gro"),
        top = str(PRE / "{pdb}" / "{pdb}_{mut}" / "topol_solv.top"),
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "solvate.log"),
    shell:
        """
        cp {input.top} {output.top}
        gmx solvate \
            -cp {input.gro} \
            -cs spc216.gro \
            -o {output.gro} \
            -p {output.top} \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 8 — Neutralization + 0.15 M NaCl
# ---------------------------------------------------------------------------

rule genion:
    input:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "solvated.gro"),
        top = str(PRE / "{pdb}" / "{pdb}_{mut}" / "topol_solv.top"),
        mdp = "mdp/em_strong.mdp",
    output:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "ions.gro"),
        top = str(PRE / "{pdb}" / "{pdb}_{mut}" / "topol_ions.top"),
        tpr = str(PRE / "{pdb}" / "{pdb}_{mut}" / "ions.tpr"),
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "genion.log"),
    shell:
        """
        cp {input.top} {output.top}
        ROOT=$(pwd)
        cd $(dirname {output.gro})
        gmx grompp \
            -f "$ROOT/{input.mdp}" \
            -c "$ROOT/{input.gro}" \
            -p topol_ions.top \
            -o ions.tpr \
            -maxwarn 1 \
            >> "$ROOT/{log}" 2>&1
        printf 'SOL' | gmx genion \
            -s ions.tpr \
            -o ions.gro \
            -p topol_ions.top \
            -pname NA -nname CL \
            -neutral -conc 0.15 \
            >> "$ROOT/{log}" 2>&1
        """


# ---------------------------------------------------------------------------
# Step 9 — GROMACS energy minimization (steepest descent, CPU)
# ---------------------------------------------------------------------------

rule gromacs_em:
    input:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "ions.gro"),
        top = str(PRE / "{pdb}" / "{pdb}_{mut}" / "topol_ions.top"),
        mdp = "mdp/em_strong.mdp",
    output:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "em.gro"),
        tpr = str(PRE / "{pdb}" / "{pdb}_{mut}" / "em.tpr"),
    threads: CPU_THREADS
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "em.log"),
    shell:
        """
        ROOT=$(pwd)
        cd $(dirname {output.gro})
        gmx grompp \
            -f "$ROOT/{input.mdp}" \
            -c ions.gro \
            -p topol_ions.top \
            -o em.tpr \
            -maxwarn 0 \
            >> "$ROOT/{log}" 2>&1
        gmx mdrun \
            -v -deffnm em \
            -ntmpi 1 -ntomp {threads} \
            >> "$ROOT/{log}" 2>&1
        """


# ---------------------------------------------------------------------------
# Step 10 — NVT equilibration (100 ps, position-restrained, CPU)
# ---------------------------------------------------------------------------

rule nvt:
    input:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "em.gro"),
        top = str(PRE / "{pdb}" / "{pdb}_{mut}" / "topol_ions.top"),
        mdp = "mdp/nvt.mdp",
    output:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "nvt.gro"),
        cpt = str(PRE / "{pdb}" / "{pdb}_{mut}" / "nvt.cpt"),
        tpr = str(PRE / "{pdb}" / "{pdb}_{mut}" / "nvt.tpr"),
    threads: CPU_THREADS
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "nvt.log"),
    shell:
        """
        ROOT=$(pwd)
        cd $(dirname {output.gro})
        gmx grompp \
            -f "$ROOT/{input.mdp}" \
            -c em.gro \
            -r em.gro \
            -p topol_ions.top \
            -o nvt.tpr \
            -maxwarn 0 \
            >> "$ROOT/{log}" 2>&1
        gmx mdrun \
            -v -deffnm nvt \
            -ntmpi 1 -ntomp {threads} \
            >> "$ROOT/{log}" 2>&1
        """


# ---------------------------------------------------------------------------
# Step 11 — NPT equilibration (100 ps, position-restrained, CPU)
#
#  This is the last step of "preprocessing" — npt.gro/.cpt is the fully
#  equilibrated system, ready for production MD.
# ---------------------------------------------------------------------------

rule npt:
    input:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "nvt.gro"),
        cpt = str(PRE / "{pdb}" / "{pdb}_{mut}" / "nvt.cpt"),
        top = str(PRE / "{pdb}" / "{pdb}_{mut}" / "topol_ions.top"),
        mdp = "mdp/npt.mdp",
    output:
        gro = str(PRE / "{pdb}" / "{pdb}_{mut}" / "npt.gro"),
        cpt = str(PRE / "{pdb}" / "{pdb}_{mut}" / "npt.cpt"),
        tpr = str(PRE / "{pdb}" / "{pdb}_{mut}" / "npt.tpr"),
    threads: CPU_THREADS
    log:
        str(PRE / "{pdb}" / "{pdb}_{mut}" / "logs" / "npt.log"),
    shell:
        """
        ROOT=$(pwd)
        cd $(dirname {output.gro})
        gmx grompp \
            -f "$ROOT/{input.mdp}" \
            -c nvt.gro \
            -r nvt.gro \
            -t nvt.cpt \
            -p topol_ions.top \
            -o npt.tpr \
            -maxwarn 0 \
            >> "$ROOT/{log}" 2>&1
        gmx mdrun \
            -v -deffnm npt \
            -ntmpi 1 -ntomp {threads} \
            >> "$ROOT/{log}" 2>&1
        """


# ---------------------------------------------------------------------------
# Checkpoint — sync the full preprocessing/ tree to object storage once every
# requested system has reached npt.gro. Every production_md job waits on this.
# Fill in config["s5cmd_dest"] before running; leave blank to skip the sync.
# ---------------------------------------------------------------------------

rule sync_preprocessing:
    input:
        expand(str(PRE / "{pdb}" / "{pdb}_{mut}" / "npt.gro"), zip, pdb=_PDBS, mut=_MUTS),
    output:
        touch(str(OUT / "preprocessing.synced")),
    params:
        dest = S5CMD_DEST,
        src  = str(PRE) + "/",
    log:
        str(OUT / "logs" / "sync_preprocessing.log"),
    shell:
        """
        if [ -z "{params.dest}" ]; then
            echo "s5cmd_dest not set in config.yaml — skipping object-storage sync" | tee {log}
        else
            {S5CMD} sync {params.src} {params.dest} > {log} 2>&1
        fi
        """


# ---------------------------------------------------------------------------
# Step 12 — Production MD (100 ns, GPU-accelerated)
# ---------------------------------------------------------------------------

rule production_md:
    input:
        gro    = str(PRE / "{pdb}" / "{pdb}_{mut}" / "npt.gro"),
        cpt    = str(PRE / "{pdb}" / "{pdb}_{mut}" / "npt.cpt"),
        top    = str(PRE / "{pdb}" / "{pdb}_{mut}" / "topol_ions.top"),
        mdp    = MDP_MD,
        synced = str(OUT / "preprocessing.synced"),
    output:
        gro = str(MD / "{pdb}" / "{pdb}_{mut}" / "md.gro"),
        xtc = str(MD / "{pdb}" / "{pdb}_{mut}" / "md.xtc"),
        tpr = str(MD / "{pdb}" / "{pdb}_{mut}" / "md.tpr"),
    threads: CPU_THREADS
    resources:
        gpu = 1,
    log:
        str(MD / "{pdb}" / "{pdb}_{mut}" / "logs" / "md.log"),
    shell:
        """
        ROOT=$(pwd)
        PREDIR=$(dirname {input.top})
        MDDIR=$(dirname {output.gro})
        TPR="$ROOT/$MDDIR/md.tpr"

        if [ ! -f "$TPR" ]; then
            cd "$PREDIR"
            gmx grompp \
                -f "$ROOT/{input.mdp}" \
                -c npt.gro \
                -t npt.cpt \
                -p topol_ions.top \
                -o "$TPR" \
                -maxwarn 0 \
                >> "$ROOT/{log}" 2>&1
            cd "$ROOT"
        fi

        cd "$MDDIR"
        CPI=""
        [ -f md.cpt ] && CPI="-cpi md.cpt"
        gmx mdrun \
            -v -deffnm md \
            -s md.tpr \
            $CPI \
            -ntmpi 1 -ntomp {threads} \
            -nb gpu -pme gpu -bonded gpu -update gpu \
            >> "$ROOT/{log}" 2>&1
        """


# ---------------------------------------------------------------------------
# Step 13 — Trajectory movie (solute only: strips water + ions, keeps protein
# and any glycans/ligands). trjconv makes molecules whole & centers, PyMOL
# ray-traces each frame, ffmpeg encodes to mp4.
# ---------------------------------------------------------------------------

rule movie:
    input:
        xtc = str(MD / "{pdb}" / "{pdb}_{mut}" / "md.xtc"),
        tpr = str(MD / "{pdb}" / "{pdb}_{mut}" / "md.tpr"),
    output:
        mp4 = str(MOVIES / "{pdb}_{mut}.mp4"),
    params:
        skip = int(config.get("movie_skip", 1)),
        fps  = int(config.get("movie_fps", 15)),
    log:
        str(MD / "{pdb}" / "{pdb}_{mut}" / "logs" / "movie.log"),
    shell:
        """
        MP4="{output.mp4}"
        STEM="${{MP4%.mp4}}"          # per-system scratch prefix (unique)
        FRAMES="$STEM.frames"
        NDX="$STEM.solute.ndx"
        SOLPDB="$STEM.solute.pdb"
        rm -rf "$FRAMES"

        # Solute = everything except water (SOL) and ions (NA/CL); keeps glycans.
        # Single unnamed group -> index 0 in the .ndx.
        gmx select -s {input.tpr} \
            -on "$NDX" \
            -select 'not resname SOL NA CL' \
            > {log} 2>&1

        # Strip solvent/ions, make molecules whole, center; optionally decimate.
        # Feed group 0 twice: centering group, then output group.
        {{ echo 0; echo 0; }} | gmx trjconv \
            -s {input.tpr} -f {input.xtc} -n "$NDX" \
            -o "$SOLPDB" \
            -pbc mol -center -skip {params.skip} \
            >> {log} 2>&1

        {PYMOL_PYTHON} scripts/render_movie.py \
            "$SOLPDB" "$FRAMES" \
            >> {log} 2>&1

        ffmpeg -y -framerate {params.fps} -i "$FRAMES/frame_%04d.png" \
            -c:v libx264 -pix_fmt yuv420p \
            -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' \
            {output.mp4} >> {log} 2>&1

        rm -rf "$FRAMES" "$SOLPDB" "$NDX"
        """

