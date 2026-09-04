# Snakefile — MD pipeline: PDB prep + MD preprocessing + production MD
#
# Covers:
#   PDB structure
#     -> extract chains (Python)                       } PDB prep
#     -> vacuum minimize (OpenMM, GPU)
#     -> PDB2PQR pH 7 protonation
#     -> pdb2gmx AMBER99SB-ILDN
#     -> editconf (triclinic, 1.2 nm)
#     -> solvate (TIP3P)                               } MD preprocessing
#     -> genion (0.15 M NaCl, neutral)
#     -> GROMACS EM (steepest descent)
#     -> NVT equilibration (100 ps)
#     -> NPT equilibration (100 ps)
#     -> production MD (md_ns / md_steps, GPU)         } production MD
#
# Input:
#   - a structures list (config key `structures_file`, default structures.yaml,
#     or an inline `structures:` list in config.yaml): one entry per structure
#     to run MD on ({name, path}, with optional per-structure `chains` and an
#     optional `md_ns`/`md_steps` that overrides the global default).
#     See scripts/build_systems.py.
#   - config["md_ns"] or config["md_steps"]: default production-MD length;
#     the base .mdp's `nsteps` is rewritten per system (rule write_md_mdp).
#
# Output layout (under config["output_dir"], default "results/"):
#   preprocessing/{system}/   — chain extraction through NPT equilibration
#   MD/{system}/              — production MD only
#   params/{system}/md.mdp    — production .mdp with that system's length
#   manifest.json             — parsed system manifest (audit)
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
#       results/MD/1bj1fv/md.gro
# Slurm:
#   sbatch slurm/submit_pipeline.sbatch

import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from build_systems import build_systems, load_structures

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
MDP_MD_BASE    = config.get("mdp_md", "mdp/md.mdp")

MAKE_MOVIE     = config.get("make_movie", True)
if isinstance(MAKE_MOVIE, str):
    MAKE_MOVIE = MAKE_MOVIE.strip().lower() not in ("false", "0", "no", "off", "")

# One MD system per structures-list entry. Each carries exactly one of
# md_ns / md_steps (per-structure override, else the global default from
# config.yaml); build_systems raises if any system ends up with neither.
SYSTEMS = build_systems(
    load_structures(config),
    default_md_ns=config.get("md_ns"),
    default_md_steps=config.get("md_steps"),
)
if not SYSTEMS:
    raise ValueError("No systems parsed — check the structures list.")
SYSTEM_NAMES = list(SYSTEMS)

# {system} is the manifest key — validated in build_systems to be a single
# filesystem-safe path segment, so it can never escape the output tree.
wildcard_constraints:
    system = r"[A-Za-z0-9][A-Za-z0-9._-]*",


def sys_data(wc) -> dict:
    return SYSTEMS[wc.system]


def md_length_args(system: str) -> str:
    """--ns/--steps argument for a system's production length."""
    entry = SYSTEMS[system]
    return (f"--ns {entry['md_ns']}" if entry["md_ns"] is not None
            else f"--steps {entry['md_steps']}")


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

_TARGETS = [str(MD / s / "md.gro") for s in SYSTEM_NAMES]
if MAKE_MOVIE:
    _TARGETS += [str(MOVIES / f"{s}.mp4") for s in SYSTEM_NAMES]


rule all:
    input:
        _TARGETS,
        str(OUT / "manifest.json"),


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
# Step 0b — Production .mdp per system, with that system's run length
# (per-structure md_ns / md_steps, else the global default)
# ---------------------------------------------------------------------------

rule write_md_mdp:
    input:
        base = MDP_MD_BASE,
    output:
        mdp = str(OUT / "params" / "{system}" / "md.mdp"),
    params:
        length = lambda wc: md_length_args(wc.system),
    log:
        str(OUT / "params" / "{system}" / "write_md_mdp.log"),
    shell:
        """
        {GMX_PYTHON} scripts/make_md_mdp.py \
            {input.base} {output.mdp} {params.length} \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 1 — Extract the relevant chains from the input PDB (PDB prep)
# ---------------------------------------------------------------------------

rule extract_chains:
    input:
        pdb = lambda wc: sys_data(wc)["pdb_file"],
    output:
        pdb = str(PRE / "{system}" / "extracted.pdb"),
    params:
        chains = lambda wc: " ".join(sys_data(wc)["chains"]),
    log:
        str(PRE / "{system}" / "logs" / "extract_chains.log"),
    shell:
        """
        {GMX_PYTHON} scripts/extract_chains.py \
            {input.pdb} {output.pdb} {params.chains} \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 2 — OpenMM vacuum energy minimization, GPU (MD preprocessing)
# ---------------------------------------------------------------------------

rule openmm_minimize:
    input:
        pdb = str(PRE / "{system}" / "extracted.pdb"),
    output:
        pdb = str(PRE / "{system}" / "minimized.pdb"),
    resources:
        gpu = 1,
    log:
        str(PRE / "{system}" / "logs" / "openmm_minimize.log"),
    shell:
        """
        {OPENMM_PYTHON} scripts/minimize_openmm.py \
            {input.pdb} {output.pdb} \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 3 — PDB2PQR: PROPKA protonation at pH 7 (MD preprocessing)
# ---------------------------------------------------------------------------

rule pdb2pqr:
    input:
        pdb = str(PRE / "{system}" / "minimized.pdb"),
    output:
        pdb = str(PRE / "{system}" / "protonated.pdb"),
        pqr = str(PRE / "{system}" / "protonated.pqr"),
    log:
        str(PRE / "{system}" / "logs" / "pdb2pqr.log"),
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
# Step 4 — GROMACS topology (pdb2gmx, AMBER99SB-ILDN + TIP3P) (MD preprocessing)
# ---------------------------------------------------------------------------

rule pdb2gmx:
    input:
        pdb = str(PRE / "{system}" / "protonated.pdb"),
    output:
        gro = str(PRE / "{system}" / "protein.gro"),
        top = str(PRE / "{system}" / "topol_base.top"),
    log:
        str(PRE / "{system}" / "logs" / "pdb2gmx.log"),
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
# Step 5 — Simulation box (triclinic, principal-axis aligned, 1.2 nm padding)
# ---------------------------------------------------------------------------

rule editconf:
    input:
        gro = str(PRE / "{system}" / "protein.gro"),
    output:
        gro = str(PRE / "{system}" / "boxed.gro"),
    log:
        str(PRE / "{system}" / "logs" / "editconf.log"),
    shell:
        """
        printf 'Protein\n' | gmx editconf \
            -f {input.gro} \
            -o {output.gro} \
            -princ -c -d 1.2 -bt triclinic \
            > {log} 2>&1
        """


# ---------------------------------------------------------------------------
# Step 6 — Solvation (TIP3P water)
# ---------------------------------------------------------------------------

rule solvate:
    input:
        gro = str(PRE / "{system}" / "boxed.gro"),
        top = str(PRE / "{system}" / "topol_base.top"),
    output:
        gro = str(PRE / "{system}" / "solvated.gro"),
        top = str(PRE / "{system}" / "topol_solv.top"),
    log:
        str(PRE / "{system}" / "logs" / "solvate.log"),
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
# Step 7 — Neutralization + 0.15 M NaCl
# ---------------------------------------------------------------------------

rule genion:
    input:
        gro = str(PRE / "{system}" / "solvated.gro"),
        top = str(PRE / "{system}" / "topol_solv.top"),
        mdp = "mdp/em_strong.mdp",
    output:
        gro = str(PRE / "{system}" / "ions.gro"),
        top = str(PRE / "{system}" / "topol_ions.top"),
        tpr = str(PRE / "{system}" / "ions.tpr"),
    log:
        str(PRE / "{system}" / "logs" / "genion.log"),
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
# Step 8 — GROMACS energy minimization (steepest descent, CPU)
# ---------------------------------------------------------------------------

rule gromacs_em:
    input:
        gro = str(PRE / "{system}" / "ions.gro"),
        top = str(PRE / "{system}" / "topol_ions.top"),
        mdp = "mdp/em_strong.mdp",
    output:
        gro = str(PRE / "{system}" / "em.gro"),
        tpr = str(PRE / "{system}" / "em.tpr"),
    threads: CPU_THREADS
    log:
        str(PRE / "{system}" / "logs" / "em.log"),
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
# Step 9 — NVT equilibration (100 ps, position-restrained, CPU)
# ---------------------------------------------------------------------------

rule nvt:
    input:
        gro = str(PRE / "{system}" / "em.gro"),
        top = str(PRE / "{system}" / "topol_ions.top"),
        mdp = "mdp/nvt.mdp",
    output:
        gro = str(PRE / "{system}" / "nvt.gro"),
        cpt = str(PRE / "{system}" / "nvt.cpt"),
        tpr = str(PRE / "{system}" / "nvt.tpr"),
    threads: CPU_THREADS
    log:
        str(PRE / "{system}" / "logs" / "nvt.log"),
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
# Step 10 — NPT equilibration (100 ps, position-restrained, CPU)
#
#  This is the last step of "preprocessing" — npt.gro/.cpt is the fully
#  equilibrated system, ready for production MD.
# ---------------------------------------------------------------------------

rule npt:
    input:
        gro = str(PRE / "{system}" / "nvt.gro"),
        cpt = str(PRE / "{system}" / "nvt.cpt"),
        top = str(PRE / "{system}" / "topol_ions.top"),
        mdp = "mdp/npt.mdp",
    output:
        gro = str(PRE / "{system}" / "npt.gro"),
        cpt = str(PRE / "{system}" / "npt.cpt"),
        tpr = str(PRE / "{system}" / "npt.tpr"),
    threads: CPU_THREADS
    log:
        str(PRE / "{system}" / "logs" / "npt.log"),
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
        [str(PRE / s / "npt.gro") for s in SYSTEM_NAMES],
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
# Step 11 — Production MD (length from md_ns / md_steps, GPU-accelerated)
# ---------------------------------------------------------------------------

rule production_md:
    input:
        gro    = str(PRE / "{system}" / "npt.gro"),
        cpt    = str(PRE / "{system}" / "npt.cpt"),
        top    = str(PRE / "{system}" / "topol_ions.top"),
        mdp    = str(OUT / "params" / "{system}" / "md.mdp"),
        synced = str(OUT / "preprocessing.synced"),
    output:
        gro = str(MD / "{system}" / "md.gro"),
        xtc = str(MD / "{system}" / "md.xtc"),
        tpr = str(MD / "{system}" / "md.tpr"),
    threads: CPU_THREADS
    resources:
        gpu = 1,
    log:
        str(MD / "{system}" / "logs" / "md.log"),
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
# Step 12 — Trajectory movie (solute only: strips water + ions, keeps protein
# and any glycans/ligands). trjconv makes molecules whole & centers, PyMOL
# ray-traces each frame, ffmpeg encodes to mp4.
# ---------------------------------------------------------------------------

rule movie:
    input:
        xtc = str(MD / "{system}" / "md.xtc"),
        tpr = str(MD / "{system}" / "md.tpr"),
    output:
        mp4 = str(MOVIES / "{system}.mp4"),
    params:
        skip = int(config.get("movie_skip", 1)),
        fps  = int(config.get("movie_fps", 15)),
    log:
        str(MD / "{system}" / "logs" / "movie.log"),
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
