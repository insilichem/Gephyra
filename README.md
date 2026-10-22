# Gephyra

Gephyra (derived from the Ancient Greek word for "bridge") is a network-graph and Fréchet clustering framework engineered for the discovery, spatial tracking, and statistical analysis of functional, macromolecularly confined water-bridge channels in molecular dynamics trajectories.

## Features

- **Multi-order water bridge discovery:** A depth-bounded graph traversal
  originating from a user-defined root residue identifies water bridges
  up to `max_depth` consecutive waters into the solvent.

- **Continuous probabilistic scoring:** Replaces binary geometric cutoffs
  with fractional switching functions that evaluate:
  - Heavy-atom O···O distance
  - Angular donor-hydrogen-acceptor alignment via a triangle-inequality proxy
  - Steric repulsion for clash geometries (r_OO < 2.40 Å)
  - Chemical specificity: per-element r0 reference distances for N, O, S

- **Weighted path enumeration:** Edge probabilities are converted to
  logarithmic weights so that path traversal naturally ranks higher-probability
  chains above weaker ones within the depth limit.

- **Temporal occupancy statistics:** Tracks which specific water-molecule
  chains appear in each frame and reports the top 20 paths ranked by
  frame-count occupancy.

- **Pathway clustering:** Optional post-processing groups spatially similar
  paths across frames in a two-stage process: first, a memory-efficient 9D-Vector
  coarse screening pass groups identical spatial channels. Then, a rigorous
  average-link hierarchical clustering using the Fréchet distance metric on full
  3D coordinates determines the final topological groups. Every unique path
  topology enters the clustering; the only frequency criterion is applied
  afterwards, where a merged cluster whose frames union to fewer than two frames
  is discarded as thermal noise. The step also applies a hard safety cap to
  prevent RAM exhaustion, and uses geometric medoid selection to avoid occupancy
  bias. This produces a
  `clustered_pathways.json` file with cluster size, occupancy, average H-quality,
  persistence statistics (`mean_persistence_frames` and `max_persistence_frames`
  accounting for stride), and a representative full-coordinate medoid geometry.

- **H-quality score (Z):** Every path is reported with an *H-quality*, not a
  probability.

  * **How it is calculated:** For a given spatial endpoint, the algorithm
    enumerates every route (unique atomic permutation) of water chains
    connecting the root to that endpoint. H-quality is the sum of the
    exponentially weighted route scores, Z = Σ exp(−w), where w is the
    accumulated, cooperativity-discounted −log penalty of the route's bonds
    (derived from the distance, angle and steric switching functions). The
    individual *bond* scores are in [0, 1]; their aggregate Z is not.
  * **How to interpret Z:**
    * *Values can exceed 1.0.* Highly branched or degenerate channels with many
      parallel routes accumulate Z > 1. Z is unbounded and is **not a
      probability**.
    * *Physical meaning.* A higher Z indicates topological degeneracy and
      geometric robustness: how many favourable, pre-organised hydrogen-bond
      routes exist within that channel.
  * **Normalised H-quality.** For plots, thresholds and comparisons between
    systems, each value also comes with a bounded companion in [0, 1):
    `hquality_norm = Z / (1 + Z)`. This is the weight of the bridged channel
    against a single reference state of weight 1 (no bridge). The map is
    monotonic, so rankings are identical to those on Z; it equals 0.5 at Z = 1
    and saturates for very degenerate channels. It is a bounded score, not a
    measured probability. The helper is `gephyra.math_utils.normalize_hquality`.
  * **Where it appears:**

    | File | Raw Z | Normalised |
    |---|---|---|
    | `.jsonl` (each path) | `hquality` | `hquality_norm` |
    | `.csv` | `HQuality_Z` | `HQuality_Norm` |
    | `clustered_pathways.json` | `avg_hquality` | `avg_hquality_norm` |

    In clusters, `avg_hquality_norm` is the mean of the per-occurrence
    normalised values, not the normalisation of `avg_hquality`.
  * **Upgrading from 0.2.x.** Earlier versions wrote the same Z under the
    misleading names `probability` (`.jsonl`), `Cumulative_Probability` (CSV)
    and `avg_prob_Hquality` (clusters). Those names are no longer written;
    update downstream scripts to the names above. `gephyra cluster` still
    reads old `.jsonl` files that use `probability`, so existing results do not
    need to be recalculated.

- **Multi-phase execution:**
  - **Phase 1 — `calculate`:** Processes trajectory frames via MDAnalysis
    and NetworkX. Streams output as JSON Lines (`.jsonl`) and optional `.csv`
    to avoid RAM exhaustion on long trajectories.
  - **Phase 1.5 — `cluster`:** Post-processes `.jsonl` outputs separating
    geometric calculation and analytical clustering to preserve memory while extracting
    high-level collective behaviours.
  - **Phase 2 — `visualize`:** Writes ready-to-run *scenes* for UCSF
    ChimeraX (`.cxc`), UCSF Chimera (`.py`), PyMOL (`.py`) and VMD (`.tcl`).
    Gephyra re-opens the analysed trajectory, writes a PDB snapshot of the
    protein, ligand and bridging water molecules, and the script loads it and
    styles the whole picture by itself: protein as cartoon, ligand and waters
    as sticks, water bridges as dashed bonds or colour-coded tubes.

- **Graphical interface:** `gephyra gui` opens a desktop window (dark or light
  theme) that runs the whole workflow without memorising command-line flags.
  A step-by-step PDF manual is in [`docs/Gephyra-GUI-Manual.pdf`](docs/Gephyra-GUI-Manual.pdf).

## Installation

Gephyra requires **Python 3.10 or newer; 3.11 is recommended**. Install it in a dedicated conda environment,
so that the Python version and the libraries it needs do not clash with other
software on your machine (for example a PyMOL installation, which ships its own
Python):

```bash
git clone https://github.com/insilichem/Gephyra.git
cd Gephyra
conda create -n gephyra -c conda-forge python=3.11 "tk=*=xft*"
conda activate gephyra
pip install .
```

Or create the same environment, with the scientific libraries taken from
conda-forge, from the file in the repository:

```bash
git clone https://github.com/insilichem/Gephyra.git
cd Gephyra
conda env create -f environment.yml
conda activate gephyra
pip install .
```

`"tk=*=xft*"` selects conda-forge's Tk build with anti-aliased fonts. The
default build only has bitmap fonts, which makes the graphical interface look
blocky. Check the installation with `gephyra --version`, and remember to run
`conda activate gephyra` in every new terminal before using Gephyra.

For development, install in editable mode with the test dependencies:

```bash
pip install -e ".[test]"
```

Run the test suite with `pytest` (or `python -m unittest discover -s tests`).
Interface tests need a display and are skipped without one (`xvfb-run pytest`
runs them headless). Set `GEPHYRA_TEST_CHIMERAX=1` with `chimerax` on your
`PATH` to also open the generated ChimeraX scripts in a real, windowless
ChimeraX session, and `GEPHYRA_TEST_VIEWERS=1` to run the generated scene
scripts in every installed viewer (ChimeraX, Chimera, PyMOL, VMD) and check
them for errors.

## Graphical interface

```bash
gephyra gui        # or: gephyra-gui
```

The full walkthrough, with screenshots, is the PDF manual
[`docs/Gephyra-GUI-Manual.pdf`](docs/Gephyra-GUI-Manual.pdf). In short:

- **Three steps in the sidebar.** **1. Pathway Analysis**, **2. Clustering** and
  **3. Visualization**. Every command-line option is a form field with the same
  default (the topology *and* the trajectory are required). Hover over a field
  for help; *Browse* opens a file picker.
- **Output folder.** Shown in the sidebar and set to the folder Gephyra was
  started from. *Change…* picks another one before a run, *Open* shows it in
  your file manager. Output names in the forms are relative to it (an absolute
  path overrides it), and a folder that does not exist yet can be created on
  the spot.
- **Clear confirmation.** When a step finishes, a pop-up says so, summarises
  the result (paths found, clusters identified, how to open the script) and
  lists every file written with the folder they are in. From there you can open
  the folder, *Save a copy to…* another location, or go straight to the next step.
- **Validation before running.** Values are checked with the same rules as the
  CLI. A problem opens a dialog naming the field and moves the cursor to it.
- **Live feedback.** The progress bar tracks frames (analysis) or distance-matrix
  pairs (clustering), the log streams messages as they happen, and *Cancel*
  stops after the current frame. Errors appear as short messages; the full
  traceback is available with `gephyra -v gui`.
- **Chained steps and reproducibility.** A finished step fills in the next one's
  input. Each run writes the equivalent `gephyra …` command to the log, and
  *Copy CLI command* puts it on the clipboard for scripts or HPC jobs.
- **Light on resources.** The interface uses only `tkinter`, which ships with
  Python. The theme choice is remembered in `~/.config/gephyra/gui.json`.

On some Linux distributions tkinter is a separate system package (for example
`sudo apt install python3-tk`). If the text looks blocky, Gephyra is running in a
Python whose Tk was built without anti-aliased fonts (the interface says so in
its log). This does not happen in the environment from [Installation](#installation);
in an existing conda environment, fix it with:

```bash
conda install -c conda-forge "tk=*=xft*"
```

## Usage

Global options go before the sub-command: `gephyra --version`, `gephyra -v …`
(debug logging) and `gephyra -q …` (warnings and errors only). Numerical
arguments are validated up front: for example, `--stride 0`, a `--cooperativity`
outside (0, 1], or a `--min_depth` larger than `--max_depth` are rejected before
any trajectory is loaded. `python -m gephyra` is equivalent to `gephyra`.

### 1. Calculate

```bash
gephyra calculate \
  --topo  my_topology.pdb \
  --traj  my_trajectory.xtc \
  --root  "resname LIG and name O1" \
  --stride 10 \
  --max_depth 10 \
  --min_depth 3 \
  --coarse_cutoff 4.5 \
  --output results.jsonl \
  --csv   summary.csv
```

| Option | Default | Description |
|---|---|---|
| `--topo` | required | Topology file (.pdb, .tpr, …) |
| `--traj` | required | Trajectory file (.xtc, .dcd, .nc, …). To analyse a single structure, pass a coordinate file such as a `.pdb`. The topology and trajectory paths are recorded in the output so that `visualize` can rebuild the molecular scene later. |
| `--root` | required | MDAnalysis selection string for the root atom(s) (e.g. `"resname LIG"`). |
| `--water` | `"resname SOL or resname WAT or resname HOH"` | Solvent selection string. |
| `--stride` | `1` | Process every Nth frame. A warning is issued when `stride=1` and the trajectory exceeds 1000 frames. |
| `--max_depth` | `10` | Maximum number of sequential water molecules in a path. |
| `--min_depth` | `1` | Minimum number of sequential water molecules in a path; shorter paths are discarded from output. |
| `--cooperativity` | `0.92` | Geometric discount applied to successive hydrogen bonds along a chain: the *k*-th bond contributes its weight scaled by `cooperativity^(k-1)`. This models the polarisation that propagates along a water wire, which makes a bond embedded in a chain stronger than the same geometry in isolation. Values below 1.0 favour longer chains; `1.0` disables the effect and recovers the plain product of edge probabilities. |
| `--coarse_cutoff` | `4.5` | Distance cutoff in Å for initial neighbour graph construction. |
| `--align` | `auto` | Atoms used to superpose every frame onto the first analysed frame before path coordinates are stored (`auto`: protein Cα, or nucleic-acid P). This puts the same channel at the same coordinates in every frame even when the solute drifts or rotates through the box, which clustering needs. The choice is recorded in the output and reused by `visualize`. |
| `--no_align` | off | Store path coordinates exactly as in the trajectory. Only for trajectories that are already fitted. |
| `--output` | `results.jsonl` | Output file for full frame-by-frame path data (JSON Lines format). |
| `--csv` | — | Optional human-readable CSV summary of detected paths (columns `Frame`, `Root_Residue`, `Path_Atom_Indices`, `Path_Length`, `HQuality_Z`, `HQuality_Norm`, `Average_OO_Distance`). |


### 1.5. Cluster

```bash
# Cluster output pathways into collective behaviour groups
gephyra cluster \
  --data results.jsonl \
  --threshold 6.0 \
  --output clustered_pathways.json
```

| Option | Default | Description |
|---|---|---|
| `--data` | required | The `.jsonl` file produced by `calculate`. |
| `--threshold` | `6.0` | Discrete Fréchet distance threshold in Å for fine clustering. |
| `--coarse_trigger` | `1000` | Number of paths below which the 9D coarse filter is bypassed for direct Fréchet calculation. |
| `--coarse_threshold` | `threshold / sqrt(3)` | 9D Feature Vector distance threshold in Å for the coarse screening pass. |
| `--max_paths` | `60000` | Hard cap on number of paths sent to the distance matrix. |
| `--output` | `clustered_pathways.json` | Output JSON file for the cluster summary. |

The cluster file is `{"type": "gephyra_clusters", "metadata": …, "clusters": […]}`.
`metadata` records the analysed system (topology, trajectory, selections) and
the clustering parameters, and each cluster gains `medoid_nodes`, the atom
indices of its medoid path at `medoid_frame`. Read it with
`gephyra.results.load_clusters(path)`, which returns `(clusters, metadata)` and
also accepts the plain list written by earlier versions.

The Fréchet matrix is computed with a vectorised implementation of the
Eiter–Mannila discrete Fréchet recurrence (`gephyra.math_utils.pairwise_discrete_frechet`).
It returns exactly the same values as `similaritymeasures.frechet_dist` but
processes all pairs with the same pair of path lengths in one NumPy pass, which
is one to two orders of magnitude faster. Medoids are read straight from the
condensed distance vector, so memory use is `8 · N(N−1)/2` bytes rather than
twice that for a full square matrix. The rule that a cluster must span at least
two frames is applied uniformly, including when only one path topology remains.

### 2. Visualize

```bash
# Cluster medoids around the protein (chimerax, chimera, pymol or vmd)
gephyra visualize --data clustered_pathways.json --format chimerax --mode cluster \
  --top_clusters 10 --output cluster_scene

# One cluster rebuilt as real molecules at its medoid frame
gephyra visualize --data clustered_pathways.json --format pymol --mode cluster \
  --cluster_id 1 --output cluster1

# A single frame: the actual water molecules and their H-bond bridges
gephyra visualize --data results.jsonl --format chimera --mode frame --frame 400 \
  --output frame_400

# Every path of every frame, overlaid
gephyra visualize --data results.jsonl --format vmd --mode density --output density
```

Each command writes the script, a `<name>_structure.pdb` snapshot next to it,
and, depending on the viewer, a `.bild` (geometry) and a `.pb` (pseudobonds)
file. **The script loads everything itself.** Do not open the structure
beforehand; run the script directly:

| Viewer | How to open the script |
|---|---|
| UCSF ChimeraX | `chimerax scene.cxc`, drag it onto the window, or `open scene.cxc` |
| UCSF Chimera | `chimera scene.py`, or *File › Open…* |
| PyMOL | `pymol scene.py`, or `run scene.py` inside PyMOL |
| VMD | `vmd -e scene.tcl`, or `source scene.tcl` in the Tk Console. Note the `-e`: `vmd scene.tcl` makes VMD try to read the script as a molecule file. |

> **VMD scripts must be run with `vmd -e script.tcl`.** Without `-e`, VMD treats
> the `.tcl` file as a molecule, fails with *"Unable to determine file type …
> Assuming pdb"*, and draws nothing. Inside a running VMD, use
> `source script.tcl` in *Extensions › Tk Console* instead.

Re-running a script replaces the previous Gephyra scene instead of piling up copies.

| Mode | What the scene shows |
|---|---|
| `frame` | Protein (cartoon) and ligand at that frame, the bridging water molecules (O and H), and every water bridge as a dashed bond between the real atoms. |
| `cluster` | Protein and ligand at the medoid frame of the top cluster, with each cluster's medoid path as a colour-coded tube and a legend with cluster ID, occupancy and size. With `--cluster_id N`, that one cluster is rebuilt as real molecules at its own medoid frame, like `frame` mode. |
| `density` | Protein and ligand at the first analysed frame, with every path segment of every frame as thin, transparent tubes. |

How the snapshot is built:

- **What is included.** The context is everything except solvent and common
  counter-ions (`--context` to choose another MDAnalysis selection), plus the
  root residue and, in `frame` / single-cluster mode, the bridging water
  molecules.
- **Periodic boundaries.** Waters are moved, as whole molecules, to the periodic
  image next to their neighbour on the path, so bridges never appear torn across
  the box. The solute keeps its trajectory coordinates, so use an imaged/centred
  trajectory (`cpptraj autoimage`, `gmx trjconv -pbc mol -center`), as for any
  analysis.
- **Clean picture.** Massless virtual sites (TIP4P/TIP5P extra points) are left
  out. In topologies without chain IDs (such as AMBER `.prmtop`) each protein
  molecule gets its own chain, so viewers do not join two monomers with a fake
  "missing segment". The root residue and the bridging waters get chains of
  their own (`Y` and `W` unless those are taken), which is how the scripts
  address them.
- **Cofactors and metal centres.** Every solute residue that is not a standard
  amino acid or nucleotide (cofactors, metal complexes, renamed metal-binding
  residues) is drawn as sticks, and single-atom metals as spheres, even when
  the topology bonds them to the protein (as metal-centre parameterisations
  do) and they therefore share its chain. They are written as `HETATM`
  records and included in the camera framing.

| Option | Default | Description |
|---|---|---|
| `--data` | required | `--mode cluster` expects the cluster file written by `cluster`; `--mode density` and `--mode frame` expect the `.jsonl` written by `calculate`. Passing the wrong one is reported as an error. |
| `--format` | `vmd` | `chimerax` (`.cxc`), `chimera` (`.py`), `pymol` (`.py`) or `vmd` (`.tcl`). |
| `--mode` | `density` | `frame`, `cluster` or `density` (see above). |
| `--frame` | — | Trajectory frame index for `--mode frame`. If omitted, the first frame with water bridges is drawn. If the frame was not analysed (skipped by `--stride`) or has no bridges, the warning lists the frames that do have them (also the `Frame` column of the CSV summary). |
| `--cluster_id` | — | Draw only this cluster (its sequential rank), rebuilt as molecules. |
| `--top_clusters` | all | Draw only the N most occupied clusters. |
| `--topo`, `--traj` | from the data file | The analysed system. Only needed for older result files that do not record them, or if the files have moved. |
| `--context` | solute | MDAnalysis selection for the molecules shown around the network. |
| `--no_structure` | off | Write geometry only and load a matching structure in the viewer yourself (the behaviour of older versions). |
| `--max_bond_draw_dist` | `6.0` | Segments longer than this (Å) are treated as periodic-boundary artifacts and skipped. |
| `--chimerax_model` | `#101` / `#1` | ChimeraX model id. Scene scripts open the structure at `#101` (geometry at `#102`, bridges at `#103`). Geometry-only scripts address the structure you opened yourself, by default `#1`. |
| `--output` | `pathways_viz` | Output script name; the viewer's extension is added if missing. The output folder must exist. |

**Geometry-only scripts** (`--no_structure`, or old result files with no
recorded topology) behave as in earlier versions: open the analysed structure
first, then the script. In ChimeraX, atoms are then addressed as
`#1@@serial_number=N` (the topology's atom IDs), so the structure must be model
`#1` (or pass `--chimerax_model`), and a multi-model PDB trajectory must be
opened as one model with `open traj.pdb coordset true`.

## Solvent naming conventions

The default water selection covers GROMACS (`SOL`), AMBER (`WAT`), and
CHARMM/PDB (`HOH`) residue names. For non-standard solvents or co-solvents
acting as bridge donors, pass a custom `--water` string, for example:

```bash
--water "resname SOL or resname GOL"
```

For united-atom force fields (GROMOS, OPLS-UA) or coarse-grained models,
consult the force-field documentation for the correct oxygen atom names
and verify that hydrogens are present in the topology.

### What `--water` controls, and why it defines the scope

The traversal starts at `--root` and every subsequent node is drawn from the
`--water` selection. Nothing else can ever appear in a path. Gephyra therefore
traces **root-to-solvent chains**: it answers "which solvent-mediated chains
leave this root", not "which two solutes are bridged".

The practical consequence is that a bridge terminating on a second solute — a
ligand-water-water-protein contact, say, or a water wire between two subunits —
will be traced only as far as the last water. The protein or second ligand atom
at the far end is invisible to the traversal unless you put it in `--water`:

```bash
# Trace ligand -> water -> ... -> protein side-chain oxygens/nitrogens
gephyra calculate \
  --topo my_topology.pdb --traj my_trajectory.xtc \
  --root  "resname LIG and name O1" \
  --water "resname SOL or (protein and name O* N*)"
```

The selection is a plain MDAnalysis string, so anything selectable can act as an
intermediate node. Widening it widens the graph, so expect longer run times and
more paths, and remember that any atom you add is treated as a potential bridge
node rather than as an endpoint: paths may continue *through* it.

## Limitations

### What the tool is designed for

- Identifying **water bridge networks** connecting a root residue to bulk
  solvent across one or more water molecules in standard all-atom MD.
- Ranking bridges by **temporal occupancy** (fraction of frames in which
  a given atom-index chain appears).
- Generating **spatial cluster representatives** for bridges that recur
  across many frames.
- Producing **overlay visualisations** of where water-mediated interactions
  occur near a binding site or protein surface. Note that `--mode density`
  draws every path segment from every analysed frame transparently on top of
  one another; it does not compute a grid, an occupancy volume, or any other
  volumetric density. Dense regions read as dense because many segments overlap
  there, and the result is not quantitative.

### What the tool does not model

**Solute imaging across periodic boundaries.** Path coordinates are stored
with each water in the periodic image nearest to its predecessor on the chain,
starting from the root, so paths are never torn across the box (older result
files stored raw wrapped positions). Frames are then superposed on the protein
(`--align`), so a solute that drifts or tumbles through the box is handled. The
solute itself is not re-imaged, though: if the protein is split across the box
in some frames, the superposition is poor there (Gephyra warns when its RMSD is
large) and the protein is drawn split. Image the trajectory before analysis
(`cpptraj autoimage`, `gmx trjconv -pbc mol`).

**Water molecule exchange along a pathway.** Occupancy is tracked by
exact atom-index tuples. If the same physical channel is traversed by
different water molecules at different times — common in trajectories
longer than a few nanoseconds — each unique permutation is counted as a
separate path. Temporal occupancy will be underestimated for highly
dynamic pathways. The spatial clustering step partially compensates
for this.

**Proton-conducting (Grotthuss) water wires.** The algorithm is
undirected: it detects chains of mutually compatible H-bond geometries
but does not verify that water dipoles are aligned head-to-tail, which
is the physical requirement for proton transport. An antiparallel pair
that blocks conductance is geometrically indistinguishable from a
conducting pair in the current model. If you are studying aquaporins,
gramicidin channels, or other systems where proton-wire directionality
matters, the occupancy output should be interpreted as a measure of
**structural presence**, not **transport competence**.

**Transmembrane pathways crossing the periodic boundary.** Wires
spanning the full membrane thickness cross periodic images. Because
path coordinates are not minimum-image corrected, these wires will
not be reliably detected or visualised.

**Consecutive-frame persistence.** A path that appears in 100 frames
scattered across a 10 000-frame trajectory receives the same
occupancy score (0.01) as a path present in 100 consecutive frames.
Only the latter constitutes a genuinely persistent structural feature.
If persistence matters for your analysis, run `gephyra cluster`, which groups
paths by spatial similarity and reports per-cluster occupancy together with
`mean_persistence_frames` and `max_persistence_frames` (the mean and longest
runs of consecutive analysed frames).

**Very large path counts.** The clustering step uses a hard maximum cap (`max_paths=60000`)
and requires a fully merged cluster to appear in at least two frames to prevent memory
bottlenecks and filter out raw thermal noise. For long trajectories with extreme branching,
the clustering step may truncate the least frequent paths. Use
`--stride` to reduce frame count or increase `--min_depth` to
reduce the number of short paths before running `gephyra cluster`.


## Method and provenance

Gephyra's hydrogen-bond criteria are adapted from, but not identical to, those
of MDAnalysis. Being explicit about which is which matters when comparing
Gephyra's output against other tools.

**Taken from MDAnalysis.** The donor and acceptor heavy-atom names used as a
fallback when a topology carries no `element` attribute (`_NAME_TO_ELEMENT` in
`core.py`) are the CHARMM27 donor/acceptor lists shipped with
`MDAnalysis.analysis.hydrogenbonds.wbridge_analysis.WaterBridgeAnalysis`.
Gephyra adds `O1`, `O2`, `OW1` and `OXT` to cover ligand, non-standard solvent
and C-terminal naming. All neighbour searching and distance evaluation is done
with MDAnalysis' periodic-aware `capped_distance` and `distance_array`.

**Gephyra's own.** Everything that converts geometry into a score is specific to
this package. `WaterBridgeAnalysis` accepts or rejects a bond with hard cutoffs —
a single 3.0 Å donor-acceptor distance and a fixed donor-H-acceptor angle, applied
identically to every element pair. Gephyra instead scores each candidate
continuously, using these empirical constants:

| Constant | Where | Value | Role |
|---|---|---|---|
| Reference distance r₀ | `core.py` | 2.80 Å (O–O), 2.9 Å (N–O), 3.0 Å (N–N), 3.3 Å (S, Cl), 3.4 Å (Br), 2.9 Å (F) | Heavy-atom separation at which the distance term starts to decay. |
| Switching width Δ | `core.py` | 0.45–0.8 Å, paired with each r₀ | How far beyond r₀ the term reaches zero. |
| H···acceptor cutoff | `core.py` | 2.5 Å | Outer edge of the H···acceptor distribution; damps long, poorly aligned contacts. |
| Covalent D–H ceiling | `core.py` | 1.1 Å | Confirms the hydrogen belongs to one of the two heavy atoms. Skipped for virtual hydrogens. |
| Angular proxy threshold | `math_utils.py` | 0.6 Å | Applied to the triangle-inequality excess r(DH) + r(HA) − r(DA), which is 0 for a linear bond. Stands in for a minimum angle without computing one. |
| Switching exponents | `math_utils.py` | 6/12 | Sharpness of the rational switching function, in the form used for coordination-number collective variables. Not a Lennard-Jones potential. |
| Steric onset | `math_utils.py` | 2.40 Å | Below this, a 12th-power repulsive term suppresses unphysical contacts. |
| Cooperativity | `core.py` | 0.92 | Per-depth discount along a chain; see `--cooperativity`. |

The halogen parameters (F, Cl, Br) are the roughest of the set. Organic fluorine
in particular is a weak acceptor, so treating C–F as a full hydrogen-bond
acceptor will over-report bridges; if that is not wanted, remove `F`, `CL` and
`BR` from `valid_elements` in `core.py:_get_element`, which drops those edges
entirely.

Because these constants set the scale of the reported scores, results produced
with different values are not comparable, and Gephyra scores are not comparable
with counts produced by hard-cutoff tools.

## Citation

If you use **Gephyra** in your research, please cite it as follows:

> Peralta-Morales, M. F., Sciortino, G. & Maréchal, J.-D. Gephyra. *GitHub* https://github.com/insilichem/Gephyra (2026).

### BibTeX
```bibtex
@misc{PeraltaMorales2026Gephyra,
  author       = {Peralta-Morales, Mar{\'i}a Fernanda and Sciortino, Giuseppe and Mar{\'e}chal, Jean-Didier},
  title        = {Gephyra: Water bridge network analysis},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/insilichem/Gephyra](https://github.com/insilichem/Gephyra)}}
}
