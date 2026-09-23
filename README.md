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
  `clustered_pathways.json` file with cluster size, occupancy, average probability,
  persistence statistics (`mean_persistence_frames` and `max_persistence_frames`
  accounting for stride), and a representative full-coordinate medoid geometry.

* **Understanding `PHquality` (Network Pathway Quality):** In the output files (e.g., `clustered_pathways.json`), the metric labeled `avg_prob_Hquality` (often referred to as `PHquality`) does not represent a standard fractional probability bounded between [0, 1]. Instead, it functions as a statistical sum of network microstates.
  
  * **How it is calculated:** For a given spatial endpoint, the algorithm evaluates all parallel microstates (unique atomic permutations) of water chains connecting the root to that solvent region. The quality score is the sum of the exponentially weighted edge probabilities across all these routing permutations: Z = sum(exp(-w)), where w is the accumulated logarithmic penalty of the path geometry (derived strictly from distance, angle, and steric switching functions).
    
  * **How to interpret this value:**
    * *Values can exceed 1.0:* Because this is an aggregated sum of multiple available structural routes, highly branched or degenerate channels with multiple parallel permutations will accumulate scores > 1.0.
    * *Physical meaning:* A higher `PHquality` indicates a high degree of topological degeneracy and geometric robustness. It measures how many favorable, pre-organized hydrogen-bond network permutations exist within that spatial channel.

  * **The same quantity appears under three names.** `avg_prob_Hquality` in `clustered_pathways.json`, the `probability` key of each path in the `.jsonl`, and the `Cumulative_Probability` column of the CSV all hold Z. Two of those names are misleading: Z is not bounded by 1 and is not a probability. They are kept as-is only so that existing output files and downstream scripts keep parsing. If you are writing new code against Gephyra output, read them as `hquality`.

- **Multi-phase execution:**
  - **Phase 1 — `calculate`:** Processes trajectory frames via MDAnalysis
    and NetworkX. Streams output as JSON Lines (`.jsonl`) and optional `.csv`
    to avoid RAM exhaustion on long trajectories.
  - **Phase 1.5 — `cluster`:** Post-processes `.jsonl` outputs separating
    geometric calculation and analytical clustering to preserve memory while extracting
    high-level collective behaviours.
  - **Phase 2 — `visualize`:** Parses `.jsonl` output and generates
    ready-to-run scripts for PyMOL (CGO cylinders), VMD (Tcl atom
    index selection), and UCSF Chimera (`.py` / `.bild`).

## Installation

Requires Python 3.8 or higher.

```bash
# Standard install from repository root
pip install .

# Development (editable) install
pip install -e .
```

## Usage

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
  --csv   summary.csv \
```

| Option | Default | Description |
|---|---|---|
| `--topo` | required | Topology file (.pdb, .tpr, …) |
| `--traj` | required | Trajectory file (.xtc, .dcd, …). Omit to evaluate topology only. |
| `--root` | required | MDAnalysis selection string for the root atom(s) (e.g. `"resname LIG"`). |
| `--water` | `"resname SOL or resname WAT or resname HOH"` | Solvent selection string. |
| `--stride` | `1` | Process every Nth frame. A warning is issued when `stride=1` and the trajectory exceeds 1000 frames. |
| `--max_depth` | `10` | Maximum number of sequential water molecules in a path. |
| `--min_depth` | `1` | Minimum number of sequential water molecules in a path; shorter paths are discarded from output. |
| `--cooperativity` | `0.92` | Geometric discount applied to successive hydrogen bonds along a chain: the *k*-th bond contributes its weight scaled by `cooperativity^(k-1)`. This models the polarisation that propagates along a water wire, which makes a bond embedded in a chain stronger than the same geometry in isolation. Values below 1.0 favour longer chains; `1.0` disables the effect and recovers the plain product of edge probabilities. |
| `--coarse_cutoff` | `4.5` | Distance cutoff in Å for initial neighbour graph construction. |
| `--output` | `results.jsonl` | Output file for full frame-by-frame path data (JSON Lines format). |
| `--csv` | — | Optional human-readable CSV summary of detected paths. |


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
| `--threshold` | `6.0` | Fréchet distance threshold in Å for fine clustering. |
| `--coarse_trigger` | `1000` | Number of paths below which the 9D coarse filter is bypassed for direct Fréchet calculation. |
| `--coarse_threshold` | `threshold / sqrt(3)` | 9D Feature Vector distance threshold in Å for the coarse screening pass. |
| `--max_paths` | `60000` | Hard cap on number of paths sent to the distance matrix. |
| `--output` | `clustered_pathways.json` | Output JSON file for the cluster summary. |

### 2. Visualize

```bash
# Clustered medoids (supports vmd, pymol, chimera)
gephyra visualize \
  --data clustered_pathways.json --format vmd --mode cluster \
  --output cluster_medoids.tcl

gephyra visualize \
  --data clustered_pathways.json --format pymol --mode cluster \
  --output cluster_medoids.py

gephyra visualize \
  --data clustered_pathways.json --format chimera --mode cluster \
  --output cluster_medoids_chimera.py

# Density overlay across all frames — PyMOL
gephyra visualize \
  --data results.jsonl --format pymol --mode density \
  --output network_density.py

# Single-frame selection — VMD
gephyra visualize \
  --data results.jsonl --format vmd --mode frame --frame 10 \
  --output frame_10.tcl

# Single-frame selection — UCSF Chimera
gephyra visualize \
  --data results.jsonl --format chimera --mode frame --frame 10 \
  --output frame_10.py
```

| Option | Default | Description |
|---|---|---|
| `--data` | required | Input file. `--mode cluster` expects the `clustered_pathways.json` written by `cluster`; `--mode density` and `--mode frame` expect the `.jsonl` written by `calculate`. Passing the wrong one for the mode will not produce a useful script. |
| `--format` | `vmd` | `vmd`, `pymol`, or `chimera`. |
| `--mode` | `density` | `density` (all frames overlaid), `frame` (single frame), or `cluster` (clustered medoids). |
| `--frame` | — | Frame index to visualize; required when `--mode frame`. |
| `--cluster_id` | — | Specific cluster ID to visualize by its sequential rank (only applies when `--mode cluster`). |
| `--max_bond_draw_dist` | `6.0` | Maximum Euclidean distance in Å between two connected path atoms before the segment is treated as a periodic-boundary artifact and skipped. Should sit slightly above your `--coarse_cutoff`; the default is safe for the default cutoff of 4.5 Å. |
| `--output` | `pathways_viz` | Output script filename or prefix. |

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

**Periodic boundary conditions in path coordinates.** The neighbour graph
is built with distance-based selection (which MDAnalysis applies with PBC
awareness), but the 3D coordinates stored for each path node are raw
wrapped positions. Paths that cross a periodic boundary will display as
broken or elongated segments in visualization software. This affects
trajectories where the root residue and solvent are in different periodic
images.

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
If persistence matters for your analysis, filter the `top_paths` output
manually or apply the `--cluster` option, which groups paths by spatial
similarity and reports per-cluster occupancy.

**Very large path counts.** The clustering step uses a hard maximum cap (`max_paths=60000`)
and requires a fully merged cluster to appear in at least two frames to prevent memory
bottlenecks and filter out raw thermal noise. For long trajectories with extreme branching,
the clustering step may truncate the least frequent paths. Use
`--stride` to reduce frame count or increase `--min_depth` to
reduce the number of short paths before enabling `--cluster`.


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
