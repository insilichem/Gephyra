"""
Molecular context for visualization scripts.

The viewer scripts used to draw only free-floating geometry for the water
paths and relied on the user to load a matching structure, at the right
model slot, before running them. This module instead re-opens the analysed
topology/trajectory, takes one frame, and writes a small PDB "snapshot"
containing the solute (protein, ligands, cofactors) and, when a specific frame
or cluster medoid is drawn, the bridging water molecules themselves. Every
viewer script then loads that snapshot and styles a complete scene.

Periodic boundaries: bridging waters are shifted, as whole molecules, to the
periodic image closest to the previous atom of their path (the chain is
anchored at the root atom), so a bridge never appears torn across the box.
The solute is written with its coordinates as stored in the trajectory, so
trajectories should be imaged/centred around the solute (for example
`cpptraj autoimage` or `gmx trjconv -pbc mol -center`), as is usual for
analysis.
"""

import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Counter-ions are excluded from the default context: they are numerous,
# scattered through the box, and rarely part of a binding-site picture.
COUNTER_IONS = "Na+ Cl- K+ NA CL K SOD CLA POT MG2 CAL"


def default_context_selection(water_sel):
    """Everything that is not solvent or a common counter-ion."""
    return f"not ({water_sel}) and not resname {COUNTER_IONS}"


@dataclass
class Scene:
    """What a viewer script needs to know about the snapshot it loads."""

    pdb_path: str                 # absolute path of the snapshot PDB
    frame: int                    # trajectory frame the snapshot comes from
    n_atoms: int
    root_chain: str = "Y"         # chain ID given to the root residue(s) in the snapshot
    water_chain: str = "W"        # chain ID given to the bridging waters
    root_serials: list = field(default_factory=list)    # every atom of the root residue(s)
    water_serials: list = field(default_factory=list)   # every atom of the bridging waters
    bridges: list = field(default_factory=list)         # (serial_a, serial_b) heavy-atom pairs along paths
    bridge_coords: list = field(default_factory=list)   # ((x, y, z), (x, y, z)) for each bridge


def _drop_virtual_sites(ag):
    if len(ag) and hasattr(ag, "masses"):
        return ag[ag.masses > 0.1]
    return ag


def _solute_chains(solute, u):
    """
    Chain ID per solute atom. Chain IDs from the topology are kept when it has
    more than one; otherwise (e.g. AMBER .prmtop) each bonded protein molecule
    gets its own letter, so viewers do not join two monomers with a fake
    "missing segment", and other solute molecules share chain 'H'.
    """
    existing = None
    if hasattr(solute, "chainIDs"):
        existing = np.array([str(c).strip() for c in solute.chainIDs], dtype=object)
        if len(set(existing) - {""}) > 1:
            return np.where(existing == "", "A", existing).astype(object)
    chains = np.array(["A"] * len(solute), dtype=object)
    try:
        frag = solute.fragindices
    except Exception:  # no bond information
        return chains
    protein = set(u.select_atoms("protein").indices)
    letters = iter("ABCDEFGIJKMNOPQRS")
    for f in np.unique(frag):
        members = frag == f
        if any(int(i) in protein for i in solute.indices[members]):
            chains[members] = next(letters, "A")
        else:
            chains[members] = "H"
    return chains


def _free_chain(preferred, used):
    """First chain ID from `preferred` (then A-Z, 0-9) not in `used`."""
    for c in list(preferred) + [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]:
        if c not in used:
            return c
    return preferred[0]


def minimum_image_chain(positions, box):
    """
    Positions of consecutive path atoms, each moved to the periodic image
    nearest to the previous one. The first position is kept as is.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if box is None or len(positions) < 2 or not np.all(np.asarray(box[:3]) > 0):
        return positions
    from MDAnalysis.lib.distances import minimize_vectors
    out = positions.copy()
    steps = minimize_vectors((positions[1:] - positions[:-1]).astype(np.float32), np.asarray(box, dtype=np.float32))
    for i, step in enumerate(steps, start=1):
        out[i] = out[i - 1] + step
    return out


# Tried in order when the alignment selection is "auto".
AUTO_ALIGN_SELECTIONS = ("protein and name CA", "nucleic and name P")
MAX_FIT_RMSD = 8.0   # above this the superposition is suspicious (e.g. a molecule split by PBC)


def resolve_alignment(u, align_sel="auto"):
    """
    The selection used to superpose frames, or None for no superposition.

    "auto" picks the first of AUTO_ALIGN_SELECTIONS with at least three atoms
    (protein C-alpha, then nucleic-acid phosphorus); None, "" or "none"
    disables it; any other string is used as an MDAnalysis selection.
    """
    if align_sel is None or str(align_sel).strip().lower() in ("", "none", "off"):
        return None
    sel = str(align_sel).strip()
    candidates = AUTO_ALIGN_SELECTIONS if sel.lower() == "auto" else (sel,)
    for candidate in candidates:
        if len(u.select_atoms(candidate)) >= 3:
            return candidate
    if sel.lower() == "auto":
        return None
    raise ValueError(f"Alignment selection '{sel}' matches fewer than three atoms.")


class FrameAligner:
    """
    Superposes frames onto a reference frame using a selection of atoms.

    Molecular dynamics trajectories that are not fitted let the solute drift
    and tumble through the box, so the same water channel appears at very
    different absolute coordinates in different frames. Path coordinates are
    therefore expressed in the frame of reference of `reference_frame`,
    superposing `selection` (by default the protein C-alpha atoms).
    """

    def __init__(self, u, selection, reference_frame):
        from MDAnalysis.analysis import align  # noqa: F401  (imported here to keep start-up light)
        self.u = u
        self.selection = selection
        self.reference_frame = int(reference_frame)
        self.atoms = u.select_atoms(selection)
        current = u.trajectory.ts.frame
        u.trajectory[self.reference_frame]
        self.ref = self.atoms.positions.astype(np.float64)
        self.ref_centre = self.ref.mean(axis=0)
        u.trajectory[current]
        self._warned = False

    def current_transform(self):
        """(R, mobile_centre, rmsd) superposing the current frame onto the reference."""
        from MDAnalysis.analysis.align import rotation_matrix
        mobile = self.atoms.positions.astype(np.float64)
        centre = mobile.mean(axis=0)
        R, rmsd = rotation_matrix(mobile - centre, self.ref - self.ref_centre)
        if rmsd > MAX_FIT_RMSD and not self._warned:
            self._warned = True
            logger.warning(f"Superposition on '{self.selection}' has RMSD {rmsd:.1f} A in frame "
                           f"{self.u.trajectory.ts.frame}. If the molecule is split across the periodic "
                           "boundary, image the trajectory first (e.g. cpptraj autoimage).")
        return np.asarray(R), centre, rmsd

    def apply(self, coords, transform):
        R, centre, _ = transform
        return (np.asarray(coords, dtype=np.float64) - centre) @ R.T + self.ref_centre

    def describe(self):
        return {"selection": self.selection, "reference_frame": self.reference_frame}


def load_universe(topo_file, traj_file=None):
    import MDAnalysis as mda
    for label, path in (("Topology", topo_file), ("Trajectory", traj_file)):
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")
    if not topo_file:
        raise ValueError("A topology file is needed to draw the molecular context.")
    return mda.Universe(topo_file, traj_file) if traj_file else mda.Universe(topo_file)


def write_snapshot(u, frame, out_pdb, root_sel, water_sel, paths=(), context_sel=None, alignment=None):
    """
    Writes the snapshot PDB for `frame` and returns a Scene.

    Args:
        u: MDAnalysis Universe of the analysed system.
        frame: trajectory frame index (as stored in the .jsonl `frame_idx`).
        out_pdb: output path.
        root_sel, water_sel: the selections used by `calculate`.
        paths: iterables of atom indices (the `nodes` of each path); their
            water residues are added to the snapshot and their consecutive
            pairs become `bridges`.
        context_sel: solute selection; defaults to default_context_selection.
        alignment: the {"selection", "reference_frame"} recorded by `calculate`;
            the snapshot is then superposed in the same way as the stored
            path coordinates, so cluster medoids and density overlays line up.
    """
    import MDAnalysis as mda

    if frame < 0 or frame >= len(u.trajectory):
        raise ValueError(f"Frame {frame} is outside the trajectory (0-{len(u.trajectory) - 1}).")
    aligner = FrameAligner(u, alignment["selection"], alignment["reference_frame"]) if alignment else None
    u.trajectory[frame]
    box = u.dimensions

    context = u.select_atoms(context_sel or default_context_selection(water_sel))
    root = u.select_atoms(root_sel)
    if len(root) == 0:
        raise ValueError(f"Root selection '{root_sel}' matches no atoms in this topology; "
                         "is it the topology that was analysed?")
    root_res = root.residues.atoms

    # Water residues on the paths and the shift that brings each one next to
    # its predecessor on the chain.
    shifts = {}      # residue index -> translation vector
    path_list = [list(map(int, p)) for p in paths]
    for nodes in path_list:
        if max(nodes) >= len(u.atoms):
            raise ValueError("Path atom indices exceed the topology size; "
                             "is it the topology that was analysed?")
        raw = u.atoms[nodes].positions
        chained = minimum_image_chain(raw, box)
        for idx, raw_pos, new_pos in zip(nodes[1:], raw[1:], chained[1:]):
            resindex = u.atoms[idx].resindex
            shifts.setdefault(resindex, new_pos - raw_pos)

    water_res_atoms = u.residues[sorted(shifts)].atoms if shifts else u.atoms[[]]
    # Massless virtual sites (TIP4P/TIP5P extra points, lone pairs) are not
    # real atoms and only clutter the picture.
    water_res_atoms = _drop_virtual_sites(water_res_atoms)
    # Order: solute first (keeps protein serials stable across snapshots), then waters.
    solute = _drop_virtual_sites((context | root_res) - water_res_atoms)
    group = solute + water_res_atoms
    if len(group) == 0:
        raise ValueError("The snapshot would be empty: the context selection matched nothing.")

    snap = mda.Merge(group)
    positions = group.positions.astype(np.float64)
    offset = len(solute)
    for i, atom in enumerate(water_res_atoms, start=offset):
        positions[i] += shifts[atom.resindex]
    if aligner is not None:
        positions = aligner.apply(positions, aligner.current_transform())
    snap.atoms.positions = positions
    if box is not None and aligner is None:
        snap.dimensions = box   # a rotated box would be misleading, so it is left out when superposed

    # Chain IDs let every viewer address the parts with a simple spec: the
    # root residue and the bridging waters get chains of their own (letters not
    # already used by the solute), other solute atoms keep theirs or get 'A'.
    # Network waters are also renumbered so that PDB's 4-digit resSeq never
    # makes two waters look like one residue.
    chains = _solute_chains(solute, u)
    chains = np.concatenate([chains, np.array(["W"] * len(water_res_atoms), dtype=object)])
    root_pos = np.isin(group.indices, root_res.indices)
    used_by_others = set(chains[:offset][~root_pos[:offset]])
    root_chain = _free_chain("YLXZ", used_by_others)
    water_chain = _free_chain("WVUT", used_by_others | {root_chain})
    chains[root_pos] = root_chain
    chains[offset:] = water_chain
    if not hasattr(snap.atoms, "chainIDs"):
        snap.add_TopologyAttr("chainIDs")
    snap.atoms.chainIDs = chains.astype(str)
    if len(water_res_atoms):
        n_sol_res = len(solute.residues)
        snap.residues[n_sol_res:].resids = np.arange(1, len(water_res_atoms.residues) + 1)

    os.makedirs(os.path.dirname(os.path.abspath(out_pdb)), exist_ok=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # missing optional PDB fields (occupancy, altLocs, ...)
        snap.atoms.write(out_pdb)

    # Serial numbers are 1-based positions in the written group.
    serial_of = {int(idx): i + 1 for i, idx in enumerate(group.indices)}
    snap_pos = {int(idx): positions[i] for i, idx in enumerate(group.indices)}

    bridges, bridge_coords, seen = [], [], set()
    for nodes in path_list:
        for a, b in zip(nodes[:-1], nodes[1:]):
            if a not in serial_of or b not in serial_of:
                continue
            key = (min(serial_of[a], serial_of[b]), max(serial_of[a], serial_of[b]))
            if key in seen:
                continue
            seen.add(key)
            bridges.append(key)
            bridge_coords.append((tuple(snap_pos[a]), tuple(snap_pos[b])))

    scene = Scene(
        pdb_path=os.path.abspath(out_pdb),
        frame=int(frame),
        n_atoms=len(group),
        root_chain=root_chain,
        water_chain=water_chain,
        root_serials=sorted(serial_of[int(i)] for i in root_res.indices if int(i) in serial_of),
        water_serials=list(range(offset + 1, len(group) + 1)),
        bridges=bridges,
        bridge_coords=bridge_coords,
    )
    logger.info(f"Structure snapshot of frame {frame} written to {scene.pdb_path} "
                f"({len(solute)} solute atoms, {len(water_res_atoms.residues)} bridging waters).")
    return scene
