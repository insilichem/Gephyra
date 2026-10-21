"""
Renders the manual's scene figures from a purely hypothetical system.

    python docs/manual/example/make_scene_figures.py

Builds a synthetic system - an idealised four-helix bundle of poly-alanine,
a made-up phenol-like ligand LIG, and a box of water molecules - writes a
short trajectory in which the waters move, runs the real Gephyra pipeline on
it (calculate -> cluster -> visualize) and renders the ChimeraX scenes to
docs/manual/images/08_scene_frame.png and 09_scene_clusters.png.

Developer tool only. Needs PyMOL's Python module (to build ideal helices),
MDAnalysis, and `chimerax` on the PATH. Nothing here depicts a real protein.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
IMAGES = os.path.join(REPO, "docs", "manual", "images")
sys.path.insert(0, os.path.join(REPO, "src"))

RNG = np.random.default_rng(7)
HELIX_LEN = 22
BOX = 46.0
N_FRAMES = 20
MIN_OO = 2.35


def build_helices(path):
    """Four ideal alpha-helices (poly-alanine) arranged as an antiparallel bundle."""
    import MDAnalysis as mda
    from pymol import cmd
    cmd.reinitialize()
    cmd.fab("A" * HELIX_LEN, "helix", ss=1)
    cmd.save(path + ".one.pdb", "helix")
    one = mda.Universe(path + ".one.pdb")
    pos = one.atoms.positions - one.atoms.center_of_geometry()
    # principal axis -> z
    _, _, vt = np.linalg.svd(pos - pos.mean(0))
    axis = vt[0]
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, z)
    c = float(np.dot(axis, z))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx / (1 + c)
    pos = pos @ R.T
    chains = []
    for k, (x, y) in enumerate([(4.9, 4.9), (-4.9, 4.9), (-4.9, -4.9), (4.9, -4.9)]):
        p = pos.copy()
        if k % 2:                         # antiparallel neighbours
            p[:, 1] *= -1
            p[:, 2] *= -1
        theta = np.deg2rad(90 * k)
        rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        chains.append(p @ rot.T + np.array([x, y, 0.0]))
    return one, chains


def build_system(workdir):
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    one, chains = build_helices(os.path.join(workdir, "helix"))
    n_prot = len(one.atoms) * 4

    # Hypothetical ligand LIG: a six-membered aromatic ring with a hydroxyl.
    ring = [(1.39 * np.cos(a), 1.39 * np.sin(a), 0.0) for a in np.deg2rad(np.arange(0, 360, 60))]
    hs = [(2.47 * np.cos(a), 2.47 * np.sin(a), 0.0) for a in np.deg2rad(np.arange(60, 360, 60))]
    lig = np.array(ring + [(2.75, 0.0, 0.0), (3.15, 0.85, 0.0)] + hs)
    lig_names = [f"C{i}" for i in range(1, 7)] + ["O1", "HO1"] + [f"H{i}" for i in range(2, 7)]
    lig_elems = ["C"] * 6 + ["O", "H"] + ["H"] * 5
    # Lay the ring flat on the bundle surface (ring normal = outward), with the
    # hydroxyl pointing up the bundle axis into the solvent.
    outward = np.array([1.0, 0.25, 0.0]) / np.linalg.norm([1.0, 0.25, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(outward, up)
    lig = lig @ np.vstack([up, outward, side])      # ring face towards the camera's side view
    lig += 6.0 * outward + np.array([0.0, 0.0, 2.0])
    # Slide it outward until it clears the helices (>= 3.2 A from any protein atom).
    protein = np.vstack(chains)
    while np.min(np.linalg.norm(protein[:, None, :] - lig[None, :, :], axis=2)) < 3.2:
        lig += 0.25 * outward

    solute = np.vstack(chains + [lig])

    # Water box on a jittered grid, avoiding the solute.
    # Liquid-like packing: a lattice at about the density of water, displaced
    # at random, with the rare pairs closer than MIN_OO thinned out.
    spacing = 2.85
    grid = np.arange(-BOX / 2 + 1.5, BOX / 2 - 1.5, spacing)
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    candidates = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1) + RNG.normal(0.0, 0.2, (gx.size, 3))
    from scipy.spatial import cKDTree
    order = RNG.permutation(len(candidates))
    pairs = cKDTree(candidates).query_pairs(MIN_OO, output_type="ndarray")
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    removed = np.zeros(len(candidates), bool)
    for a, b in sorted(pairs.tolist(), key=lambda ab: min(rank[ab[0]], rank[ab[1]])):
        if not removed[a] and not removed[b]:
            removed[b if rank[b] > rank[a] else a] = True
    centres = candidates[~removed]
    from MDAnalysis.lib.distances import capped_distance
    clash = capped_distance(centres, solute, max_cutoff=2.7, return_distances=False)
    keep = np.setdiff1d(np.arange(len(centres)), np.unique(clash[:, 0]))
    centres = centres[keep]
    # A first-shell water hydrogen-bonded to the ligand hydroxyl, so the chains
    # always have somewhere to start.
    first = lig[6] + 2.75 * (outward + np.array([0.0, 0.0, 0.25])) / np.linalg.norm(outward + [0.0, 0.0, 0.25])
    near = np.linalg.norm(centres - first, axis=1) < 2.6
    centres = np.vstack([centres[~near], first])

    def water(o, rot):
        h1 = np.array([0.757, 0.586, 0.0])
        h2 = np.array([-0.757, 0.586, 0.0])
        return [o, o + rot @ h1, o + rot @ h2]

    def random_rotation():
        q = RNG.normal(size=4)
        q /= np.linalg.norm(q)
        a, b, c, d = q
        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                         [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                         [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])

    # Point one O-H of every water at a neighbouring oxygen (or the ligand's
    # hydroxyl), as in a hydrogen-bonded network, then add a little disorder.
    acceptors = np.vstack([centres, lig[6:7]])
    tree = cKDTree(acceptors)

    def facing(o):
        _, idx = tree.query(o, k=4)
        target = acceptors[idx[RNG.integers(1, 4)]]
        a = target - o
        a /= np.linalg.norm(a)
        h1_dir = np.array([0.757, 0.586, 0.0]) / np.linalg.norm([0.757, 0.586, 0.0])
        v = np.cross(h1_dir, a)
        c = float(np.dot(h1_dir, a))
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c) if c > -0.999 else -np.eye(3)
        # random spin about the O-H axis keeps the second hydrogen disordered
        t = RNG.uniform(0, 2 * np.pi)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        spin = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
        return spin @ R

    rotations = [facing(o) for o in centres]
    waters = np.vstack([water(o, r) for o, r in zip(centres, rotations)])
    n_wat = len(centres)

    # Topology
    n_lig = len(lig)
    n_atoms = n_prot + n_lig + 3 * n_wat
    res_per_chain = len(one.residues)
    resindex = []
    for k in range(4):
        resindex += [k * res_per_chain + r for r in one.atoms.resindices]
    lig_res = 4 * res_per_chain
    resindex += [lig_res] * n_lig
    resindex += [lig_res + 1 + i // 3 for i in range(3 * n_wat)]
    n_res = lig_res + 1 + n_wat
    u = mda.Universe.empty(n_atoms, n_residues=n_res, n_segments=1, atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("name", list(one.atoms.names) * 4 + lig_names + ["O", "H1", "H2"] * n_wat)
    u.add_TopologyAttr("element", [e if e else n[0] for e, n in zip(one.atoms.elements, one.atoms.names)] * 4
                       + lig_elems + ["O", "H", "H"] * n_wat)
    u.add_TopologyAttr("type", list(u.atoms.elements))   # guess_bonds looks up vdW radii by type
    u.add_TopologyAttr("resname", list(one.residues.resnames) * 4 + ["LIG"] + ["WAT"] * n_wat)
    u.add_TopologyAttr("resid", list(range(1, res_per_chain + 1)) * 4 + [1] + list(range(1, n_wat + 1)))
    u.add_TopologyAttr("chainID", sum([[c] * len(one.atoms) for c in "ABCD"], []) + ["L"] * n_lig + ["W"] * (3 * n_wat))
    u.add_TopologyAttr("record_type", ["ATOM"] * n_prot + ["HETATM"] * (n_lig + 3 * n_wat))
    masses = {"C": 12.011, "N": 14.007, "O": 15.999, "H": 1.008, "S": 32.06}
    u.add_TopologyAttr("mass", [masses.get(e, 12.0) for e in u.atoms.elements])

    # Trajectory: the solute is still, waters translate and librate a little.
    frames = []
    for f in range(N_FRAMES):
        moved = []
        for o, r in zip(centres, rotations):
            dr = RNG.normal(0, 0.18, 3)
            moved += water(o + dr, facing(o + dr) if RNG.random() < 0.3 else r)
        frames.append(np.vstack([solute, np.array(moved)]))
    u.load_new(np.array(frames, dtype=np.float32), format=MemoryReader)
    for ts in u.trajectory:
        ts.dimensions = [BOX, BOX, BOX, 90, 90, 90]

    # Bonds (by distance) so Gephyra can use the full H-bond geometry.
    from MDAnalysis.topology.guessers import guess_bonds
    u.add_TopologyAttr("bonds", guess_bonds(u.atoms, u.atoms.positions, box=u.dimensions))

    topo = os.path.join(workdir, "hypothetical_system.pdb")
    traj = os.path.join(workdir, "hypothetical_system.dcd")
    u.trajectory[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mda.coordinates.PDB.PDBWriter(topo, bonds="all") as w:
            w.write(u.atoms)
        with mda.Writer(traj, n_atoms) as w:
            for ts in u.trajectory:
                w.write(u.atoms)
    print(f"system: {n_prot} protein atoms, ligand {n_lig}, {n_wat} waters, {N_FRAMES} frames")
    return topo, traj


def render(cxc, png, workdir, focus, width=1266, height=800):
    """Opens the scene script, turns to a side view of the bundle and saves an image."""
    wrapper = os.path.join(workdir, os.path.basename(png) + ".render.cxc")
    with open(wrapper, "w") as f:
        f.write(f"windowsize {width} {height}\nset bgColor white\nopen \"{cxc}\"\n"
                f"view orient\nturn x -90\nturn y -35\nview {focus}\n"
                f"save \"{png}\" width {width} height {height} supersample 3\n")
    subprocess.run(["chimerax", "--offscreen", "--exit", "--cmd", f'open "{wrapper}"'], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    if not os.path.exists(png):
        raise SystemExit(f"ChimeraX did not write {png}")
    print("wrote", os.path.relpath(png, REPO))


def main():
    import json
    from gephyra.analysis import cluster_pathways, run_analysis
    from gephyra.visualize import run_visualization

    workdir = tempfile.mkdtemp(prefix="gephyra-example-")
    try:
        topo, traj = build_system(workdir)
        results = os.path.join(workdir, "results.jsonl")
        clusters = os.path.join(workdir, "clustered_pathways.json")
        run_analysis(topo, traj, "resname LIG and name O1", water_sel="resname WAT", max_depth=6,
                     output_file=results)
        cluster_pathways(results, output_file=clusters)

        frames = [json.loads(l) for l in open(results) if l.startswith('{"type": "frame"')]
        best = max(frames, key=lambda o: len(o["paths"]))["frame_idx"]
        frame_cxc = run_visualization(results, format="chimerax", mode="frame", frame_idx=best,
                                      output_file=os.path.join(workdir, "scene_frame"))[0]
        cluster_cxc = run_visualization(clusters, format="chimerax", mode="cluster", top_clusters=10,
                                        output_file=os.path.join(workdir, "scene_clusters"))[0]
        os.makedirs(IMAGES, exist_ok=True)
        render(frame_cxc, os.path.join(IMAGES, "08_scene_frame.png"), workdir, focus="#101")
        render(cluster_cxc, os.path.join(IMAGES, "09_scene_clusters.png"), workdir, focus="#101 #102")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
