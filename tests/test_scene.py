"""
Molecular scenes: structure snapshots and the viewer scripts that load them.

The synthetic system is a ligand oxygen at the origin and three TIP4P-like
waters (O, H1, H2 and a massless EPW extra point), one of which sits across
the periodic boundary from its neighbour on the path.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

import MDAnalysis as mda
import numpy as np

from gephyra.analysis import cluster_pathways, run_analysis
from gephyra.results import detect_data_kind, load_clusters, structure_info
from gephyra.scene import minimum_image_chain, write_snapshot
from gephyra.visualize import build_plan, resolve_structure, run_visualization

BOX = np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0])


def make_system(path):
    """Writes the synthetic system as a PDB and returns the Universe."""
    n_waters = 3
    names = ["O1"] + ["O", "H1", "H2", "EPW"] * n_waters
    resindex = [0] + [1 + i // 4 for i in range(4 * n_waters)]
    u = mda.Universe.empty(len(names), n_residues=1 + n_waters, atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", ["LIG"] + ["WAT"] * n_waters)
    u.add_TopologyAttr("resid", list(range(1, 2 + n_waters)))
    u.add_TopologyAttr("masses", [16.0] + [16.0, 1.008, 1.008, 0.0] * n_waters)
    u.add_TopologyAttr("elements", ["O"] + ["O", "H", "H", ""] * n_waters)
    pos = [[1.0, 15.0, 15.0]]
    # water 1 at x = 3.8; water 2 wrapped to x = 28.0 (i.e. -2.0, next to the ligand across the box)
    for ox in (3.8, 28.0, 6.6):
        pos += [[ox, 15.0, 15.0], [ox + 0.6, 15.8, 15.0], [ox + 0.6, 14.2, 15.0], [ox + 0.2, 15.0, 15.0]]
    u.atoms.positions = np.array(pos)
    u.dimensions = BOX
    u.atoms.write(path)
    return mda.Universe(path)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pdb = os.path.join(self.tmp, "sys.pdb")
        self.u = make_system(self.pdb)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def p(self, *parts):
        return os.path.join(self.tmp, *parts)

    def read(self, path):
        with open(path) as f:
            return f.read()


class TestMinimumImage(unittest.TestCase):
    def test_chain_follows_nearest_images(self):
        out = minimum_image_chain([[1, 0, 0], [29, 0, 0], [27, 0, 0]], BOX)
        np.testing.assert_allclose(out, [[1, 0, 0], [-1, 0, 0], [-3, 0, 0]], atol=1e-5)

    def test_no_box_leaves_positions(self):
        np.testing.assert_allclose(minimum_image_chain([[1, 0, 0], [29, 0, 0]], None), [[1, 0, 0], [29, 0, 0]])


class TestSnapshot(_Tmp):
    def test_waters_are_unwrapped_whole_and_virtual_sites_dropped(self):
        # path: ligand O (0) -> water 2 O (index 5) which is across the boundary
        scene = write_snapshot(self.u, 0, self.p("snap.pdb"), "resname LIG", "resname WAT",
                               paths=[[0, 5]], context_sel="resname LIG")
        snap = mda.Universe(scene.pdb_path)
        self.assertNotIn("EPW", list(snap.atoms.names))
        water = snap.select_atoms(f"chainID {scene.water_chain}")
        self.assertEqual(len(water), 3)                        # O, H1, H2
        self.assertAlmostEqual(water[0].position[0], -2.0, places=2)   # moved next to the ligand
        self.assertAlmostEqual(water[1].position[0], -1.4, places=2)   # hydrogens moved with it
        self.assertEqual(list(water.residues.resids), [1])
        self.assertEqual(scene.bridges, [(1, 2)])
        self.assertAlmostEqual(scene.bridge_coords[0][1][0], -2.0, places=2)
        self.assertEqual(snap.select_atoms(f"chainID {scene.root_chain}").names.tolist(), ["O1"])

    def test_root_and_frame_are_validated(self):
        with self.assertRaises(ValueError):
            write_snapshot(self.u, 0, self.p("s.pdb"), "resname NOPE", "resname WAT")
        with self.assertRaises(ValueError):
            write_snapshot(self.u, 7, self.p("s.pdb"), "resname LIG", "resname WAT")

    def test_default_context_excludes_solvent(self):
        scene = write_snapshot(self.u, 0, self.p("snap.pdb"), "resname LIG", "resname WAT")
        self.assertEqual(scene.n_atoms, 1)
        self.assertEqual(scene.water_serials, [])


class TestPipelineScenes(_Tmp):
    """calculate -> cluster -> visualize, all viewers, with the molecular scene."""

    def setUp(self):
        super().setUp()
        self.jsonl = self.p("r.jsonl")
        run_analysis(self.pdb, self.pdb, "resname LIG", water_sel="resname WAT", max_depth=3,
                     output_file=self.jsonl)
        # Two frames so a cluster survives the >= 2 frames rule.
        lines = open(self.jsonl).read().splitlines()
        meta = json.loads(lines[0])
        meta["n_frames_analyzed"] = 2
        frame = json.loads(lines[1])
        frame2 = dict(frame, frame_idx=1)
        with open(self.p("r2.jsonl"), "w") as f:
            f.write(json.dumps(meta) + "\n" + json.dumps(frame) + "\n" + json.dumps(frame2) + "\n")
        self.clusters = self.p("c.json")
        cluster_pathways(self.p("r2.jsonl"), output_file=self.clusters)

    def test_jsonl_paths_are_unwrapped(self):
        frame = [json.loads(l) for l in open(self.jsonl) if '"frame"' in l[:20]][0]
        for path in frame["paths"]:
            for a, b in zip(path["coords"][:-1], path["coords"][1:]):
                self.assertLess(np.linalg.norm(np.subtract(a, b)), 4.5)

    def test_cluster_file_carries_metadata_and_medoid_atoms(self):
        self.assertEqual(detect_data_kind(self.clusters), "cluster")
        clusters, meta = load_clusters(self.clusters)
        self.assertTrue(clusters)
        self.assertEqual(meta["topo_file"], os.path.abspath(self.pdb))
        self.assertIn("medoid_nodes", clusters[0])
        info = structure_info(self.clusters)
        self.assertEqual((info["root_sel"], info["water_sel"]), ("resname LIG", "resname WAT"))

    def test_legacy_list_cluster_file_is_still_read(self):
        with open(self.p("old.json"), "w") as f:
            json.dump([{"cluster_id": 1, "medoid_coords": [[0, 0, 0], [1, 0, 0]]}], f)
        self.assertEqual(detect_data_kind(self.p("old.json")), "cluster")
        self.assertEqual(load_clusters(self.p("old.json"))[1], {})
        self.assertIsNone(resolve_structure(self.p("old.json")))

    def test_frame_scene_scripts_for_every_viewer(self):
        expect = {
            "chimerax": ["close #101-103", "id 101", "show #101/Y,W", ".pb\" id 103"],
            "chimera": ["openModels.open(r'", "getPseudoBondGroup", "newPseudoBond"],
            "pymol": ["cmd.load(r'", "cmd.distance('gephyra_bridges'", "cmd.unbond("],
            "vmd": ["mol new {", "style dashed", "Licorice", "# Run it with: vmd -e vmd.tcl", "The -e is required"],
        }
        for fmt, needles in expect.items():
            with self.subTest(fmt=fmt):
                files = run_visualization(self.jsonl, format=fmt, mode="frame", frame_idx=0, output_file=self.p(fmt))
                self.assertTrue(files[-1].endswith("_structure.pdb"))
                script = self.read(files[0])
                for needle in needles:
                    self.assertIn(needle, script)
                self.assertNotIn("runCommand(\"open '", script)   # the Chimera quoting bug
                if fmt == "chimerax":
                    pb = self.read(next(f for f in files if f.endswith(".pb")))
                    self.assertIn("#101@@serial_number=", pb)

    def test_cluster_scene_draws_medoids_over_the_structure(self):
        files = run_visualization(self.clusters, format="chimerax", mode="cluster", output_file=self.p("cx"))
        cxc = self.read(files[0])
        self.assertIn("open", cxc)
        self.assertIn("id 102", cxc)          # geometry next to the structure
        self.assertIn("2dlabels text \"Cluster 1", cxc)

    def test_single_cluster_is_rebuilt_as_molecules(self):
        plan = build_plan(self.clusters, "cluster", cluster_id=1, structure=resolve_structure(self.clusters),
                          snapshot_path=self.p("one.pdb"))
        self.assertIsNotNone(plan.scene)
        self.assertEqual(plan.groups, [])
        self.assertTrue(plan.scene.bridges)

    def test_geometry_only_when_requested(self):
        files = run_visualization(self.jsonl, format="pymol", mode="frame", frame_idx=0, output_file=self.p("g"),
                                  with_structure=False)
        self.assertEqual(len(files), 1)
        self.assertNotIn("cmd.load(", self.read(files[0]))

    def test_missing_output_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            run_visualization(self.jsonl, format="vmd", mode="frame", output_file=self.p("nope", "x"))

    def test_moved_topology_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            run_visualization(self.jsonl, format="vmd", mode="frame", output_file=self.p("x"),
                              topo_file=self.p("moved.pdb"))


@unittest.skipUnless(os.environ.get("GEPHYRA_TEST_VIEWERS"), "set GEPHYRA_TEST_VIEWERS=1 to open scripts in installed viewers")
class TestViewersOpenScenes(TestPipelineScenes):
    """Runs the generated scene scripts in each installed viewer and checks for errors."""

    def _run(self, cmd, timeout=300):
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return res.stdout + res.stderr

    def test_viewers(self):
        ran = 0
        for fmt in ("chimerax", "chimera", "pymol", "vmd"):
            exe = {"chimerax": "chimerax", "chimera": "chimera", "pymol": "pymol", "vmd": "vmd"}[fmt]
            if not shutil.which(exe):
                continue
            for mode, data in (("frame", self.jsonl), ("cluster", self.clusters)):
                with self.subTest(fmt=fmt, mode=mode):
                    script = run_visualization(data, format=fmt, mode=mode, frame_idx=0,
                                               output_file=self.p(f"{fmt}_{mode}"))[0]
                    if fmt == "chimerax":
                        out = self._run(["chimerax", "--nogui", "--exit", "--cmd", f'open "{script}"'])
                    elif fmt == "chimera":
                        out = self._run(["xvfb-run", "-a", "chimera", "--nogui", "--nostatus", script])
                    elif fmt == "pymol":
                        out = self._run(["pymol", "-cq", script])
                    else:
                        wrapper = self.p(f"vmd_{mode}_run.tcl")
                        with open(wrapper, "w") as f:
                            f.write(f"source {{{script}}}\nquit\n")
                        out = self._run(["vmd", "-dispdev", "text", "-e", wrapper])
                    for bad in ("Traceback", "MidasError", "invalid graphics molecule", "Selector-Error",
                                "require exactly 1", "Unable to determine file type", "is undefined"):
                        self.assertNotIn(bad, out)
                    ran += 1
        if not ran:
            self.skipTest("no viewer installed")


if __name__ == "__main__":
    unittest.main()


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def make_tumbling_system(directory, n_frames=4):
    """
    A ligand oxygen with a three-water channel, in a system that is rotated
    and translated rigidly in every frame, like an unfitted trajectory. Two
    sets of waters take turns occupying the channel (even frames: set A,
    odd frames: set B, the other set waits far away), so only geometry - not
    atom identity - can tell that it is the same channel. Returns
    (topology, trajectory) paths.
    """
    from MDAnalysis.coordinates.memory import MemoryReader
    # A rigid carbon scaffold (resname SCF) plays the part of the protein C-alpha atoms.
    names = ["O1"] + ["O", "H1", "H2"] * 6 + ["CA"] * 4
    resindex = [0] + [1 + i // 3 for i in range(18)] + [7] * 4
    u = mda.Universe.empty(len(names), n_residues=8, atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", ["LIG"] + ["WAT"] * 6 + ["SCF"])
    u.add_TopologyAttr("resid", list(range(1, 9)))
    u.add_TopologyAttr("elements", ["O"] + ["O", "H", "H"] * 6 + ["C"] * 4)
    u.add_TopologyAttr("masses", [16.0] + [16.0, 1.008, 1.008] * 6 + [12.0] * 4)
    channel = []
    for x in (2.8, 5.6, 8.4):
        channel += [[x, 0.0, 0.0], [x + 0.6, 0.8, 0.0], [x + 0.6, -0.8, 0.0]]
    channel = np.array(channel)
    parked = channel + np.array([0.0, 20.0, 0.0])
    frames = []
    for f in range(n_frames):
        set_a, set_b = (channel, parked) if f % 2 == 0 else (parked, channel)
        scaffold = np.array([[-6.0, 0.0, 0.0], [-6.0, 4.0, 0.0], [-6.0, 0.0, 4.0], [-9.0, 1.0, 1.0]])
        system = np.vstack([[[0.0, 0.0, 0.0]], set_a, set_b, scaffold])
        R = _rotation([1, 2, 3], 0.7 * f)
        frames.append(system @ R.T + np.array([50.0, 50.0, 50.0]) + np.array([6.0, -4.0, 3.0]) * f)
    u.load_new(np.array(frames, dtype=np.float32), format=MemoryReader)
    for ts in u.trajectory:
        ts.dimensions = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0]
    topo, traj = os.path.join(directory, "tumble.pdb"), os.path.join(directory, "tumble.dcd")
    u.trajectory[0]
    u.atoms.write(topo)
    with mda.Writer(traj, len(u.atoms)) as w:
        for _ in u.trajectory:
            w.write(u.atoms)
    return topo, traj


class TestAlignment(unittest.TestCase):
    """Unfitted trajectories: the same channel must be compared in a common frame."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.topo, self.traj = make_tumbling_system(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def p(self, *parts):
        return os.path.join(self.tmp, *parts)

    def _paths(self, align):
        out = self.p(f"r_{align}.jsonl")
        run_analysis(self.topo, self.traj, "resname LIG", water_sel="resname WAT", max_depth=3,
                     output_file=out, align_sel=align)
        lines = [json.loads(l) for l in open(out)]
        return out, lines[0], [l for l in lines if l["type"] == "frame"]

    def test_resolve_alignment(self):
        from gephyra.scene import resolve_alignment
        u = mda.Universe(self.topo)
        self.assertIsNone(resolve_alignment(u, "auto"))          # no protein, no nucleic acid
        self.assertIsNone(resolve_alignment(u, None))
        self.assertIsNone(resolve_alignment(u, "none"))
        self.assertEqual(resolve_alignment(u, "all"), "all")
        with self.assertRaises(ValueError):
            resolve_alignment(u, "resname LIG")                   # a single atom cannot define a frame

    def test_superposed_paths_coincide_across_frames(self):
        out, meta, frames = self._paths("resname SCF")
        self.assertEqual(meta["alignment"], {"selection": "resname SCF", "reference_frame": 0})
        # Different water molecules, same place once superposed.
        longest = [max(f["paths"], key=lambda p: p["length"]) for f in frames]
        self.assertNotEqual(longest[0]["nodes"], longest[1]["nodes"])
        reference = np.array(longest[0]["coords"])
        for path in longest[1:]:
            np.testing.assert_allclose(path["coords"], reference, atol=0.05)

    def test_without_superposition_paths_move_and_do_not_cluster(self):
        out, meta, frames = self._paths(None)
        self.assertIsNone(meta["alignment"])
        first = [f["paths"][0]["coords"][0] for f in frames]
        self.assertGreater(np.linalg.norm(np.subtract(first[-1], first[0])), 10.0)
        cluster_pathways(out, threshold=3.0, output_file=self.p("c_raw.json"))
        occupancies = [c["occupancy"] for c in load_clusters(self.p("c_raw.json"))[0]]
        self.assertLess(max(occupancies, default=0.0), 1.0)   # the channel is never seen in all frames

    def test_superposition_lets_the_channel_cluster(self):
        out, _, _ = self._paths("resname SCF")
        cluster_pathways(out, threshold=3.0, output_file=self.p("c.json"))
        clusters, meta = load_clusters(self.p("c.json"))
        self.assertTrue(clusters)
        self.assertAlmostEqual(clusters[0]["occupancy"], 1.0)
        self.assertEqual(meta["alignment"]["selection"], "resname SCF")

    def test_snapshot_is_superposed_like_the_paths(self):
        out, meta, frames = self._paths("resname SCF")
        structure = resolve_structure(out)
        self.assertEqual(structure["alignment"], meta["alignment"])
        scene = write_snapshot(mda.Universe(self.topo, self.traj), 3, self.p("snap.pdb"), "resname LIG", "resname WAT",
                               paths=[frames[3]["paths"][-1]["nodes"]], context_sel="resname LIG",
                               alignment=meta["alignment"])
        root = mda.Universe(scene.pdb_path).select_atoms(f"chainID {scene.root_chain}").positions[0]
        np.testing.assert_allclose(root, frames[0]["paths"][0]["coords"][0], atol=0.05)


def make_cofactor_system(path):
    """
    A hypothetical protein residue (ALA) with a cofactor (COF) and a metal (MTL)
    covalently bonded to it - so all three form one molecule, as with a
    parameterised metal centre - plus a separate ligand LIG and two waters.
    """
    names = ["N", "CA", "C", "O", "C1", "C2", "O3", "FE", "O1", "O", "H1", "H2", "O", "H1", "H2"]
    resnames = ["ALA", "COF", "MTL", "LIG", "WAT", "WAT"]
    resindex = [0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5]
    u = mda.Universe.empty(len(names), n_residues=6, atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", [1, 2, 3, 4, 5, 6])
    u.add_TopologyAttr("elements", ["N", "C", "C", "O", "C", "C", "O", "Fe", "O", "O", "H", "H", "O", "H", "H"])
    u.add_TopologyAttr("masses", [14.0, 12.0, 12.0, 16.0, 12.0, 12.0, 16.0, 55.8, 16.0, 16.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    u.atoms.positions = np.array([
        [10.0, 10, 10], [11.4, 10, 10], [12.0, 11.3, 10], [11.4, 12.3, 10],     # ALA
        [12.0, 8.7, 10], [13.5, 8.7, 10], [14.2, 7.6, 10],                     # COF bonded to CA
        [12.0, 7.0, 11.5],                                                      # MTL bonded to COF
        [16.0, 7.6, 10],                                                        # LIG O1 (root)
        [18.8, 7.6, 10], [19.4, 8.4, 10], [19.4, 6.8, 10],                      # water 1
        [21.6, 7.6, 10], [22.2, 8.4, 10], [22.2, 6.8, 10]])                     # water 2
    u.dimensions = np.array([60.0, 60.0, 60.0, 90.0, 90.0, 90.0])
    u.add_TopologyAttr("bonds", [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (4, 7), (9, 10), (9, 11),
                                 (12, 13), (12, 14)])
    u.atoms.write(path, bonds="all")
    return mda.Universe(path)


class TestCofactorsAreShown(unittest.TestCase):
    """Cofactors bonded to the protein must still be drawn (not only as cartoon)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pdb = os.path.join(self.tmp, "cof.pdb")
        self.u = make_cofactor_system(self.pdb)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def p(self, *parts):
        return os.path.join(self.tmp, *parts)

    def _scene(self):
        return write_snapshot(self.u, 0, self.p("snap.pdb"), "resname LIG", "resname WAT", paths=[[8, 9, 12]])

    def test_bonded_cofactor_is_classified_as_hetero(self):
        scene = self._scene()
        self.assertEqual(scene.hetero_resnames, ["COF", "MTL"])
        self.assertEqual(scene.metal_resnames, ["MTL"])
        records = {line[17:20]: line[:6].strip() for line in open(scene.pdb_path) if line.startswith(("ATOM", "HETATM"))}
        self.assertEqual(records["ALA"], "ATOM")
        for resname in ("COF", "MTL", "LIG", "WAT"):
            self.assertEqual(records[resname], "HETATM", resname)
        # Bonded to the protein, so it keeps the protein's chain: chains alone cannot find it.
        chains = {line[17:20]: line[21] for line in open(scene.pdb_path) if line.startswith(("ATOM", "HETATM"))}
        self.assertEqual(chains["COF"], chains["ALA"])

    def test_every_viewer_script_shows_the_cofactor(self):
        from gephyra.visualize import Plan, write_chimera, write_chimerax, write_pymol, write_vmd
        plan = Plan("frame", "frame 0", "test", scene=self._scene(), frame=0)
        expected = {
            write_chimerax: ["show #101:COF,MTL target a", "style #101:MTL ball", "view #101/W | #101/Y | #101:COF,MTL"],
            write_pymol: ["cmd.show('sticks', 'gephyra_structure and resn COF+MTL')",
                          "cmd.show('spheres', 'gephyra_structure and resn MTL')"],
            write_chimera: ["('COF', 'MTL')", "represent ball %s:MTL"],
            write_vmd: ["mol selection {resname COF MTL}", "mol representation VDW", "or resname COF MTL"],
        }
        for writer, needles in expected.items():
            with self.subTest(viewer=writer.__name__):
                ext = ".cxc" if writer is write_chimerax else ".tcl" if writer is write_vmd else ".py"
                script = writer(plan, self.p(writer.__name__ + ext))[0]
                text = open(script).read()
                for needle in needles:
                    self.assertIn(needle, text)

    @unittest.skipUnless(os.environ.get("GEPHYRA_TEST_VIEWERS") and shutil.which("chimerax"),
                         "set GEPHYRA_TEST_VIEWERS=1 with chimerax on PATH")
    def test_chimerax_displays_the_cofactor_atoms(self):
        from gephyra.visualize import Plan, write_chimerax
        plan = Plan("frame", "frame 0", "test", scene=self._scene(), frame=0)
        script = write_chimerax(plan, self.p("scene.cxc"))[0]
        check = self.p("check.cxc")
        with open(check, "w") as f:
            f.write(f'open "{script}"\ninfo atoms #101:COF,MTL attribute display\n')
        out = subprocess.run(["chimerax", "--nogui", "--exit", "--cmd", f'open "{check}"'],
                             capture_output=True, text=True, timeout=300).stdout
        shown = [line for line in out.splitlines() if line.startswith("atom id") and "display" in line]
        self.assertTrue(shown)
        self.assertTrue(all(line.endswith("True") for line in shown), shown)
