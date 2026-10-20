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
            "vmd": ["mol new {", "style dashed", "Licorice"],
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
