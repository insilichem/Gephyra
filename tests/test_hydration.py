"""
Hydration-site analysis on a hypothetical, rigidly tumbling system in which
different water molecules take turns at the same spot - so only positions,
superposed on a rigid scaffold, can reveal the sites.
"""

import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

import MDAnalysis as mda
import numpy as np

from gephyra import forms
from gephyra.cli import build_parser, main
from gephyra.hydration import assign_shells, bulk_occupancy, run_hydration_sites, water_oxygens
from gephyra.results import detect_data_kind, load_sites, structure_info
from gephyra.visualize import run_visualization

N_FRAMES = 6
SITE_A = np.array([2.8, 0.0, 0.0])          # always occupied, by a different water each frame
SITE_B = np.array([0.0, 2.8, 0.0])          # occupied in even frames only
SITE_C = np.array([2.8, 2.8, 0.0])          # second shell: next to site A, every frame


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def make_system(directory):
    from MDAnalysis.coordinates.memory import MemoryReader
    n_waters = 6
    names = ["O1"] + ["O", "H1", "H2"] * n_waters + ["CA"] * 4
    resindex = [0] + [1 + i // 3 for i in range(3 * n_waters)] + [1 + n_waters] * 4
    u = mda.Universe.empty(len(names), n_residues=n_waters + 2, atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", ["LIG"] + ["WAT"] * n_waters + ["SCF"])
    u.add_TopologyAttr("resid", list(range(1, n_waters + 3)))
    u.add_TopologyAttr("elements", ["O"] + ["O", "H", "H"] * n_waters + ["C"] * 4)
    u.add_TopologyAttr("masses", [16.0] + [16.0, 1.008, 1.008] * n_waters + [12.0] * 4)
    water = lambda o: [o, o + [0.6, 0.8, 0.0], o + [0.6, -0.8, 0.0]]
    frames = []
    for f in range(N_FRAMES):
        o = [np.array([30.0 + 4 * k, 30.0, 30.0]) for k in range(n_waters)]   # parked far away
        o[f % 3] = SITE_A.copy()                                            # waters 0-2 take turns at A
        if f % 2 == 0:
            o[3] = SITE_B.copy()
        o[4] = SITE_C.copy()
        system = np.vstack([[[0.0, 0.0, 0.0]], np.vstack([water(x) for x in o]),
                            [[-6.0, 0, 0], [-6.0, 4.0, 0], [-6.0, 0, 4.0], [-9.0, 1.0, 1.0]]])
        R = _rotation([1, 2, 3], 0.6 * f)
        frames.append(system @ R.T + np.array([60.0, 60.0, 60.0]) + np.array([5.0, -3.0, 2.0]) * f)
    u.load_new(np.array(frames, dtype=np.float32), format=MemoryReader)
    for ts in u.trajectory:
        ts.dimensions = [150.0, 150.0, 150.0, 90.0, 90.0, 90.0]
    topo, traj = os.path.join(directory, "sys.pdb"), os.path.join(directory, "sys.dcd")
    u.trajectory[0]
    u.atoms.write(topo)
    with mda.Writer(traj, len(u.atoms)) as w:
        for _ in u.trajectory:
            w.write(u.atoms)
    return topo, traj


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.topo, self.traj = make_system(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def p(self, *parts):
        return os.path.join(self.tmp, *parts)

    def run_sites(self, **kwargs):
        options = dict(water_sel="resname WAT", align_sel="resname SCF", max_shell=2, min_occupancy=0.2,
                       output_file=self.p("sites.json"), csv_file=self.p("sites.csv"),
                       dx_file=self.p("map.dx"), pdb_file=self.p("sites.pdb"))
        options.update(kwargs)
        return run_hydration_sites(self.topo, self.traj, "resname LIG", **options)


class TestShells(unittest.TestCase):
    def test_shells_follow_hydrogen_bond_contacts(self):
        oxygens = np.array([[2.8, 0, 0], [5.6, 0, 0], [8.4, 0, 0], [20.0, 0, 0]], dtype=np.float32)
        root = np.zeros((1, 3), dtype=np.float32)
        np.testing.assert_array_equal(assign_shells(root, oxygens, None, 1, 3.5), [1, 0, 0, 0])
        np.testing.assert_array_equal(assign_shells(root, oxygens, None, 2, 3.5), [1, 2, 0, 0])
        np.testing.assert_array_equal(assign_shells(root, oxygens, None, 3, 3.5), [1, 2, 3, 0])

    def test_bulk_reference(self):
        self.assertAlmostEqual(float(bulk_occupancy(1.0)), 0.13, places=2)

    def test_one_oxygen_per_water_ignoring_virtual_sites(self):
        u = mda.Universe.empty(4, n_residues=1, atom_resindex=[0, 0, 0, 0], trajectory=True)
        u.add_TopologyAttr("name", ["O", "H1", "H2", "EPW"])
        u.add_TopologyAttr("resname", ["WAT"])
        u.add_TopologyAttr("masses", [16.0, 1.008, 1.008, 0.0])
        self.assertEqual(list(water_oxygens(u.atoms).names), ["O"])


class TestHydrationSites(_Tmp):
    def test_sites_occupancy_residence_shell_and_contacts(self):
        result = self.run_sites()
        sites, meta = load_sites(self.p("sites.json"))
        self.assertEqual(result["n_sites"], 3)
        by_occ = sorted(sites, key=lambda s: -s["occupancy"])
        occupancies = sorted(s["occupancy"] for s in sites)
        self.assertEqual(occupancies, [0.5, 1.0, 1.0])
        site_b = next(s for s in sites if s["occupancy"] == 0.5)
        self.assertEqual(site_b["max_residence_frames"], 1)             # every other frame
        self.assertEqual(site_b["shell"], 1)
        full = [s for s in sites if s["occupancy"] == 1.0]
        self.assertEqual(sorted(s["shell"] for s in full), [1, 2])      # site A (shell 1), site C (shell 2)
        for s in full:
            self.assertEqual(s["max_residence_frames"], N_FRAMES)
            self.assertGreater(s["enrichment"], 7)
        shell1 = next(s for s in full if s["shell"] == 1)
        self.assertEqual(shell1["contacts"][0]["atom"], "LIG1:O1")
        self.assertAlmostEqual(shell1["contacts"][0]["mean_distance"], 2.8, places=1)
        self.assertEqual(by_occ[0]["occupancy"], result["top_occupancy"])
        self.assertEqual(meta["alignment"], {"selection": "resname SCF", "reference_frame": 0})
        self.assertGreater(meta["map_level"], 1.0)

    def test_first_shell_only(self):
        self.run_sites(max_shell=1)
        sites, _ = load_sites(self.p("sites.json"))
        self.assertEqual(sorted(s["occupancy"] for s in sites), [0.5, 1.0])
        self.assertTrue(all(s["shell"] == 1 for s in sites))

    def test_without_superposition_the_tumbling_sites_are_lost(self):
        result = self.run_sites(align_sel=None)
        self.assertEqual(result["n_sites"], 0)
        self.assertIn("No spot is occupied in at least 20%", result["warning"])
        self.assertIn("the most occupied one reaches", result["warning"])

    def test_output_files(self):
        self.run_sites()
        rows = open(self.p("sites.csv")).read().splitlines()
        self.assertTrue(rows[0].startswith("Site,X,Y,Z,Occupancy,Enrichment_vs_bulk"))
        self.assertEqual(len(rows), 4)
        pdb = [l for l in open(self.p("sites.pdb")) if l.startswith("HETATM")]
        self.assertEqual(len(pdb), 3)
        self.assertEqual(float(pdb[0][54:60]), 1.0)                     # occupancy column
        from gridData import Grid
        g = Grid(self.p("map.dx"))
        self.assertGreater(g.grid.max(), 1.0)

    def test_validation(self):
        for bad in ({"max_shell": 4}, {"min_occupancy": 1.5}, {"site_radius": 0}, {"stride": 0}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                self.run_sites(**bad)
        with self.assertRaises(ValueError):
            run_hydration_sites(self.topo, self.traj, "resname NOPE", output_file=self.p("x.json"))

    def test_file_kind_and_structure_info(self):
        self.run_sites()
        self.assertEqual(detect_data_kind(self.p("sites.json")), "sites")
        info = structure_info(self.p("sites.json"))
        self.assertEqual((info["root_sel"], info["water_sel"]), ("resname LIG", "resname WAT"))
        self.assertEqual(info["alignment"]["selection"], "resname SCF")


class TestSitesVisualization(_Tmp):
    def test_every_viewer_draws_sites_and_map(self):
        self.run_sites()
        expect = {"chimerax": ["volume #104 style surface", "2dlabels text \"Site 1", "close #101-104"],
                  "chimera": ["vol = openModels.open(", "style mesh"],
                  "pymol": ["cmd.isomesh('gephyra_mesh'", "SPHERE"],
                  "vmd": ["type dx", "Isosurface", "graphics $gmol sphere"]}
        for fmt, needles in expect.items():
            with self.subTest(fmt=fmt):
                files = run_visualization(self.p("sites.json"), format=fmt, mode="sites", output_file=self.p(f"v_{fmt}"))
                text = open(files[0]).read()
                for needle in needles:
                    self.assertIn(needle, text)

    def test_modes_and_files_must_match(self):
        self.run_sites()
        with self.assertLogs("gephyra.visualize", level="ERROR"):
            self.assertIsNone(run_visualization(self.p("sites.json"), format="pymol", mode="cluster",
                                                output_file=self.p("x")))

    def _vmd_wrapper(self, script):
        wrapper = self.p("vmd_run.tcl")
        with open(wrapper, "w") as f:
            f.write(f"source {{{script}}}\nquit\n")
        return wrapper

    @unittest.skipUnless(os.environ.get("GEPHYRA_TEST_VIEWERS"), "set GEPHYRA_TEST_VIEWERS=1 to open scripts in viewers")
    def test_viewers_open_the_sites_scene(self):
        self.run_sites()
        for fmt, cmd in (("chimerax", lambda s: ["chimerax", "--nogui", "--exit", "--cmd", f'open "{s}"']),
                         ("chimera", lambda s: ["xvfb-run", "-a", "chimera", "--nogui", "--nostatus", s]),
                         ("pymol", lambda s: ["pymol", "-cq", s]),
                         ("vmd", lambda s: ["vmd", "-dispdev", "text", "-e", self._vmd_wrapper(s)])):
            if not shutil.which(cmd("x")[0]):
                continue
            with self.subTest(fmt=fmt):
                script = run_visualization(self.p("sites.json"), format=fmt, mode="sites",
                                           output_file=self.p(f"o_{fmt}"))[0]
                res = subprocess.run(cmd(script), capture_output=True, text=True, timeout=300)
                out = res.stdout + res.stderr
                for bad in ("Traceback", "MidasError", "Error:", "is undefined", "invalid graphics", "Unable to"):
                    self.assertNotIn(bad, out)


class TestInterfaceAndCli(_Tmp):
    def test_cli(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["-q", "hydration", "--topo", self.topo, "--traj", self.traj, "--root", "resname LIG",
                         "--water", "resname WAT", "--align", "resname SCF", "--max_shell", "2",
                         "--output", self.p("cli.json")])
        self.assertEqual(code, 0)
        self.assertEqual(len(load_sites(self.p("cli.json"))[0]), 3)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            main(["hydration", "--topo", "t", "--traj", "x", "--root", "r", "--max_shell", "4"])
        self.assertEqual(cm.exception.code, 2)

    def test_form_parsing_and_command_line(self):
        values = dict(forms.DEFAULTS["hydration"], topo=self.topo, traj=self.traj, root="resname LIG",
                      water="resname WAT", max_shell="2", align="resname SCF")
        kw = forms.parse_hydration(values, self.tmp)
        self.assertEqual((kw["max_shell"], kw["min_occupancy"], kw["align_sel"]), (2, 0.2, "resname SCF"))
        self.assertEqual(kw["dx_file"], self.p("hydration_map.dx"))
        args = build_parser().parse_args(shlex.split(forms.build_command_line("hydration", kw))[1:])
        self.assertEqual((args.max_shell, args.align, args.map), (2, "resname SCF", self.p("hydration_map.dx")))
        for field, bad in (("max_shell", "4"), ("min_occupancy", "2"), ("traj", "")):
            with self.subTest(field=field), self.assertRaises(forms.FormError) as cm:
                forms.parse_hydration(dict(values, **{field: bad}), self.tmp)
            self.assertEqual(cm.exception.field, field)

    def test_visualize_form_checks_sites_files(self):
        self.run_sites()
        base = dict(forms.DEFAULTS["visualize"], data=self.p("sites.json"))
        kw = forms.parse_visualize(dict(base, mode="sites", top_clusters="2"), self.tmp)
        self.assertEqual((kw["mode"], kw["top_clusters"]), ("sites", 2))
        with self.assertRaises(forms.FormError):
            forms.parse_visualize(dict(base, mode="cluster"), self.tmp)
