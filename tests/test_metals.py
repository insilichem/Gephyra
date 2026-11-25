"""
Metals as starting points, for both techniques, on a hypothetical system: a
copper ion (no element record, as in MCPB-style topologies) with one
coordinated water and a hydrogen-bonded chain of waters leading away from it.
"""

import json
import os
import shutil
import tempfile
import unittest

import MDAnalysis as mda
import numpy as np

from gephyra.analysis import run_analysis
from gephyra.hydration import run_hydration_sites
from gephyra.metals import METALS, coordination_probability, metal_symbol
from gephyra.results import load_sites
from gephyra.visualize import run_visualization

N_FRAMES = 4


def make_copper_system(directory):
    """Cu at the origin; waters at 2.0, 4.8, 7.6, 10.4 A along x, each donating to the next."""
    from MDAnalysis.coordinates.memory import MemoryReader
    xs = (2.0, 4.8, 7.6, 10.4)
    names = ["CU"] + ["O", "H1", "H2"] * len(xs)
    resindex = [0] + [1 + i // 3 for i in range(3 * len(xs))]
    u = mda.Universe.empty(len(names), n_residues=1 + len(xs), atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", ["CU"] + ["WAT"] * len(xs))
    u.add_TopologyAttr("resid", list(range(1, 2 + len(xs))))
    u.add_TopologyAttr("masses", [63.55] + [16.0, 1.008, 1.008] * len(xs))
    bonds = []
    for k in range(len(xs)):
        o = 1 + 3 * k
        bonds += [(o, o + 1), (o, o + 2)]
    u.add_TopologyAttr("bonds", bonds)
    base = [[0.0, 0.0, 0.0]]
    for i, x in enumerate(xs):
        # the coordinated water's hydrogens point away from Cu; the others donate along +x
        base += [[x, 0.0, 0.0], [x + 0.96, 0.0, 0.0], [x - 0.24, 0.93, 0.0]]
    rng = np.random.default_rng(3)
    frames = [np.array(base) + 30.0 + rng.normal(0, 0.03, (len(names), 3)) for _ in range(N_FRAMES)]
    u.load_new(np.array(frames, dtype=np.float32), format=MemoryReader)
    for ts in u.trajectory:
        ts.dimensions = [80.0, 80.0, 80.0, 90.0, 90.0, 90.0]
    topo, traj = os.path.join(directory, "cu.pdb"), os.path.join(directory, "cu.dcd")
    u.trajectory[0]
    u.atoms.write(topo, bonds="all")
    with mda.Writer(traj, len(u.atoms)) as w:
        for _ in u.trajectory:
            w.write(u.atoms)
    return topo, traj


def _atom(name, resname, n_atoms=1, mass=None, element=None):
    u = mda.Universe.empty(n_atoms, n_residues=1, atom_resindex=[0] * n_atoms, trajectory=True)
    u.add_TopologyAttr("name", [name] + ["X"] * (n_atoms - 1))
    u.add_TopologyAttr("resname", [resname])
    if mass is not None:
        u.add_TopologyAttr("masses", [mass] + [12.0] * (n_atoms - 1))
    if element is not None:
        u.add_TopologyAttr("elements", [element] + ["C"] * (n_atoms - 1))
    return u.atoms[0]


class TestRecognition(unittest.TestCase):
    def test_spellings(self):
        for name, resname in (("Cu", "CU"), ("CU", "CU"), ("CU1", "CUA"), ("Zn2+", "ZN"), ("FE", "HEM")):
            with self.subTest(name=name):
                self.assertIsNotNone(metal_symbol(_atom(name, resname)))
        self.assertEqual(metal_symbol(_atom("Cu", "CU")), "CU")
        self.assertEqual(metal_symbol(_atom("Zn2+", "ZN")), "ZN")

    def test_not_metals(self):
        self.assertIsNone(metal_symbol(_atom("CA", "ALA", mass=12.01)))          # protein C-alpha
        self.assertIsNone(metal_symbol(_atom("CA", "LIG", n_atoms=5, mass=12.0)))  # ligand carbon named CA
        self.assertIsNone(metal_symbol(_atom("OW", "WAT")))
        self.assertIsNone(metal_symbol(_atom("C1", "LIG", element="C")))

    def test_element_and_mass(self):
        self.assertEqual(metal_symbol(_atom("M1", "MET", element="Zn")), "ZN")
        self.assertEqual(metal_symbol(_atom("M1", "MX1", mass=63.55)), "CU")     # name says nothing; mass does
        self.assertEqual(metal_symbol(_atom("CA", "CA", mass=40.08)), "CA")      # calcium ion

    def test_coordination_score(self):
        ideal = METALS["CU"][2]
        self.assertAlmostEqual(coordination_probability(ideal, "CU"), 1.0)
        self.assertGreater(coordination_probability(ideal + 0.3, "CU"), 0.0)
        self.assertEqual(coordination_probability(ideal + 0.7, "CU"), 0.0)
        self.assertLess(coordination_probability(1.2, "CU"), 0.1)               # clash


class TestMetalRoots(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.topo, self.traj = make_copper_system(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def p(self, *parts):
        return os.path.join(self.tmp, *parts)

    def test_water_network_starts_at_the_metal(self):
        with self.assertLogs("gephyra.analysis", level="INFO") as logs:
            summary = run_analysis(self.topo, self.traj, "resname CU", water_sel="resname WAT", max_depth=4,
                                   min_depth=1, output_file=self.p("r.jsonl"))
        self.assertIn("is a metal: Cu (copper", "\n".join(logs.output))
        self.assertGreater(summary["n_paths"], 0)
        frames = [json.loads(l) for l in open(self.p("r.jsonl")) if l.startswith('{"type": "frame"')]
        longest = max((p for f in frames for p in f["paths"]), key=lambda p: p["length"])
        self.assertEqual(longest["length"], 4)                                 # Cu -> 4 waters
        self.assertEqual(longest["nodes"][0], 0)

    def test_default_min_depth_keeps_bridges_from_the_metal(self):
        summary = run_analysis(self.topo, self.traj, "resname CU", water_sel="resname WAT", max_depth=4,
                               output_file=self.p("r3.jsonl"))
        self.assertGreater(summary["n_paths"], 0)                              # chains of 3-4 waters

    def test_hydration_marks_the_coordinated_water(self):
        with self.assertLogs("gephyra.analysis", level="INFO") as logs:
            run_hydration_sites(self.topo, self.traj, "resname CU", water_sel="resname WAT", align_sel=None,
                                max_shell=2, output_file=self.p("s.json"), csv_file=self.p("s.csv"))
        self.assertIn("is a metal: Cu (copper", "\n".join(logs.output))
        sites, _ = load_sites(self.p("s.json"))
        coordinated = [s for s in sites if s["coordinated_to"]]
        self.assertEqual(len(coordinated), 1)
        self.assertEqual(coordinated[0]["coordinated_to"], "CU1:CU")
        self.assertEqual(coordinated[0]["shell"], 1)
        self.assertAlmostEqual(coordinated[0]["contacts"][0]["mean_distance"], 2.0, places=1)
        self.assertTrue(all(s["coordinated_to"] is None for s in sites if s is not coordinated[0]))
        self.assertIn("Coordinated_to", open(self.p("s.csv")).readline())

    def test_metal_root_is_drawn_as_a_sphere(self):
        run_analysis(self.topo, self.traj, "resname CU", water_sel="resname WAT", max_depth=4, min_depth=1,
                     output_file=self.p("r.jsonl"))
        script = run_visualization(self.p("r.jsonl"), format="chimerax", mode="frame", output_file=self.p("v"))[0]
        self.assertIn("style #101:CU ball", open(script).read())


if __name__ == "__main__":
    unittest.main()
