import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np

from gephyra import estimate
from gephyra.analysis import run_analysis
from gephyra.cli import main
from gephyra.estimate import _sample_frames, estimate_runtime, format_duration


def make_system(directory, n_frames=6):
    """A ligand oxygen and a three-water chain; waters jiggle between frames."""
    from MDAnalysis.coordinates.memory import MemoryReader
    names = ["O1"] + ["O", "H1", "H2"] * 3
    u = mda.Universe.empty(len(names), n_residues=4, atom_resindex=[0] + [1 + i // 3 for i in range(9)],
                           trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", ["LIG"] + ["WAT"] * 3)
    u.add_TopologyAttr("resid", [1, 2, 3, 4])
    u.add_TopologyAttr("elements", ["O"] + ["O", "H", "H"] * 3)
    base = [[0.0, 0.0, 0.0]]
    for x in (2.8, 5.6, 8.4):
        base += [[x, 0.0, 0.0], [x + 0.6, 0.8, 0.0], [x + 0.6, -0.8, 0.0]]
    rng = np.random.default_rng(1)
    frames = [np.array(base) + 20.0 + rng.normal(0, 0.05, (10, 3)) for _ in range(n_frames)]
    u.load_new(np.array(frames, dtype=np.float32), format=MemoryReader)
    for ts in u.trajectory:
        ts.dimensions = [60.0, 60.0, 60.0, 90.0, 90.0, 90.0]
    topo, traj = os.path.join(directory, "sys.pdb"), os.path.join(directory, "sys.dcd")
    u.trajectory[0]
    u.atoms.write(topo)
    with mda.Writer(traj, len(u.atoms)) as w:
        for _ in u.trajectory:
            w.write(u.atoms)
    return topo, traj


class TestHelpers(unittest.TestCase):
    def test_format_duration(self):
        self.assertEqual(format_duration(0.2), "about 1 s")
        self.assertEqual(format_duration(42), "about 42 s")
        self.assertEqual(format_duration(240), "about 4 min")
        self.assertEqual(format_duration(2 * 3600 + 15 * 60), "about 2 h 15 min")
        self.assertEqual(format_duration(3 * 3600), "about 3 h")

    def test_sample_frames_spread_over_the_analysed_frames(self):
        self.assertEqual(_sample_frames(range(0, 11200, 100), 3), [0, 5600, 11100])
        self.assertEqual(_sample_frames(range(0, 3), 5), [0, 1, 2])


class TestEstimate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.topo, self.traj = make_system(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_estimate_reports_system_and_time(self):
        calls = []
        result = estimate_runtime(self.topo, self.traj, "resname LIG", water_sel="resname WAT", stride=2,
                                  max_depth=3, output_file="ignored.jsonl", align_sel="auto",
                                  progress_callback=lambda d, t: calls.append((d, t)))
        self.assertEqual((result["n_atoms"], result["n_residues"], result["n_waters"]), (10, 4, 3))
        self.assertEqual((result["n_trajectory_frames"], result["n_analysed"]), (6, 3))
        self.assertEqual(result["sampled_frames"], [0, 2, 4])
        self.assertEqual(calls[-1], (3, 3))
        self.assertLessEqual(result["low_s"], result["estimate_s"])
        self.assertLessEqual(result["estimate_s"], result["high_s"])
        self.assertAlmostEqual(sum(result["stage_share"].values()), 1.0, places=6)
        self.assertIn("for 3 of 6 frames (stride 2)", result["summary"])
        self.assertIn("10 atoms, 4 residues, 3 water molecules", result["summary"])

    def test_long_runs_suggest_a_stride_and_max_depth(self):
        with patch.object(estimate, "TARGET_SECONDS", 1e-4), \
                patch.object(estimate, "_sample_frames", side_effect=lambda frames, n: list(frames)[:1]):
            # Pretend the system is huge: 10 s per frame.
            with patch.object(estimate.time, "perf_counter", side_effect=iter(np.arange(0, 1000, 2.5))):
                result = estimate_runtime(self.topo, self.traj, "resname LIG", water_sel="resname WAT",
                                          stride=1, max_depth=10)
        text = " ".join(result["hints"])
        self.assertIn("A stride of", text)
        self.assertIn("Max depth of 6 instead of 10", text)

    def test_errors_are_raised_before_anything_runs(self):
        with self.assertRaises(ValueError):
            estimate_runtime(self.topo, self.traj, "resname NOPE")
        with self.assertRaises(ValueError):
            estimate_runtime(self.topo, "", "resname LIG")
        with self.assertRaises(FileNotFoundError):
            estimate_runtime(self.topo, self.p_missing(), "resname LIG")
        with self.assertRaises(ValueError):
            estimate_runtime(self.topo, self.traj, "resname LIG", stride=0)

    def p_missing(self):
        return os.path.join(self.tmp, "missing.dcd")

    def test_cli_estimate_only_prints_and_writes_nothing(self):
        out_file = os.path.join(self.tmp, "should_not_exist.jsonl")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main(["-q", "calculate", "--topo", self.topo, "--traj", self.traj, "--root", "resname LIG",
                         "--water", "resname WAT", "--estimate", "--output", out_file])
        self.assertEqual(code, 0)
        self.assertIn("Estimated runtime:", out.getvalue())
        self.assertFalse(os.path.exists(out_file))

    def test_run_logs_time_left(self):
        with self.assertLogs("gephyra.analysis", level="INFO") as logs:
            run_analysis(self.topo, self.traj, "resname LIG", water_sel="resname WAT", max_depth=3,
                         output_file=os.path.join(self.tmp, "r.jsonl"))
        text = "\n".join(logs.output)
        self.assertIn("(1/6)", text)
        self.assertIn("(6/6, about", text)
        self.assertIn("left)", text)


if __name__ == "__main__":
    unittest.main()
