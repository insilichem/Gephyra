import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import MDAnalysis as mda
import numpy as np

from gephyra import __version__
from gephyra.analysis import cluster_pathways, run_analysis, validate_calculation_parameters
from gephyra.results import load_clusters
from gephyra.cli import main
from gephyra.math_utils import condensed_submatrix, normalize_hquality, pairwise_discrete_frechet
from gephyra.visualize import _bild_path, _resolve_frame, detect_data_kind, run_visualization

try:
    import similaritymeasures
except ImportError:  # optional test dependency
    similaritymeasures = None


class TestPairwiseFrechet(unittest.TestCase):
    def test_hand_computed_value(self):
        # Two parallel 3-point lines offset by 1 A: every coupling is 1 A apart.
        a = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        b = [[0, 1, 0], [1, 1, 0], [2, 1, 0]]
        self.assertAlmostEqual(pairwise_discrete_frechet([a, b])[0], 1.0)

    def test_degenerate_inputs(self):
        self.assertEqual(len(pairwise_discrete_frechet([])), 0)
        self.assertEqual(len(pairwise_discrete_frechet([[[0, 0, 0], [1, 1, 1]]])), 0)

    @unittest.skipIf(similaritymeasures is None, "similaritymeasures not installed")
    def test_matches_reference_with_mixed_lengths_and_small_batches(self):
        rng = np.random.default_rng(42)
        curves = [rng.normal(size=(rng.integers(2, 9), 3)) * 3 for _ in range(40)]
        expected = [similaritymeasures.frechet_dist(curves[i], curves[j])
                    for i in range(len(curves)) for j in range(i + 1, len(curves))]
        # A tiny batch size forces every length group to be split across chunks.
        got = pairwise_discrete_frechet(curves, max_pairs_per_batch=7)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_condensed_submatrix_matches_squareform(self):
        from scipy.spatial.distance import squareform
        rng = np.random.default_rng(0)
        curves = [rng.normal(size=(4, 3)) for _ in range(12)]
        cond = pairwise_discrete_frechet(curves)
        idx = [7, 2, 11, 0]
        np.testing.assert_array_equal(condensed_submatrix(cond, 12, idx), squareform(cond)[np.ix_(idx, idx)])

    def test_progress_callback_reaches_total(self):
        calls = []
        curves = [np.zeros((3, 3)) + i for i in range(6)]
        pairwise_discrete_frechet(curves, max_pairs_per_batch=4, progress_callback=lambda d, t: calls.append((d, t)))
        self.assertEqual(calls[-1], (15, 15))


def _write_jsonl(path, n_frames, frames):
    with open(path, 'w') as f:
        f.write(json.dumps({"type": "metadata", "n_frames_analyzed": n_frames, "parameters": {"stride": 1}}) + "\n")
        for idx, paths in frames:
            f.write(json.dumps({"type": "frame", "frame_idx": idx, "paths": paths}) + "\n")


class TestClusterSinglePath(unittest.TestCase):
    PATH = {"nodes": [0, 1, 2], "coords": [[0, 0, 0], [1, 0, 0], [2, 0, 0]], "hquality": 0.5}

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = os.path.join(self.tmp, "r.jsonl")
        self.out = os.path.join(self.tmp, "c.json")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_single_path_in_one_frame_is_discarded_like_any_other_cluster(self):
        _write_jsonl(self.data, 5, [(0, [self.PATH]), (1, [])])
        cluster_pathways(self.data, output_file=self.out)
        self.assertEqual(load_clusters(self.out)[0], [])

    def test_single_path_in_several_frames_is_kept(self):
        _write_jsonl(self.data, 4, [(0, [self.PATH]), (1, [self.PATH]), (3, [self.PATH])])
        cluster_pathways(self.data, output_file=self.out)
        clusters = load_clusters(self.out)[0]
        self.assertEqual(len(clusters), 1)
        self.assertAlmostEqual(clusters[0]["occupancy"], 0.75)
        self.assertEqual(clusters[0]["max_persistence_frames"], 2)
        self.assertAlmostEqual(clusters[0]["avg_hquality"], 0.5)
        self.assertAlmostEqual(clusters[0]["avg_hquality_norm"], 0.5 / 1.5)
        self.assertNotIn("avg_prob_Hquality", clusters[0])

    def test_legacy_probability_key_is_still_read(self):
        legacy = {k: v for k, v in self.PATH.items() if k != "hquality"}
        legacy["probability"] = 2.0
        _write_jsonl(self.data, 2, [(0, [legacy]), (1, [legacy])])
        cluster_pathways(self.data, output_file=self.out)
        clusters = load_clusters(self.out)[0]
        self.assertAlmostEqual(clusters[0]["avg_hquality"], 2.0)
        self.assertAlmostEqual(clusters[0]["avg_hquality_norm"], 2.0 / 3.0)

    def test_rejects_non_positive_threshold(self):
        _write_jsonl(self.data, 1, [(0, [self.PATH])])
        with self.assertRaises(ValueError):
            cluster_pathways(self.data, threshold=0, output_file=self.out)


class TestNormalizeHquality(unittest.TestCase):
    def test_bounded_and_monotonic(self):
        values = [0.0, 1e-6, 0.5, 1.0, 3.0, 50.0, 1e6]
        normed = [normalize_hquality(z) for z in values]
        self.assertEqual(normed[0], 0.0)
        self.assertAlmostEqual(normalize_hquality(1.0), 0.5)
        self.assertTrue(all(0.0 <= v < 1.0 for v in normed))
        self.assertEqual(normed, sorted(normed))


class TestParameterValidation(unittest.TestCase):
    def test_valid_defaults(self):
        validate_calculation_parameters()

    def test_invalid_values(self):
        for kwargs in ({"stride": 0}, {"max_depth": 0}, {"min_depth": 0},
                       {"min_depth": 5, "max_depth": 3}, {"cooperativity": 0.0},
                       {"cooperativity": 1.5}, {"coarse_cutoff": -1.0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                validate_calculation_parameters(**kwargs)


class TestRunAnalysisEndToEnd(unittest.TestCase):
    """Runs `calculate` on a 4-atom PDB: LIG O1 -> OW -> OW, plus a distant OW."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        u = mda.Universe.empty(4, trajectory=True, n_residues=4, atom_resindex=[0, 1, 2, 3])
        u.add_TopologyAttr('resname', ['LIG', 'SOL', 'SOL', 'SOL'])
        u.add_TopologyAttr('name', ['O1', 'OW', 'OW', 'OW'])
        u.add_TopologyAttr('resid', [1, 2, 3, 4])
        u.atoms.positions = np.array([[0.0, 0, 0], [2.8, 0, 0], [5.6, 0, 0], [20.0, 0, 0]])
        u.dimensions = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0])
        self.pdb = os.path.join(self.tmp, "sys.pdb")
        u.atoms.write(self.pdb)
        self.out = os.path.join(self.tmp, "r.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, **kwargs):
        with self.assertLogs("gephyra.analysis", level="INFO") as logs:
            run_analysis(self.pdb, self.pdb, "resname LIG", water_sel="resname SOL",
                         output_file=self.out, **kwargs)
        return "\n".join(logs.output)

    def test_min_depth_filters_and_stats_count_only_written_paths(self):
        log = self._run(min_depth=2, max_depth=3)
        with open(self.out) as f:
            objs = [json.loads(line) for line in f if line.strip()]
        paths = [p for o in objs if o["type"] == "frame" for p in o["paths"]]
        self.assertEqual([p["nodes"] for p in paths], [[0, 1, 2]])
        self.assertIn("Total paths written across analyzed frames: 1", log)
        self.assertIn("Paths discarded by --min_depth 2: 1", log)
        # Average depth must be that of the written path alone (2), not diluted
        # by the discarded depth-1 path.
        self.assertIn("Average path length (depth): 2.00", log)

    def test_outputs_use_hquality_names(self):
        import csv
        csv_out = os.path.join(self.tmp, "r.csv")
        self._run(max_depth=3, csv_file=csv_out)
        with open(self.out) as f:
            objs = [json.loads(line) for line in f if line.strip()]
        paths = [p for o in objs if o["type"] == "frame" for p in o["paths"]]
        self.assertTrue(paths)
        for p in paths:
            self.assertNotIn("probability", p)
            self.assertGreater(p["hquality"], 0.0)
            self.assertAlmostEqual(p["hquality_norm"], normalize_hquality(p["hquality"]))
        with open(csv_out, newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertNotIn("Cumulative_Probability", rows[0])
        self.assertEqual(len(rows), len(paths))
        for row, p in zip(rows, paths):
            self.assertAlmostEqual(float(row["HQuality_Z"]), p["hquality"])
            self.assertAlmostEqual(float(row["HQuality_Norm"]), p["hquality_norm"])

    def test_empty_root_selection_raises(self):
        with self.assertRaises(ValueError):
            run_analysis(self.pdb, self.pdb, "resname NOPE", water_sel="resname SOL", output_file=self.out)

    def test_trajectory_is_required(self):
        for traj in (None, ""):
            with self.subTest(traj=traj), self.assertRaises(ValueError) as cm:
                run_analysis(self.pdb, traj, "resname LIG", output_file=self.out)
            self.assertIn("trajectory file is required", str(cm.exception))
        with self.assertRaises(FileNotFoundError):
            run_analysis(self.pdb, "missing.dcd", "resname LIG", output_file=self.out)

    def test_metadata_records_the_analysed_system(self):
        self._run(min_depth=1, max_depth=2)
        with open(self.out) as f:
            meta = json.loads(f.readline())
        self.assertEqual(meta["topo_file"], os.path.abspath(self.pdb))
        self.assertEqual(meta["traj_file"], os.path.abspath(self.pdb))

    def test_invalid_parameters_raise_before_loading(self):
        with self.assertRaises(ValueError):
            run_analysis("does_not_exist.pdb", None, "resname LIG", stride=0, output_file=self.out)


class TestVisualizationHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_bild_path_only_replaces_extension(self):
        p = _bild_path(os.path.join("my.python_dir", "out.py"))
        self.assertTrue(p.endswith(os.path.join("my.python_dir", "out.bild")))

    def test_resolve_frame_warns_on_missing_frame(self):
        frames = {"0": [], "10": []}
        self.assertEqual(_resolve_frame(frames, 10), "10")
        self.assertIsNone(_resolve_frame({}, 3))
        with self.assertLogs("gephyra.visualize", level="WARNING"):
            self.assertEqual(_resolve_frame(frames, 5), "0")

    def test_detect_kind_and_reject_mismatched_mode(self):
        jsonl = os.path.join(self.tmp, "r.jsonl")
        _write_jsonl(jsonl, 1, [(0, [])])
        cl = os.path.join(self.tmp, "c.json")
        with open(cl, "w") as f:
            json.dump([], f)
        self.assertEqual(detect_data_kind(jsonl), "jsonl")
        self.assertEqual(detect_data_kind(cl), "cluster")

        out = os.path.join(self.tmp, "viz")
        with self.assertLogs("gephyra.visualize", level="ERROR"):
            run_visualization(jsonl, format="vmd", mode="cluster", output_file=out)
        with self.assertLogs("gephyra.visualize", level="ERROR"):
            run_visualization(cl, format="pymol", mode="density", output_file=out)
        self.assertFalse(os.path.exists(out + ".tcl") or os.path.exists(out + ".py"))

    def test_chimera_frame_writes_bild_next_to_script(self):
        jsonl = os.path.join(self.tmp, "r.jsonl")
        path = {"nodes": [0, 1], "atom_ids": [1, 2], "coords": [[0, 0, 0], [1, 1, 1]]}
        _write_jsonl(jsonl, 1, [(0, [path])])
        out_dir = os.path.join(self.tmp, "a.python")
        os.mkdir(out_dir)
        run_visualization(jsonl, format="chimera", mode="frame", frame_idx=0,
                          output_file=os.path.join(out_dir, "frame.py"))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "frame.bild")))


class TestCLI(unittest.TestCase):
    def _exit_code(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit) as cm:
                main(argv)
        return cm.exception.code, out.getvalue()

    def test_version(self):
        code, out = self._exit_code(["--version"])
        self.assertEqual(code, 0)
        self.assertIn(__version__, out)

    def test_rejects_bad_numeric_arguments(self):
        base = ["calculate", "--topo", "x.pdb", "--root", "resname LIG"]
        for extra in (["--stride", "0"], ["--cooperativity", "1.5"], ["--coarse_cutoff", "-2"],
                      ["--min_depth", "5", "--max_depth", "3"]):
            with self.subTest(extra=extra):
                code, _ = self._exit_code(base + extra)
                self.assertEqual(code, 2)

    def test_cluster_threshold_must_be_positive(self):
        code, _ = self._exit_code(["cluster", "--data", "x.jsonl", "--threshold", "0"])
        self.assertEqual(code, 2)


if __name__ == '__main__':
    unittest.main()
