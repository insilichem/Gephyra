import unittest
import os
import tempfile
import networkx as nx
from unittest.mock import patch
import MDAnalysis as mda
import numpy as np

from gephyra.math_utils import switching_function, calculate_hbond_probability
from gephyra.core import build_graph, compute_edge_probabilities, traverse_network
from gephyra.visualize import export_vmd_script, export_pymol_script, run_visualization
from gephyra.analysis import sanitize_csv_field

class TestAnalysis(unittest.TestCase):
    def test_sanitize_csv_field(self):
        # Normal strings should remain unchanged
        self.assertEqual(sanitize_csv_field("NormalRes"), "NormalRes")
        self.assertEqual(sanitize_csv_field("1-2-3"), "1-2-3")
        self.assertEqual(sanitize_csv_field(123), "123")

        # Dangerous characters at start should have a single quote prepended
        self.assertEqual(sanitize_csv_field("=cmd|' /C calc'!A0"), "'=cmd|' /C calc'!A0")
        self.assertEqual(sanitize_csv_field("+1+2"), "'+1+2")
        self.assertEqual(sanitize_csv_field("-1-2-3"), "'-1-2-3")
        self.assertEqual(sanitize_csv_field("@SUM(A1:A10)"), "'@SUM(A1:A10)")

        # Dangerous characters not at start should remain unchanged
        self.assertEqual(sanitize_csv_field("Res="), "Res=")
        self.assertEqual(sanitize_csv_field("A+B"), "A+B")
        self.assertEqual(sanitize_csv_field("1-2-3-"), "1-2-3-")
        self.assertEqual(sanitize_csv_field("foo@bar"), "foo@bar")

class TestMathUtils(unittest.TestCase):
    def test_switching_function(self):
        # When dist >= threshold (ratio >= 1.0), it should return 0.0 to enforce monotonic decay
        self.assertAlmostEqual(switching_function(1.0, 1.0, 8, 12), 0.0)

        # When dist <= 0 (ratio <= 0.0), it should return 1.0 to clamp fully formed bonds
        self.assertAlmostEqual(switching_function(0.0, 1.0, 8, 12), 1.0)

        # When dist < threshold, it should be high
        prob_high = switching_function(0.5, 1.0)
        self.assertTrue(0.5 < prob_high <= 1)

    def test_calculate_hbond_probability(self):
        # Ideal mock h-bond values
        prob = calculate_hbond_probability(2.7, 1.0, 1.7)
        self.assertTrue(prob >= 0.0)

class TestCoreFunctions(unittest.TestCase):
    def setUp(self):
        # Create a tiny mock topology in memory using MDAnalysis Universe
        # 3 waters, 1 ligand
        self.u = mda.Universe.empty(4, trajectory=True, n_residues=4, atom_resindex=[0, 1, 2, 3])
        self.u.add_TopologyAttr('resname', ['LIG', 'SOL', 'SOL', 'SOL'])
        self.u.add_TopologyAttr('name', ['O1', 'OW', 'OW', 'OW'])
        self.u.atoms.positions = np.array([
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0], # Connected to LIG
            [5.0, 0.0, 0.0], # Connected to first water
            [10.0, 0.0, 0.0] # Far away, no connection
        ])
        # Box dimensions needed for capped_distance
        self.u.dimensions = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])

    def test_build_graph(self):
        water_atoms = self.u.select_atoms("resname SOL")
        root_atoms = self.u.select_atoms("resname LIG")
        g, roots = build_graph(self.u, water_atoms, root_atoms, max_distance=3.5)
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0], 0)
        # Edges expected: 0-1 (dist 2.5), 1-2 (dist 2.5)
        self.assertIn((0, 1), g.edges)
        self.assertIn((1, 2), g.edges)
        self.assertNotIn((2, 3), g.edges) # Distance 5.0 is > 3.5

    def test_traverse_network(self):
        g = nx.Graph()
        g.add_node(0)
        g.add_node(1)
        g.add_node(2)

        g.add_edge(0, 1, prob=0.8, weight=-np.log(0.8))
        g.add_edge(1, 2, prob=0.8, weight=-np.log(0.8))

        paths = traverse_network(g, [0], max_depth=5, prob_threshold=0.5)

        # Valid paths should include [0, 1] and [0, 1, 2]
        # (Actually, because of the loop structure, when it expands to 2,
        # [0, 1] might just expand and not be recorded separately if we only record terminal paths,
        # but let's just check length)

        self.assertTrue(len(paths) > 0)

        # We should find the path to node 2
        found_longest = False
        for p, prob in paths:
            if p == [0, 1, 2]:
                found_longest = True
                # weight = -np.log(0.8) + (-np.log(0.8) * 0.92)
                # prob = np.exp(-weight) = 0.8 * (0.8 ** 0.92) = 0.6515275355087862
                self.assertAlmostEqual(prob, 0.6515275355087862)
        self.assertTrue(found_longest)

class TestVisualization(unittest.TestCase):
    def test_export_vmd_script(self):
        # We need to mock a jsonl file now since visualization reads files directly
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode='w') as f_json:
            json_name = f_json.name
            # Write mock frame data
            f_json.write('{"type": "metadata"}\n')
            f_json.write('{"type": "frame", "frame_idx": 0, "paths": [{"nodes": [0, 1], "coords": [[0,0,0], [1,1,1]]}]}\n')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".tcl") as f:
            temp_name = f.name

        export_vmd_script(json_name, output_file=temp_name, mode="frame", frame_idx=0)

        with open(temp_name, 'r') as f:
            content = f.read()
            self.assertIn("mol selection \"index 0 1\"", content)

        os.remove(temp_name)
        os.remove(json_name)

    @patch('gephyra.visualize.logger.error')
    def test_run_visualization_file_not_found(self, mock_logger_error):
        run_visualization("non_existent_file.jsonl", format="vmd")
        mock_logger_error.assert_called_once_with("Data file non_existent_file.jsonl not found.")

    @patch('gephyra.visualize.export_vmd_script')
    def test_run_visualization_vmd_routing(self, mock_export):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pass
        try:
            run_visualization(temp_file.name, format="vmd", mode="frame", frame_idx=0, output_file="test_out")
            mock_export.assert_called_once_with(temp_file.name, output_file="test_out.tcl", mode="frame", frame_idx=0, cluster_id=None, max_bond_draw_dist=6.0)
        finally:
            os.remove(temp_file.name)

    @patch('gephyra.visualize.export_pymol_script')
    def test_run_visualization_pymol_routing(self, mock_export):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pass
        try:
            run_visualization(temp_file.name, format="pymol", mode="density", output_file="test_out")
            mock_export.assert_called_once_with(temp_file.name, output_file="test_out.py", mode="density", frame_idx=None, cluster_id=None, max_bond_draw_dist=6.0)
        finally:
            os.remove(temp_file.name)

    @patch('gephyra.visualize.export_chimera_script')
    def test_run_visualization_chimera_routing(self, mock_export):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pass
        try:
            run_visualization(temp_file.name, format="chimera", mode="frame", frame_idx=10, output_file="test_out.py")
            mock_export.assert_called_once_with(temp_file.name, output_file="test_out.py", mode="frame", frame_idx=10, cluster_id=None, max_bond_draw_dist=6.0)
        finally:
            os.remove(temp_file.name)

    @patch('gephyra.visualize.logger.error')
    def test_run_visualization_unknown_format(self, mock_logger_error):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pass
        try:
            run_visualization(temp_file.name, format="unknown", mode="density", output_file="test_out")
            mock_logger_error.assert_called_once_with("Unknown format: unknown")
        finally:
            os.remove(temp_file.name)


    @patch('gephyra.visualize.read_cluster_json')
    @patch('gephyra.visualize.read_jsonl')
    def test_export_vmd_cluster_script(self, mock_read_jsonl, mock_read_cluster_json):
        mock_read_cluster_json.return_value = [
            {"cluster_id": 0, "size": 10, "medoid_coords": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}
        ]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tcl") as f:
            temp_name = f.name

        export_vmd_script("dummy.json", output_file=temp_name, mode="cluster")

        with open(temp_name, 'r') as f:
            content = f.read()
            self.assertIn("graphics top cylinder {0.000 0.000 0.000}", content)
            self.assertIn("graphics top color orange", content)

        os.remove(temp_name)

    @patch('gephyra.analysis.logger.info')
    def test_cluster_coarse_trigger(self, mock_logger_info):
        from gephyra.analysis import cluster_pathways

        # Create a dummy jsonl file with 3 distinct paths
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode='w') as f_json:
            json_name = f_json.name
            f_json.write('{"type": "metadata", "n_frames_analyzed": 1}\n')
            # Path 1
            f_json.write('{"type": "frame", "frame_idx": 0, "paths": [{"nodes": [0, 1], "coords": [[0,0,0], [1,1,1]], "probability": 1.0}]}\n')
            # Path 2
            f_json.write('{"type": "frame", "frame_idx": 0, "paths": [{"nodes": [0, 2], "coords": [[0,0,0], [2,2,2]], "probability": 1.0}]}\n')
            # Path 3
            f_json.write('{"type": "frame", "frame_idx": 0, "paths": [{"nodes": [0, 3], "coords": [[0,0,0], [3,3,3]], "probability": 1.0}]}\n')

        # Test bypass logic: coarse_trigger=5 (we have 3 paths, so 3 <= 5 -> bypass)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f_out:
            out_name = f_out.name

        cluster_pathways(data_file=json_name, coarse_trigger=5, output_file=out_name)

        # Verify bypass message
        bypass_msg_found = any("Dataset small" in call.args[0] and "Bypassing 9D coarse filter" in call.args[0] for call in mock_logger_info.call_args_list)
        self.assertTrue(bypass_msg_found, "Expected bypass message for n_filtered <= coarse_trigger")

        mock_logger_info.reset_mock()

        # Test non-bypass logic: coarse_trigger=2 (we have 3 paths, so 3 > 2 -> run filter)
        cluster_pathways(data_file=json_name, coarse_trigger=2, output_file=out_name)

        # Verify run message
        run_msg_found = any("Dataset large" in call.args[0] and "Performing coarse 9D screening pass" in call.args[0] for call in mock_logger_info.call_args_list)
        self.assertTrue(run_msg_found, "Expected run message for n_filtered > coarse_trigger")

        os.remove(json_name)
        os.remove(out_name)

class TestFixtureClustering(unittest.TestCase):
    """
    End-to-end check against tests/data/dummy_data.jsonl, which carries the same
    line types the real `calculate` command emits: a metadata line, one frame
    line per frame with atom_ids, and a trailing path_statistics line.

    The fixture contains four spatial channels:
      - [1, 2, 3]   present in all 10 frames
      - [4, 5, 6]   present in all 10 frames
      - [1, 7, 8] / [1, 9, 10]  the same channel occupied by two different
        water pairs that exchange halfway through, 5 frames each
      - [1, 11, 12] present in a single frame only
    so a correct run merges the exchanging pair into one cluster at full
    occupancy and discards the single-frame path.
    """

    FIXTURE = os.path.join(os.path.dirname(__file__), "data", "dummy_data.jsonl")

    def test_fixture_has_full_output_schema(self):
        import json
        types = []
        with open(self.FIXTURE) as f:
            objs = [json.loads(line) for line in f if line.strip()]
        types = [o["type"] for o in objs]

        self.assertEqual(types[0], "metadata")
        self.assertEqual(types[-1], "path_statistics")
        self.assertIn("frame", types)

        frames = [o for o in objs if o["type"] == "frame"]
        self.assertEqual(len(frames), objs[0]["n_frames_analyzed"])
        for p in frames[0]["paths"]:
            for key in ("nodes", "atom_ids", "coords", "probability", "length", "avg_oo_dist"):
                self.assertIn(key, p)
            self.assertEqual(len(p["atom_ids"]), len(p["nodes"]))
            self.assertEqual(p["length"], len(p["nodes"]) - 1)

    def test_cluster_merges_exchange_and_drops_single_frame_noise(self):
        import json
        from gephyra.analysis import cluster_pathways

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            out_name = f.name

        cluster_pathways(data_file=self.FIXTURE, threshold=6.0, output_file=out_name)

        with open(out_name) as f:
            clusters = json.load(f)

        # Four spatial channels, minus the single-frame one discarded after merging.
        self.assertEqual(len(clusters), 3)

        # The exchanging pair merges into one cluster of size 2 at full occupancy.
        merged = [c for c in clusters if c["size"] == 2]
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["occupancy"], 1.0)
        self.assertEqual(merged[0]["max_persistence_frames"], 10)

        os.remove(out_name)

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
