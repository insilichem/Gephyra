import importlib
import unittest
import sys
from unittest.mock import MagicMock

# `gephyra.analysis` imports MDAnalysis, numpy, networkx, scipy and
# similaritymeasures at module level, so this file has to make those names
# importable before it can pull in compute_persistence.
#
# Only stand in for a dependency that is genuinely missing. Unconditionally
# assigning mocks into sys.modules poisons the whole pytest session: every test
# collected after this file inherits them, so any later test doing real
# numerical work silently operates on MagicMocks and fails in ways that have
# nothing to do with what it is testing.
_MOCKED = {}

def _mock_if_missing(name, configure=None):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
    except ImportError:
        mock = MagicMock()
        if configure is not None:
            configure(mock)
        _MOCKED[name] = mock
        sys.modules[name] = mock

def _configure_numpy(mock):
    # Just enough behaviour for compute_persistence.
    mock.mean.side_effect = lambda x, **kwargs: float(sum(x) / len(x)) if x else 0.0
    mock.max.side_effect = lambda x, **kwargs: int(max(x)) if x else 0

for _name in (
    "MDAnalysis",
    "MDAnalysis.lib",
    "MDAnalysis.lib.distances",
    "MDAnalysis.exceptions",
    "networkx",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.distance",
    "scipy.cluster",
    "scipy.cluster.hierarchy",
    "similaritymeasures",
):
    _mock_if_missing(_name)

_mock_if_missing("numpy", _configure_numpy)

def tearDownModule():
    # Leave sys.modules exactly as it was found.
    for _name in list(_MOCKED):
        if sys.modules.get(_name) is _MOCKED[_name]:
            del sys.modules[_name]
    _MOCKED.clear()

# Now import the function to test
from gephyra.analysis import compute_persistence

class TestComputePersistence(unittest.TestCase):
    def test_empty_list(self):
        mean_p, max_p = compute_persistence([], 10, 1)
        self.assertEqual(mean_p, 0.0)
        self.assertEqual(max_p, 0)

    def test_single_frame(self):
        mean_p, max_p = compute_persistence([5], 10, 1)
        self.assertEqual(mean_p, 1.0)
        self.assertEqual(max_p, 1)

    def test_continuous_sequence(self):
        # Frame indices: 1, 2, 3. Stride: 1.
        # Run: [1, 2, 3] -> length 3
        mean_p, max_p = compute_persistence([1, 2, 3], 10, 1)
        self.assertEqual(mean_p, 3.0)
        self.assertEqual(max_p, 3)

    def test_discontinuous_sequence(self):
        # Frame indices: 1, 2, 4, 5, 6. Stride: 1.
        # Runs: [1, 2] (len 2), [4, 5, 6] (len 3)
        # Mean: (2+3)/2 = 2.5
        # Max: 3
        mean_p, max_p = compute_persistence([1, 2, 4, 5, 6], 10, 1)
        self.assertEqual(mean_p, 2.5)
        self.assertEqual(max_p, 3)

    def test_stride_greater_than_one(self):
        # Frame indices: 0, 2, 4. Stride: 2.
        # Run: [0, 2, 4] -> length 3
        mean_p, max_p = compute_persistence([0, 2, 4], 10, 2)
        self.assertEqual(mean_p, 3.0)
        self.assertEqual(max_p, 3)

    def test_discontinuous_with_stride(self):
        # Frame indices: 0, 2, 6, 8, 10. Stride: 2.
        # Runs: [0, 2] (len 2), [6, 8, 10] (len 3)
        # Mean: 2.5, Max: 3
        mean_p, max_p = compute_persistence([0, 2, 6, 8, 10], 10, 2)
        self.assertEqual(mean_p, 2.5)
        self.assertEqual(max_p, 3)

    def test_all_single_frame_runs(self):
        # Frame indices: 0, 2, 4. Stride: 1.
        # Runs: [0] (len 1), [2] (len 1), [4] (len 1)
        # Mean: 1.0, Max: 1
        mean_p, max_p = compute_persistence([0, 2, 4], 10, 1)
        self.assertEqual(mean_p, 1.0)
        self.assertEqual(max_p, 1)

if __name__ == '__main__':
    unittest.main()
