"""
Continuous scoring primitives for hydrogen-bond geometry.

Every numerical constant in this module is an empirical smoothing parameter
chosen for Gephyra. None of them is inherited from MDAnalysis, whose
WaterBridgeAnalysis applies hard geometric cutoffs (a single 3.0 A
donor-acceptor distance and a fixed donor-H-acceptor angle) rather than a
graded score. Specifically:

- The 6/12 exponents of `switching_function` come from the rational switching
  form used in coordination-number collective variables (as in PLUMED and
  similar codes). They control only how sharply the function falls between 1
  and 0 across the interval [0, threshold]; they carry no energetic meaning and
  are not a Lennard-Jones 6-12 potential.
- The 0.6 A threshold on the angular proxy in `calculate_hbond_probability` is
  applied to the triangle-inequality excess r(DH) + r(HA) - r(DA), which is zero
  for a perfectly linear bond and grows as the bond bends. 0.6 A is the excess
  at which the bond is taken to have vanished, and it stands in for a minimum
  donor-H-acceptor angle without ever computing an angle.
- The 2.40 A onset of the steric term, and its 12th power, are likewise chosen
  to suppress unphysically short contacts rather than to reproduce any
  particular force field.

Changing these values rescales the reported scores, so results obtained with
different values are not comparable.
"""

import numpy as np

def switching_function(distance, threshold, power_num=6, power_den=12):
    ratio = distance / threshold
    if ratio >= 1.0:
        return 0.0
    if ratio <= 0.0:
        return 1.0

    return (1.0 - (ratio ** power_num)) / (1.0 - (ratio ** power_den))

def calculate_hbond_probability(mod_rOO, mod_rOiH, mod_rOjH, r0_oo=2.80, r0_threshold=0.45, ignore_angle=False):
    """
    Evaluates the continuous hydrogen bond weight between two molecules.
    """
    # Angle/Hydrogen placement component
    if ignore_angle:
        p_angle = 1.0
    else:
        h_dist_factor = mod_rOiH + mod_rOjH - mod_rOO
        p_angle = switching_function(h_dist_factor, threshold=0.6)

    # Heavy atom distance component
    oo_dist_factor = mod_rOO - r0_oo
    p_dist = switching_function(oo_dist_factor, threshold=r0_threshold)

    # Asymmetric 12th-power repulsive potential for steric clashes
    if mod_rOO < 2.40:
        p_steric = (mod_rOO / 2.40)**12
    else:
        p_steric = 1.0

    # Total fractional probability
    return p_angle * p_dist * p_steric


def normalize_hquality(z):
    """
    Maps the unbounded H-quality Z = sum(exp(-w)) onto [0, 1).

    Uses Z / (1 + Z): the Boltzmann-like weight of the bridged channel against
    a single reference state of weight 1 (no bridge). The map is monotonic, so
    it preserves every ranking made on Z; it is 0.5 at Z = 1 and saturates as
    degenerate channels accumulate routes. It is a bounded score, not a
    measured probability.
    """
    z = float(z)
    if z <= 0.0:
        return 0.0
    return z / (1.0 + z)


def _condensed_index(n, i, j):
    """Position of pair (i, j), i < j, in a SciPy condensed distance vector."""
    return n * i - (i * (i + 1)) // 2 + (j - i - 1)


def pairwise_discrete_frechet(curves, max_pairs_per_batch=200000, progress_callback=None):
    """
    Condensed matrix of discrete Frechet distances between every pair of curves.

    This is the Eiter-Mannila recurrence, the same one implemented by
    `similaritymeasures.frechet_dist`, so the values are identical. Instead of
    running the recurrence once per pair in pure Python, pairs whose curves have
    the same pair of lengths (n, m) are stacked and the n*m dynamic-programming
    steps are executed once for the whole stack with NumPy. Gephyra paths are at
    most max_depth + 1 points long, so there are only a handful of length
    combinations and each batch costs at most ~n*m vectorised operations.

    Args:
        curves: sequence of (L_k, D) array-likes. Lengths may differ.
        max_pairs_per_batch: upper bound on pairs processed in one vectorised
            step; bounds peak memory at roughly
            max_pairs_per_batch * n * m * 8 bytes per working array.
        progress_callback: optional callable(pairs_done, total_pairs).

    Returns:
        1-D float64 array of length N*(N-1)/2 in SciPy condensed order, suitable
        for scipy.cluster.hierarchy.linkage and scipy.spatial.distance.squareform.
    """
    curves = [np.asarray(c, dtype=np.float64) for c in curves]
    n_curves = len(curves)
    total_pairs = n_curves * (n_curves - 1) // 2
    condensed = np.empty(total_pairs, dtype=np.float64)
    if total_pairs == 0:
        return condensed

    # Group curve indices by length so each group can be stacked into one array.
    groups = {}
    for idx, c in enumerate(curves):
        groups.setdefault(len(c), []).append(idx)
    stacks = {L: (np.array(idxs, dtype=np.int64), np.stack([curves[i] for i in idxs]))
              for L, idxs in groups.items()}
    lengths = sorted(stacks)

    done = 0
    for a_pos, La in enumerate(lengths):
        idx_a, arr_a = stacks[La]
        for Lb in lengths[a_pos:]:
            idx_b, arr_b = stacks[Lb]
            if La == Lb:
                ia, ib = np.triu_indices(len(idx_a), k=1)
            else:
                ia, ib = np.meshgrid(np.arange(len(idx_a)), np.arange(len(idx_b)), indexing='ij')
                ia, ib = ia.ravel(), ib.ravel()

            for start in range(0, len(ia), max_pairs_per_batch):
                sa = ia[start:start + max_pairs_per_batch]
                sb = ib[start:start + max_pairs_per_batch]
                P = arr_a[sa]  # (k, La, D)
                Q = arr_b[sb]  # (k, Lb, D)
                # Point-to-point distances for every pair: (k, La, Lb)
                c = np.sqrt(((P[:, :, None, :] - Q[:, None, :, :]) ** 2).sum(axis=-1))

                ca = np.empty_like(c)
                ca[:, 0, 0] = c[:, 0, 0]
                for p in range(1, La):
                    ca[:, p, 0] = np.maximum(ca[:, p - 1, 0], c[:, p, 0])
                for q in range(1, Lb):
                    ca[:, 0, q] = np.maximum(ca[:, 0, q - 1], c[:, 0, q])
                for p in range(1, La):
                    for q in range(1, Lb):
                        best_prev = np.minimum(np.minimum(ca[:, p - 1, q], ca[:, p, q - 1]), ca[:, p - 1, q - 1])
                        ca[:, p, q] = np.maximum(best_prev, c[:, p, q])

                gi = idx_a[sa]
                gj = idx_b[sb]
                lo = np.minimum(gi, gj)
                hi = np.maximum(gi, gj)
                condensed[_condensed_index(n_curves, lo, hi)] = ca[:, La - 1, Lb - 1]

                done += len(sa)
                if progress_callback is not None:
                    progress_callback(done, total_pairs)

    return condensed


def condensed_submatrix(condensed, n, indices):
    """
    Square distance sub-matrix for `indices`, read directly from a condensed
    vector, without materialising the full n x n matrix.
    """
    indices = np.asarray(indices, dtype=np.int64)
    k = len(indices)
    sub = np.zeros((k, k), dtype=np.float64)
    if k < 2:
        return sub
    ia, ib = np.triu_indices(k, k=1)
    gi, gj = indices[ia], indices[ib]
    lo, hi = np.minimum(gi, gj), np.maximum(gi, gj)
    vals = condensed[_condensed_index(n, lo, hi)]
    sub[ia, ib] = vals
    sub[ib, ia] = vals
    return sub
