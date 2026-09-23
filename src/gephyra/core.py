import re
import logging
import numpy as np
import networkx as nx
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import capped_distance, distance_array
from .math_utils import calculate_hbond_probability, switching_function

logger = logging.getLogger(__name__)

# Fallback name -> element map, used only when the topology does not carry an
# `element` attribute (see _get_element).
#
# Provenance: the O*, N* and S* entries are the CHARMM27 donor and acceptor atom
# names that MDAnalysis ships with WaterBridgeAnalysis
# (MDAnalysis.analysis.hydrogenbonds.wbridge_analysis, DEFAULT_DONORS /
# DEFAULT_ACCEPTORS for the 'CHARMM27' force field). O1, O2, OW1 and OXT are
# added by Gephyra: OXT for C-terminal carboxylates, and O1/O2/OW1 because they
# are common in ligand and non-standard solvent topologies that the CHARMM27
# lists do not cover.
_NAME_TO_ELEMENT = {
    'OW': 'O', 'O1': 'O', 'O2': 'O', 'OD1': 'O', 'OD2': 'O', 'OE1': 'O', 'OE2': 'O', 'OG': 'O', 'OG1': 'O', 'OH': 'O',
    'OXT': 'O', 'OH2': 'O', 'OC1': 'O', 'OC2': 'O', 'OW1': 'O',
    'NZ': 'N', 'ND1': 'N', 'ND2': 'N', 'NE': 'N', 'NE1': 'N', 'NE2': 'N', 'NH1': 'N', 'NH2': 'N',
    'SG': 'S', 'SD': 'S'
}

_warned_united_atom = set()
_warned_no_bonds = set()

def _is_hydrogen(a):
    """
    Tiered fallback to correctly identify if an atom is hydrogen.
    1. Element (most reliable)
    2. Mass (force-field agnostic)
    3. Name/Type (last resort)
    """
    if getattr(a, 'element', '') and a.element.strip().upper() == 'H':
        return True

    try:
        if 0.5 < a.mass < 2.5:
            return True
    except Exception:
        pass

    return bool(re.search(r'(?i)^[0-9]*h', a.name)) or getattr(a, 'type', '') == 'H'

def _get_element(atom):
    """
    Robustly resolves the element of an atom, preventing misclassification.
    Priority 1: atom.element (MDAnalysis standard).
    Priority 2: Exact match in _NAME_TO_ELEMENT dictionary.
    Priority 3: Strip leading digits from atom.name and take the leading alphabetic substring.
    """
    # Elements accepted as hydrogen-bond donors or acceptors. Every element in
    # this set must have an explicit (r0, delta) entry in
    # compute_edge_probabilities; otherwise it silently inherits the O-O
    # parameters, which are wrong for the heavier elements.
    valid_elements = {"O", "N", "S", "F", "CL", "BR"}
    try:
        if atom.element:
            e = atom.element.strip().upper()
            if e in valid_elements:
                return e
    except AttributeError:
        pass

    # Fallback 1: Lookup exact names for common topologies (e.g., OW -> O, NZ -> N)
    atom_name_upper = atom.name.strip().upper()
    if atom_name_upper in _NAME_TO_ELEMENT:
        return _NAME_TO_ELEMENT[atom_name_upper]

    # Fallback 2: Strip leading digits and extract the leading alphabetic substring
    match = re.search(r'^[0-9]*([A-Za-z]+)', atom.name)
    if match:
        e = match.group(1).upper()
        if e in valid_elements:
            return e

    return "UNKNOWN"

def build_graph(u, water_atoms, root_atoms, max_distance=4.5, max_depth=5):
    """
    Builds a NetworkX graph representing potential hydrogen bonds
    using a shell-based iterative expansion to avoid global N^2 distance matrices.
    """
    g = nx.Graph()

    for a in root_atoms:
        g.add_node(a.index, resname=a.resname, name=a.name, pos=a.position)

    current_shell_indices = list(root_atoms.indices)
    visited_indices = set(current_shell_indices)

    box = u.dimensions

    for depth in range(max_depth):
        if not current_shell_indices:
            break

        current_shell = u.atoms[current_shell_indices]

        # capped_distance between current shell and all waters
        pairs, distances = capped_distance(
            current_shell.positions,
            water_atoms.positions,
            max_cutoff=max_distance,
            box=box,
            return_distances=True
        )

        next_shell_indices = set()

        for (idx_current, idx_water), dist in zip(pairs, distances):
            u_node = current_shell.indices[idx_current]
            v_node = water_atoms.indices[idx_water]

            if v_node not in g:
                a2 = u.atoms[v_node]
                g.add_node(v_node, resname=a2.resname, name=a2.name, pos=a2.position)

            g.add_edge(u_node, v_node, dist=dist)

            if v_node not in visited_indices:
                next_shell_indices.add(v_node)

        # Connect nodes within the new shell
        if next_shell_indices:
            next_shell_list = list(next_shell_indices)
            next_shell_ag = u.atoms[next_shell_list]
            intra_pairs, intra_dist = capped_distance(
                next_shell_ag.positions,
                next_shell_ag.positions,
                max_cutoff=max_distance,
                box=box,
                return_distances=True
            )
            for (i1, i2), d in zip(intra_pairs, intra_dist):
                if i1 < i2:
                    n1 = next_shell_ag.indices[i1]
                    n2 = next_shell_ag.indices[i2]
                    g.add_edge(n1, n2, dist=d)

        visited_indices.update(next_shell_indices)
        current_shell_indices = list(next_shell_indices)

    return g, root_atoms.indices

def compute_edge_probabilities(g, u):
    """
    Iterates over edges in the graph and computes the continuous probability
    by dynamically finding attached hydrogens.
    """
    edges_to_remove = []
    ua_atom_indices = set()
    h_cache = {}

    def get_hydrogens(atom):
        if atom.index in h_cache:
            return h_cache[atom.index]

        try:
            atom_bonds = atom.bonds
        except (NoDataError, AttributeError):
            # The topology carries no bond records at all (a bare PDB without
            # CONECT lines, a .gro, an .xyz ...). We cannot separate explicit
            # hydrogens from heavy neighbours, and we cannot place virtual ones
            # either, so return an empty list. The caller then takes the
            # distance-only branch below (ignore_angle=True), exactly as it does
            # for a united-atom topology.
            if not _warned_no_bonds:
                _warned_no_bonds.add(True)
                logger.warning(
                    "Topology contains no bond information, so hydrogen positions "
                    "cannot be resolved. Falling back to a distance-only "
                    "hydrogen-bond criterion for every edge: the angular term is "
                    "ignored and scores will be systematically higher. Supply a "
                    "topology with connectivity (e.g. a .tpr, .psf, or a PDB with "
                    "CONECT records) for the full criterion."
                )
            h_cache[atom.index] = []
            return []

        explicit_hs = []
        bonded_heavy_atoms = []
        for bond in atom_bonds:
            neighbor = bond.atoms[1] if bond.atoms[0].index == atom.index else bond.atoms[0]
            if _is_hydrogen(neighbor):
                explicit_hs.append(neighbor)
            else:
                bonded_heavy_atoms.append(neighbor)

        if explicit_hs:
            h_positions = [h.position for h in explicit_hs]
            h_cache[atom.index] = h_positions
            return h_positions

        ua_atom_indices.add(atom.index)

        if not bonded_heavy_atoms:
            h_cache[atom.index] = []
            return []

        neighbor_pos = np.array([neighbor.position for neighbor in bonded_heavy_atoms])

        if len(bonded_heavy_atoms) == 1:
            v1 = atom.position - neighbor_pos[0]
            v1 /= np.linalg.norm(v1)
            arb = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            perp = np.cross(v1, arb)
            perp /= np.linalg.norm(perp)
            cos_120, sin_120 = -0.5, 0.866
            lp1 = atom.position + (v1 * cos_120 + perp * sin_120) * 1.0
            lp2 = atom.position + (v1 * cos_120 - perp * sin_120) * 1.0
            h_cache[atom.index] = [lp1, lp2]
            return [lp1, lp2]

        elif len(bonded_heavy_atoms) == 2:
            v1 = neighbor_pos[0] - atom.position
            v2 = neighbor_pos[1] - atom.position
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            n = np.cross(v1, v2)
            n /= np.linalg.norm(n)
            bisector = v1 + v2
            bisector /= np.linalg.norm(bisector)
            cos_tilt, sin_tilt = -0.577, 0.816
            lp1 = atom.position + (-bisector * cos_tilt + n * sin_tilt) * 1.0
            lp2 = atom.position + (-bisector * cos_tilt - n * sin_tilt) * 1.0
            h_cache[atom.index] = [lp1, lp2]
            return [lp1, lp2]

        else:
            h_cache[atom.index] = []
            return []

    for u_node, v_node, data in g.edges(data=True):
        a1 = u.atoms[u_node]
        a2 = u.atoms[v_node]

        e1 = _get_element(a1)
        e2 = _get_element(a2)

        if e1 == "UNKNOWN" or e2 == "UNKNOWN":
            edges_to_remove.append((u_node, v_node))
            continue

        mod_rOO = data['dist']

        hs1 = get_hydrogens(a1)
        hs2 = get_hydrogens(a2)
        all_hs = hs1 + hs2

        # Per-pair reference distance r0 (A) and switching width delta (A).
        #
        # These are Gephyra's own empirical parameters. They are NOT inherited
        # from MDAnalysis: WaterBridgeAnalysis applies a single 3.0 A
        # donor-acceptor distance cutoff to every element pair. Here r0 sets
        # where the distance switching function starts to decay and delta sets
        # how far beyond r0 it reaches zero, so an edge is scored down smoothly
        # between r0 and r0 + delta rather than being accepted or rejected at a
        # hard cutoff. Values follow the typical heavy-atom separation of each
        # pair, widening with the van der Waals radius of the heavier partner.
        #
        # Every element in _get_element's `valid_elements` needs a branch here.
        # The halogen values are the roughest of the set: organic F in
        # particular is a weak acceptor, and treating C-F as a full H-bond
        # acceptor will over-report bridges. If that is not wanted for your
        # system, drop F/CL/BR from `valid_elements` instead, which makes those
        # edges UNKNOWN and removes them.
        if 'BR' in (e1, e2):
            r0_oo_fixed = 3.4
            r0_threshold_fixed = 0.8
        elif 'CL' in (e1, e2):
            r0_oo_fixed = 3.3
            r0_threshold_fixed = 0.8
        elif 'S' in (e1, e2):
            r0_oo_fixed = 3.3
            r0_threshold_fixed = 0.8
        elif 'F' in (e1, e2):
            r0_oo_fixed = 2.9
            r0_threshold_fixed = 0.5
        elif (e1, e2) in (('N', 'N'),):
            r0_oo_fixed = 3.0
            r0_threshold_fixed = 0.6
        elif (e1, e2) in (('N', 'O'), ('O', 'N')):
            r0_oo_fixed = 2.9
            r0_threshold_fixed = 0.55
        else: # O-O and defaults
            r0_oo_fixed = 2.80
            r0_threshold_fixed = 0.45

        if not hs1 or not hs2:
            # United-atom fallback: either side is missing hydrogens, ignore angle
            best_prob = calculate_hbond_probability(
                mod_rOO, None, None,
                r0_oo=r0_oo_fixed,
                r0_threshold=r0_threshold_fixed,
                ignore_angle=True
            )
        else:
            p_hs = np.array(all_hs)
            d1_array = distance_array(np.array([a1.position]), p_hs, box=u.dimensions)[0]
            d2_array = distance_array(np.array([a2.position]), p_hs, box=u.dimensions)[0]

            best_prob = 0.0

            for idx, h_pos in enumerate(p_hs):
                dist_DH = min(d1_array[idx], d2_array[idx])
                dist_HA = max(d1_array[idx], d2_array[idx])

                p_base = calculate_hbond_probability(
                    mod_rOO, dist_DH, dist_HA,
                    r0_oo=r0_oo_fixed,
                    r0_threshold=r0_threshold_fixed
                )
                # Two further empirical switching terms, also Gephyra's own and
                # not taken from MDAnalysis:
                #   2.5 A - the H...acceptor separation at which the interaction
                #           is taken to have vanished. Chosen as the outer edge
                #           of the H...O distribution for a water-water bond
                #           (which peaks near 1.8 A), so it damps long, poorly
                #           aligned contacts that the heavy-atom term alone
                #           would still accept.
                #   1.1 A - the covalent D-H bond length ceiling, used to confirm
                #           that the hydrogen really belongs to one of the two
                #           heavy atoms rather than to a third molecule that
                #           happens to lie between them. An X-H bond for X in
                #           {O, N, S} is ~0.96-1.34 A, so this is deliberately
                #           tight and is skipped entirely for virtual hydrogens.
                p_ha = switching_function(dist_HA, threshold=2.5, power_num=6, power_den=12)
                is_virtual_h = (a1.index in ua_atom_indices or a2.index in ua_atom_indices)
                if is_virtual_h:
                    # Virtual hydrogens were placed at exactly 1.0 A by
                    # get_hydrogens, so the covalent test carries no information.
                    p_covalent = 1.0
                else:
                    p_covalent = switching_function(dist_DH, threshold=1.1, power_num=6, power_den=12)

                p_i = p_base * p_ha * p_covalent
                best_prob = 1.0 - (1.0 - best_prob) * (1.0 - p_i)

        if best_prob <= 0:
            edges_to_remove.append((u_node, v_node))
        else:
            score = -np.log(best_prob) if best_prob > 0 else float('inf')
            g[u_node][v_node]['prob'] = best_prob
            g[u_node][v_node]['weight'] = score

    g.remove_edges_from(edges_to_remove)

    return g

def traverse_network(g, root_indices, max_depth=5, prob_threshold=None, cooperativity=0.92):
    """
    Enumerates self-avoiding paths from the root atoms outwards and groups them
    by endpoint.

    Each edge carries weight w = -log(p), so the weight of a path is the
    negative log of the product of its bond probabilities. The k-th edge of a
    path (k = 1 for the first edge leaving the root) is scaled by

        cooperativity ** (k - 1)

    before being added, which discounts each successive bond relative to the one
    before it. This models hydrogen-bond cooperativity: polarisation propagates
    along a water wire, so a bond that is already embedded in a chain is
    stronger, i.e. costs less, than the same geometry in isolation. The
    discount is geometric, so the penalty contributed by deep bonds decays and
    long chains are not dismissed purely for being long.

    cooperativity=1.0 disables the effect and recovers the plain product of edge
    probabilities. Values below 1.0 favour longer chains; the default of 0.92 is
    an empirical choice, not a measured constant.

    Args:
        g: graph with 'weight' on every edge, as produced by
            compute_edge_probabilities.
        root_indices: atom indices to start from.
        max_depth: maximum number of waters in a path.
        prob_threshold: deprecated and ignored. Kept only so that existing
            callers do not break; passing anything other than None raises a
            DeprecationWarning.
        cooperativity: per-depth discount factor described above.

    Returns:
        A list of (path, Z) tuples, one per endpoint reached. `path` is the
        lowest-weight route to that endpoint; Z is sum(exp(-w)) over every route
        to it, i.e. the unbounded quantity the README calls PHquality, not a
        probability in [0, 1].
    """
    import heapq
    import itertools
    from collections import defaultdict

    pq = []
    counter = itertools.count()

    if prob_threshold is not None:
        import warnings
        warnings.warn("prob_threshold has no effect and will be removed in a future version. "
                      "Path termination is controlled solely by max_depth.",
                      DeprecationWarning, stacklevel=2)

    endpoint_groups = defaultdict(list)

    for root in root_indices:
        if root in g:
            heapq.heappush(pq, (0.0, next(counter), root, 0, [root]))

    while pq:
        curr_weight, _, u_node, depth, path = heapq.heappop(pq)

        # No memoisation here by design. A (node, path) key can never repeat,
        # because every path is pushed exactly once and already ends at its own
        # node, so keying on it prunes nothing while retaining every path
        # visited so far. Keying on the node alone would be wrong: this
        # enumerates all routes to an endpoint in order to accumulate Z, not
        # just the best one. A correct optimisation would need a dominance test
        # over (endpoint, remaining depth), which is left undone; the cost of
        # the full enumeration is what makes deep --max_depth values expensive.
        if len(path) > 1:
            endpoint_groups[u_node].append((curr_weight, path))

        if depth >= max_depth:
            continue

        for v_node in g.neighbors(u_node):
            if v_node in path:
                continue

            edge_weight = g[u_node][v_node]['weight']
            next_weight = curr_weight + edge_weight * (cooperativity ** depth)

            next_path = path + [v_node]
            heapq.heappush(pq, (next_weight, next(counter), v_node, depth + 1, next_path))

    final_results = []
    for endpoint, paths_data in endpoint_groups.items():
        z_total = sum(np.exp(-w) for w, p in paths_data)
        best_path = min(paths_data, key=lambda x: x[0])[1]
        final_results.append((best_path, float(z_total)))

    return final_results
