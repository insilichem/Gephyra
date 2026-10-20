import contextlib
import json
import os
import logging
import csv
import copy
import MDAnalysis as mda
import numpy as np

from .core import build_graph, compute_edge_probabilities, traverse_network
from .math_utils import pairwise_discrete_frechet, condensed_submatrix, normalize_hquality
from .scene import minimum_image_chain
from .results import load_clusters  # noqa: F401  (re-exported for convenience)

logger = logging.getLogger(__name__)


def path_hquality(path):
    """
    H-quality Z of one path record from a .jsonl file.

    Files written before 0.3.0 stored Z under the misleading key
    'probability'; it is still read so that old outputs can be clustered.
    """
    if 'hquality' in path:
        return float(path['hquality'])
    return float(path['probability'])

def _write_clusters(output_file, clusters, source_metadata, cluster_params):
    """
    Writes the cluster summary as {"type", "metadata", "clusters"}.

    `metadata` carries the analysed system (topology, trajectory, selections)
    from the .jsonl, so `visualize` can rebuild the molecular scene from the
    cluster file alone. Readers should use `results.load_clusters`, which also accepts
    the plain list written by Gephyra < 0.4.
    """
    metadata = {k: v for k, v in source_metadata.items() if k != "type"}
    metadata["clustering"] = cluster_params
    with open(output_file, 'w') as f:
        json.dump({"type": "gephyra_clusters", "metadata": metadata, "clusters": clusters}, f, indent=2)


def compute_persistence(frame_indices_sorted, total_frames, stride):
    """
    Calculates the mean and maximum continuous run-lengths of frames.
    """
    if not frame_indices_sorted:
        return 0.0, 0

    run_lengths = []
    current_run = 1

    for i in range(1, len(frame_indices_sorted)):
        if frame_indices_sorted[i] - frame_indices_sorted[i-1] == stride:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1

    run_lengths.append(current_run)

    mean_persistence = float(np.mean(run_lengths)) if run_lengths else 0.0
    max_persistence = int(np.max(run_lengths)) if run_lengths else 0

    return mean_persistence, max_persistence

def cluster_pathways(data_file, threshold=6.0, coarse_threshold=None, coarse_trigger=1000, max_paths=60000, output_file="clustered_pathways.json",
                     progress_callback=None):
    """
    Reads JSONL trajectory data and performs temporal clustering to identify
    collective pathways using a Hybrid 9D-Vector representation.

    Every unique path topology in the input is carried into the spatial
    clustering step; there is no pre-filter on frame count. The only frequency
    criterion is applied after clustering, where a merged cluster present in
    fewer than two frames is discarded.

    Args:
        coarse_trigger (int): Dataset size threshold. If the number of paths is less than or equal to this value, the 9D coarse screening is bypassed to maximize topological fidelity.
        progress_callback: optional callable(done, total) reporting progress of
            the Fréchet distance matrix. It may raise to abort the run.
    """
    logger.info("Starting temporal clustering of pathways...")
    import scipy.spatial.distance as ssd
    from scipy.cluster.hierarchy import linkage, fcluster

    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")

    unique_paths = {}
    source_metadata = {}
    cluster_params = {"data_file": os.path.abspath(data_file), "threshold": threshold,
                      "coarse_threshold": coarse_threshold, "coarse_trigger": coarse_trigger,
                      "max_paths": max_paths}
    total_frames = 0
    stride = 1
    frame_set = set()

    # Read paths into memory for clustering
    with open(data_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            if obj.get('type') == 'metadata':
                source_metadata = obj
                total_frames = obj.get('n_frames_analyzed', 0)
                stride = obj.get('parameters', {}).get('stride', 1)
            elif obj.get('type') == 'frame':
                frame_idx = obj['frame_idx']
                frame_set.add(frame_idx)
                for p in obj['paths']:
                    nodes_tuple = tuple(p['nodes'])
                    coords = np.array(p['coords'])

                    if nodes_tuple not in unique_paths:
                        unique_paths[nodes_tuple] = {
                            'frames': set(),
                            'hquality_sum': 0.0,
                            'hquality_norm_sum': 0.0,
                            'count': 0,
                            'coords': coords.tolist(), # Keep one full sequence for medoid
                            '9d_sum': np.zeros(9)
                        }

                    # Running sums instead of per-occurrence lists: only the
                    # means are ever used, and a long trajectory can contain
                    # the same path in tens of thousands of frames.
                    entry = unique_paths[nodes_tuple]
                    entry['frames'].add(frame_idx)
                    z = path_hquality(p)
                    entry['hquality_sum'] += z
                    entry['hquality_norm_sum'] += normalize_hquality(z)
                    entry['count'] += 1
                    entry['9d_sum'] += np.concatenate([coords[0], coords[len(coords)//2], coords[-1]])

    if total_frames == 0:
        total_frames = len(frame_set)

    if not unique_paths:
        logger.info("No paths found to cluster.")
        _write_clusters(output_file, [], source_metadata, cluster_params)
        return {'output_file': os.path.abspath(output_file), 'n_clusters': 0}

    # Build candidate list — all unique paths proceed to spatial clustering.
    filtered_paths = []
    for nodes, data in unique_paths.items():
        avg_hquality = data['hquality_sum'] / data['count']
        avg_hquality_norm = data['hquality_norm_sum'] / data['count']
        avg_9d = data['9d_sum'] / data['count']
        filtered_paths.append({
            'nodes': nodes,
            'frames': data['frames'],
            'occupancy': len(data['frames']) / total_frames if total_frames > 0 else 0.0,
            'avg_hquality': avg_hquality,
            'avg_hquality_norm': avg_hquality_norm,
            'coords': data['coords'],
            'avg_9d': avg_9d
        })

    n_filtered = len(filtered_paths)
    logger.info(f"Loaded {n_filtered} unique path topologies for spatial clustering.")

    if n_filtered == 0:
        logger.info("No paths available for clustering.")
        _write_clusters(output_file, [], source_metadata, cluster_params)
        return {'output_file': os.path.abspath(output_file), 'n_clusters': 0}

    # Memory Safety Cap
    if n_filtered > max_paths:
        logger.warning(f"Number of paths ({n_filtered}) exceeds maximum cap ({max_paths}). "
                       f"Truncating to top {max_paths} paths to prevent memory crash. "
                       "Tie-breaking identical occupancies using average H-quality.")
        # Sort primarily by occupancy, secondarily by H-quality
        filtered_paths.sort(key=lambda x: (x['occupancy'], x['avg_hquality']), reverse=True)
        filtered_paths = filtered_paths[:max_paths]
        n_filtered = len(filtered_paths)

    if n_filtered > coarse_trigger:
        # 9D Coarse Screening Pass
        logger.info(f"Dataset large ({n_filtered} paths > trigger={coarse_trigger}). Performing coarse 9D screening pass...")

        if coarse_threshold is None:
            coarse_threshold = threshold / np.sqrt(3)

        all_9d = np.array([p['avg_9d'] for p in filtered_paths])
        dist_matrix_9d = ssd.pdist(all_9d, metric='euclidean')
        Z_9d = linkage(dist_matrix_9d, method='average')
        labels_9d = fcluster(Z_9d, t=coarse_threshold, criterion='distance')

        unique_labels_9d = set(labels_9d)
        logger.info(f"Coarse 9D screening reduced {n_filtered} paths to {len(unique_labels_9d)} spatial channels.")

        coarse_screened_paths = []
        for label in unique_labels_9d:
            cluster_indices = np.where(labels_9d == label)[0]
            cluster_paths = [filtered_paths[i] for i in cluster_indices]

            # Select representative
            cluster_features = np.array([p['avg_9d'] for p in cluster_paths])
            mean_9d = np.mean(cluster_features, axis=0)
            dists_to_mean = np.linalg.norm(cluster_features - mean_9d, axis=1)
            medoid_local_idx = np.argmin(dists_to_mean)

            rep_path = copy.deepcopy(cluster_paths[medoid_local_idx])

            # Merge frames and H-quality from all paths in this coarse cluster
            merged_frames = set()
            for p in cluster_paths:
                merged_frames.update(p['frames'])

            rep_path['frames'] = merged_frames
            rep_path['occupancy'] = len(merged_frames) / total_frames if total_frames > 0 else 0.0
            rep_path['avg_hquality'] = np.mean([p['avg_hquality'] for p in cluster_paths])
            rep_path['avg_hquality_norm'] = np.mean([p['avg_hquality_norm'] for p in cluster_paths])

            coarse_screened_paths.append(rep_path)

        filtered_paths = coarse_screened_paths
        n_filtered = len(filtered_paths)

    elif n_filtered > 1:
        logger.info(
            f"Dataset small ({n_filtered} paths ≤ trigger={coarse_trigger}). "
            "Bypassing 9D coarse filter — proceeding directly to Fréchet distance "
            "for maximum topological fidelity."
        )

    if n_filtered == 1:
        logger.info("Only 1 unique path remaining. Bypassing clustering and exporting directly.")

        frames_val = filtered_paths[0]['frames']

        # Same stability rule as the multi-path branch below: a channel seen in
        # fewer than two frames is thermal noise.
        if len(frames_val) < 2:
            logger.info("The only remaining path occurs in fewer than 2 frames; discarded as thermal noise.")
            _write_clusters(output_file, [], source_metadata, cluster_params)
            return {'output_file': os.path.abspath(output_file), 'n_clusters': 0}

        medoid_frame = int(min(frames_val))
        mean_pers, max_pers = compute_persistence(sorted(frames_val), total_frames, stride)

        clusters_data = [{
            "cluster_id": 1,
            "size": 1,
            "occupancy": float(filtered_paths[0]['occupancy']),
            "avg_hquality": float(filtered_paths[0]['avg_hquality']),
            "avg_hquality_norm": float(filtered_paths[0]['avg_hquality_norm']),
            "medoid_frame": medoid_frame,
            "mean_persistence_frames": mean_pers,
            "max_persistence_frames": max_pers,
            "medoid_nodes": [int(n) for n in filtered_paths[0]['nodes']],
            "medoid_coords": filtered_paths[0]['coords']
        }]
        _write_clusters(output_file, clusters_data, source_metadata, cluster_params)
        return {'output_file': os.path.abspath(output_file), 'n_clusters': 1}

    logger.info(f"Computing Fréchet distance matrix for {n_filtered} unique pathways...")

    n = n_filtered
    next_log = [0.10]

    def _log_progress(done, total):
        while done / total >= next_log[0] and next_log[0] <= 1.0:
            logger.info(f"Fréchet matrix calculation: {int(round(next_log[0] * 100))}% complete...")
            next_log[0] += 0.10
        if progress_callback is not None:
            progress_callback(done, total)

    # Discrete Fréchet distance (Eiter-Mannila), vectorised over pairs of equal
    # length combination. Numerically identical to similaritymeasures.frechet_dist.
    dist_matrix = pairwise_discrete_frechet(
        [p['coords'] for p in filtered_paths],
        progress_callback=_log_progress,
    )

    logger.info("Performing hierarchical average-link clustering...")
    Z = linkage(dist_matrix, method='average')
    labels = fcluster(Z, t=threshold, criterion='distance')

    unique_labels = set(labels)
    n_clusters = len(unique_labels)
    logger.info(f"Identified {n_clusters} unique collective pathways.")

    clusters_data = []

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_paths = [filtered_paths[i] for i in cluster_indices]

        # Calculate length variance warning
        path_lengths = [len(p['nodes']) for p in cluster_paths]
        if np.std(path_lengths) > 2.0:
            logger.warning(
                f"Cluster {label} exhibits high length variance (std > 2.0). "
                "Fréchet distances for this cluster may be dominated by length discrepancies rather than topological shape differences."
            )

        # Cluster occupancy is computed from the fully merged frame set, i.e.
        # the union of the frames of every path in the cluster. A cluster seen
        # in fewer than two frames is discarded as thermal noise.
        #
        # The check is deliberately applied here, after merging, and nowhere
        # else: no path is filtered on frame count before clustering. A single
        # atom-index chain appearing in one frame may still be one sampling of a
        # channel that is occupied throughout the trajectory by exchanging
        # waters, and those permutations only come back together once the paths
        # have been grouped geometrically. Filtering earlier would delete them.
        cluster_frames = set()
        for p in cluster_paths:
            cluster_frames.update(p['frames'])

        if len(cluster_frames) < 2:
            logger.debug(
                f"Cluster {label} discarded: merged occupancy "
                f"({len(cluster_frames)} frames) < 2 (minimum stability threshold)."
            )
            continue

        occupancy = len(cluster_frames) / total_frames if total_frames > 0 else 0.0

        # Average H-quality across all paths in the cluster
        avg_hquality = np.mean([p['avg_hquality'] for p in cluster_paths])
        avg_hquality_norm = np.mean([p['avg_hquality_norm'] for p in cluster_paths])

        # Medoid calculation
        # Extract the sub-matrix of distances between elements in this cluster
        # (read straight from the condensed vector; a full n x n square copy
        # would need 8*n^2 bytes, i.e. ~29 GB at the default max_paths cap)
        cluster_dist_submatrix = condensed_submatrix(dist_matrix, n, cluster_indices)

        # Sum distances for each path to all other paths in the cluster
        sum_distances = np.sum(cluster_dist_submatrix, axis=1)

        # The medoid is the path with the minimum sum of distances
        medoid_local_idx = np.argmin(sum_distances)
        medoid_path = cluster_paths[medoid_local_idx]
        medoid_coords = medoid_path['coords']

        medoid_frame = int(min(medoid_path['frames']))
        mean_pers, max_pers = compute_persistence(sorted(cluster_frames), total_frames, stride)

        clusters_data.append({
            "cluster_id": int(label),
            "size": len(cluster_indices),
            "occupancy": float(occupancy),
            "avg_hquality": float(avg_hquality),
            "avg_hquality_norm": float(avg_hquality_norm),
            "medoid_frame": medoid_frame,
            "mean_persistence_frames": mean_pers,
            "max_persistence_frames": max_pers,
            # Atom indices of the medoid path at medoid_frame (the frame its
            # coordinates come from); lets `visualize` rebuild it as molecules.
            "medoid_nodes": [int(n) for n in medoid_path['nodes']],
            "medoid_coords": medoid_coords
        })

    # Sort by occupancy descending
    clusters_data.sort(key=lambda x: x['occupancy'], reverse=True)

    # Reassign cluster_ids sequentially (1 = highest occupancy)
    for i, cluster in enumerate(clusters_data):
        cluster["cluster_id"] = i + 1

    _write_clusters(output_file, clusters_data, source_metadata, cluster_params)

    logger.info(f"Clustering complete. Results saved to {output_file}")
    return {'output_file': os.path.abspath(output_file), 'n_clusters': len(clusters_data)}


def validate_calculation_parameters(stride=1, max_depth=10, min_depth=1, cooperativity=0.92, coarse_cutoff=4.5):
    """
    Raises ValueError with an actionable message for out-of-range parameters,
    before any trajectory is loaded.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}.")
    if min_depth < 1:
        raise ValueError(f"min_depth must be >= 1, got {min_depth}.")
    if min_depth > max_depth:
        raise ValueError(f"min_depth ({min_depth}) cannot exceed max_depth ({max_depth}); no path could ever be written.")
    if not 0.0 < cooperativity <= 1.0:
        raise ValueError(f"cooperativity must be in (0, 1], got {cooperativity}.")
    if coarse_cutoff <= 0:
        raise ValueError(f"coarse_cutoff must be positive, got {coarse_cutoff}.")


def sanitize_csv_field(field_value):
    """
    Sanitizes a field value for CSV export to prevent formula injection.
    If the value starts with =, +, -, or @, it prepends a single quote.
    """
    field_str = str(field_value)
    if field_str.startswith(('=', '+', '-', '@')):
        return f"'{field_str}"
    return field_str

def run_analysis(topo_file, traj_file, root_sel, water_sel="resname SOL or resname WAT or resname HOH",
                 stride=1, max_depth=10, min_depth=1, prob_threshold=None, cooperativity=0.92, coarse_cutoff=4.5,
                 output_file="results.jsonl", csv_file=None, cluster=False, cluster_threshold=6.0,
                 cluster_output="clustered_pathways.json", progress_callback=None):
    """
    Iterates over the trajectory and aggregates network pathways.
    Streams output as JSON Lines (JSONL) to prevent memory exhaustion,
    and optionally writes to CSV.

    Args:
        prob_threshold: deprecated and ignored; see traverse_network.
        cluster: deprecated. Runs cluster_pathways inline once the trajectory
            has been processed. Prefer the dedicated 'cluster' sub-command,
            which can be re-run with different thresholds without repeating the
            MD analysis.
        cluster_threshold: Frechet threshold for the inline clustering. Matches
            the default of the 'cluster' sub-command.
        cluster_output: output path for the inline clustering. Matches the
            default of the 'cluster' sub-command.
        progress_callback: optional callable(frames_done, frames_total), called
            after every analysed frame. It may raise to abort the run; output
            files are closed and keep the frames written so far.

    Raises:
        ValueError: if a numerical parameter is out of range or the root
            selection matches no atoms.
    """
    validate_calculation_parameters(stride=stride, max_depth=max_depth, min_depth=min_depth,
                                    cooperativity=cooperativity, coarse_cutoff=coarse_cutoff)
    if not traj_file:
        # A topology alone is not enough: many formats (.prmtop, .psf, .tpr
        # without coordinates) carry no positions at all, which used to fail
        # deep inside MDAnalysis with an unhelpful error.
        raise ValueError("A trajectory file is required (e.g. .xtc, .dcd, .nc). To analyse a single "
                         "structure, pass a coordinate file such as a .pdb as the trajectory.")
    for label, path in (("Topology", topo_file), ("Trajectory", traj_file)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    logger.info(f"Loading topology: {topo_file}")
    logger.info(f"Loading trajectory: {traj_file}")
    u = mda.Universe(topo_file, traj_file)

    n_frames = len(u.trajectory)

    if n_frames > 1000 and stride == 1:
        logger.warning(
            "Trajectory has >1000 frames. This might consume significant time. "
            "Consider using a larger --stride parameter or clustering your trajectory."
        )

    frames_to_process = range(0, n_frames, stride)
    total_processed = len(frames_to_process)

    logger.info(f"Processing {total_processed} frames out of {n_frames} with stride {stride}.")

    # Pre-compute selections
    water_atoms = u.select_atoms(water_sel)
    root_atoms = u.select_atoms(root_sel)

    if len(root_atoms) == 0:
        raise ValueError(f"Root selection '{root_sel}' returned no atoms. Check your resname/resid.")

    if len(water_atoms) == 0:
        logger.warning(
            f"Water selection '{water_sel}' returned no atoms; no paths can be found. "
            "Check the solvent residue names in your topology (see --water)."
        )

    path_frequency = {}

    with contextlib.ExitStack() as stack:
        out_f = stack.enter_context(open(output_file, 'w'))
        csv_writer = None
        if csv_file:
            csv_f = stack.enter_context(open(csv_file, 'w', newline=''))
            csv_writer = csv.writer(csv_f)
            csv_writer.writerow(["Frame", "Root_Residue", "Path_Atom_Indices", "Path_Length", "HQuality_Z", "HQuality_Norm", "Average_OO_Distance"])

        found_large_path, total_paths, total_discarded, total_length_sum, total_hquality_sum = _process_trajectory(
            u, water_atoms, root_atoms, out_f, csv_writer, path_frequency,
            topo_file=os.path.abspath(topo_file), traj_file=os.path.abspath(traj_file),
            root_sel=root_sel, water_sel=water_sel, stride=stride, max_depth=max_depth,
            min_depth=min_depth, prob_threshold=prob_threshold, cooperativity=cooperativity,
            coarse_cutoff=coarse_cutoff, total_processed=total_processed,
            progress_callback=progress_callback,
        )

    if csv_file:
        logger.info(f"CSV saved to {csv_file}")

    # Check threshold notification
    if max_depth >= 8 and not found_large_path:
        notice_msg = "Notice: No water-bridges larger than 8 layers were detected in this analysis."
        logger.info(notice_msg)
        with open(output_file, 'a') as f:
            f.write(json.dumps({"type": "notice", "message": notice_msg}) + '\n')

    avg_length = float(total_length_sum) / total_paths if total_paths > 0 else 0.0
    avg_hquality = float(total_hquality_sum) / total_paths if total_paths > 0 else 0.0

    logger.info("=== Analysis Complete ===")
    logger.info(f"Total paths written across analyzed frames: {total_paths}")
    if total_discarded:
        logger.info(f"Paths discarded by --min_depth {min_depth}: {total_discarded}")
    logger.info(f"Average path length (depth): {avg_length:.2f}")
    logger.info(f"Average H-quality (Z): {avg_hquality:.4f} "
                f"(normalized: {normalize_hquality(avg_hquality):.4f})")
    logger.info(f"Results saved to {output_file}")

    if cluster:
        import warnings
        warnings.warn(
            "Running clustering inline from 'calculate' is deprecated. Use the "
            "'gephyra cluster' sub-command instead, which takes the same "
            "defaults and can be re-run without repeating the MD analysis.",
            DeprecationWarning, stacklevel=2
        )
        cluster_pathways(
            data_file=output_file,
            threshold=cluster_threshold,
            output_file=cluster_output,
        )

    return {
        "output_file": os.path.abspath(output_file),
        "csv_file": os.path.abspath(csv_file) if csv_file else None,
        "n_frames": total_processed,
        "n_paths": total_paths,
        "avg_length": avg_length,
    }


def _process_trajectory(u, water_atoms, root_atoms, out_f, csv_writer, path_frequency, *,
                        root_sel, water_sel, stride, max_depth, min_depth, prob_threshold,
                        cooperativity, coarse_cutoff, total_processed, progress_callback=None,
                        topo_file=None, traj_file=None):
    """
    Frame loop of run_analysis: writes the metadata line, one line per frame
    and the trailing path_statistics line to `out_f`.

    Returns (found_large_path, total_paths, total_discarded, total_length_sum,
    total_hquality_sum). `total_paths` counts only the paths actually written, i.e.
    after the --min_depth filter, so averages derived from these totals refer
    to the same population as the output file.
    """
    total_length_sum = 0
    total_hquality_sum = 0
    total_paths = 0
    total_discarded = 0
    found_large_path = False

    # Metadata as the first line of JSONL
    metadata = {
        "type": "metadata",
        "n_frames_analyzed": total_processed,
        # Absolute paths of the analysed system, so `visualize` can rebuild the
        # molecular context (protein, ligand, waters) without asking again.
        "topo_file": topo_file,
        "traj_file": traj_file,
        "parameters": {
            "root_sel": root_sel,
            "water_sel": water_sel,
            "stride": stride,
            "max_depth": max_depth,
            "min_depth": min_depth,
            "cooperativity": cooperativity,
            "coarse_cutoff": coarse_cutoff
        }
    }
    out_f.write(json.dumps(metadata) + '\n')

    frames_done = 0
    for ts in u.trajectory[::stride]:
        frame_idx = ts.frame
        logger.info(f"Analyzing frame {frame_idx}...")

        # 1. Build coarse graph
        g, root_indices = build_graph(u, water_atoms, root_atoms, max_distance=coarse_cutoff, max_depth=max_depth)

        # 2. Compute fine probabilities
        g = compute_edge_probabilities(g, u)

        # 3. Traverse
        paths = traverse_network(g, root_indices, max_depth=max_depth, prob_threshold=prob_threshold, cooperativity=cooperativity)

        frame_paths_data = []
        for path_indices, hquality in paths:
            path_len = len(path_indices) - 1

            if path_len < min_depth:
                total_discarded += 1
                continue

            if path_len >= 8:
                found_large_path = True

            total_paths += 1
            total_length_sum += path_len
            total_hquality_sum += hquality
            hquality_norm = normalize_hquality(hquality)

            path_tuple = tuple(int(n) for n in path_indices)
            path_frequency.setdefault(path_tuple, set()).add(frame_idx)

            # Each atom is placed in the periodic image nearest to the previous
            # one, starting from the root, so stored paths are never torn
            # across the box (clustering and drawing both use these).
            coords = minimum_image_chain(u.atoms[list(path_indices)].positions, ts.dimensions).tolist()
            distances = [g[a][b]['dist'] for a, b in zip(path_indices[:-1], path_indices[1:])]

            avg_oo = float(np.mean(distances)) if distances else 0.0

            # Also store explicit atom ids for Chimera indexing
            atom_ids = [int(u.atoms[n].id) for n in path_indices]

            frame_paths_data.append({
                "nodes": [int(n) for n in path_indices],
                "atom_ids": atom_ids,
                "coords": coords,
                "hquality": float(hquality),
                "hquality_norm": hquality_norm,
                "length": path_len,
                "avg_oo_dist": avg_oo
            })

            if csv_writer:
                root_res = u.atoms[path_indices[0]].resname
                path_str = "-".join(str(n) for n in path_indices)

                # Sanitize to prevent CSV formula injection
                safe_root_res = sanitize_csv_field(root_res)
                safe_path_str = sanitize_csv_field(path_str)

                csv_writer.writerow([frame_idx, safe_root_res, safe_path_str, path_len, hquality, hquality_norm, avg_oo])

        # Write frame data immediately to release RAM
        out_f.write(json.dumps({"type": "frame", "frame_idx": frame_idx, "paths": frame_paths_data}) + '\n')
        frames_done += 1
        if progress_callback is not None:
            progress_callback(frames_done, total_processed)

    # Calculate path statistics
    top_paths = []
    for p_tuple, frames_set in path_frequency.items():
        frame_count = len(frames_set)
        occupancy = float(frame_count) / total_processed if total_processed > 0 else 0.0
        top_paths.append({
            "nodes": list(p_tuple),
            "frame_count": frame_count,
            "occupancy": occupancy
        })

    # Sort descending by frame count and select top 20
    top_paths.sort(key=lambda x: x["frame_count"], reverse=True)
    top_paths = top_paths[:20]

    path_stats_obj = {
        "type": "path_statistics",
        "total_frames_analyzed": total_processed,
        "top_paths": top_paths
    }
    out_f.write(json.dumps(path_stats_obj) + '\n')

    return found_large_path, total_paths, total_discarded, total_length_sum, total_hquality_sum
