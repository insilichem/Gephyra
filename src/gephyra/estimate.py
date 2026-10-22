"""
Runtime estimate for `calculate`, made before the analysis starts.

The cost of a frame depends far more on the system than on its size alone:
how many waters lie within reach of the root, how branched their hydrogen-bond
network is, and above all Max depth, because every extra water layer brings a
new shell of waters into the network that has to be built and scored. A formula in atoms and frames alone would
therefore be unreliable. Instead the estimate loads the system and times the
real per-frame work (reading the frame, building the graph, scoring the
hydrogen bonds and enumerating the paths) on a few frames spread over the
trajectory, then extrapolates to every frame that will be analysed.
"""

import logging
import math
import os
import time

import numpy as np

from .core import build_graph, compute_edge_probabilities, traverse_network

logger = logging.getLogger(__name__)

STAGES = ("read", "graph", "scoring", "paths")
TARGET_SECONDS = 600          # what a "comfortable" run is, for the stride suggestion


def format_duration(seconds):
    """'about 40 s', 'about 4 min', 'about 2 h 15 min'."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"about {max(1, int(round(seconds)))} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"about {int(round(minutes))} min"
    hours, rest = divmod(int(round(minutes)), 60)
    return f"about {hours} h {rest} min" if rest else f"about {hours} h"


def _sample_frames(frames, n_samples):
    """Evenly spread frame indices among the frames that will be analysed."""
    frames = list(frames)
    if len(frames) <= n_samples:
        return frames
    positions = np.linspace(0, len(frames) - 1, n_samples).round().astype(int)
    return [frames[i] for i in sorted(set(positions))]


def estimate_runtime(topo_file, traj_file, root_sel, water_sel="resname SOL or resname WAT or resname HOH",
                     stride=1, max_depth=10, coarse_cutoff=4.5, cooperativity=0.92, n_samples=3,
                     progress_callback=None, **_ignored):
    """
    Estimated duration of `run_analysis` with these settings.

    Returns a dict with the system size (atoms, residues, waters, trajectory
    frames, frames to analyse), the measured time per sampled frame and per
    stage, the estimate and its range in seconds, a one-line `summary` and a
    list of `hints` for making the run faster. Extra keyword arguments (the
    other run_analysis options) are accepted and ignored, so the same kwargs
    can be passed to both.

    `progress_callback(done, total)` is called after each sampled frame and
    may raise to abort.
    """
    import MDAnalysis as mda
    from .analysis import validate_calculation_parameters

    validate_calculation_parameters(stride=stride, max_depth=max_depth, cooperativity=cooperativity,
                                    coarse_cutoff=coarse_cutoff)
    if not traj_file:
        raise ValueError("A trajectory file is required (e.g. .xtc, .dcd, .nc).")
    for label, path in (("Topology", topo_file), ("Trajectory", traj_file)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    t0 = time.perf_counter()
    u = mda.Universe(topo_file, traj_file)
    load_s = time.perf_counter() - t0

    root = u.select_atoms(root_sel)
    if len(root) == 0:
        raise ValueError(f"Root selection '{root_sel}' returned no atoms. Check your resname/resid.")
    water = u.select_atoms(water_sel)

    n_traj = len(u.trajectory)
    frames = range(0, n_traj, stride)
    n_analysed = len(frames)
    samples = _sample_frames(frames, n_samples)

    per_frame, stage_totals, endpoints = [], dict.fromkeys(STAGES, 0.0), []
    for i, frame in enumerate(samples):
        t = time.perf_counter()
        u.trajectory[frame]
        t_read = time.perf_counter()
        g, root_indices = build_graph(u, water, root, max_distance=coarse_cutoff, max_depth=max_depth)
        t_graph = time.perf_counter()
        g = compute_edge_probabilities(g, u)
        t_score = time.perf_counter()
        paths = traverse_network(g, root_indices, max_depth=max_depth, cooperativity=cooperativity)
        t_paths = time.perf_counter()
        for stage, dt in zip(STAGES, (t_read - t, t_graph - t_read, t_score - t_graph, t_paths - t_score)):
            stage_totals[stage] += dt
        per_frame.append(t_paths - t)
        endpoints.append(len(paths))
        if progress_callback is not None:
            progress_callback(i + 1, len(samples))

    mean = float(np.mean(per_frame)) if per_frame else 0.0
    # Writing and bookkeeping add a little to each frame; 5 % is generous.
    estimate = load_s + 1.05 * mean * n_analysed
    low = load_s + 1.05 * min(per_frame, default=0.0) * n_analysed
    high = load_s + 1.05 * max(per_frame, default=0.0) * n_analysed
    stage_share = {k: (v / sum(stage_totals.values()) if sum(stage_totals.values()) else 0.0)
                   for k, v in stage_totals.items()}

    hints = []
    if estimate > 2 * TARGET_SECONDS and mean > 0:
        # How many frames fit in about TARGET_SECONDS, and the stride that gives that many.
        frames_that_fit = max(2, int((TARGET_SECONDS - min(load_s, TARGET_SECONDS / 2)) / (1.05 * mean)))
        suggested = int(math.ceil(n_traj / frames_that_fit))
        if suggested > stride:
            n_new = len(range(0, n_traj, suggested))
            hints.append(f"A stride of {suggested} instead of {stride} would analyse {n_new} frames in "
                         f"{format_duration(load_s + 1.05 * mean * n_new)}.")
    network = stage_share["graph"] + stage_share["scoring"] + stage_share["paths"]
    if network > 0.5 and max_depth > 6:
        hints.append(f"Most of the time goes into building and scoring the hydrogen-bond network around the root, "
                     f"which grows with Max depth: every extra water layer brings a new shell of waters into it. "
                     f"A Max depth of 6 instead of {max_depth} can be several times faster.")
    if stage_share["read"] > 0.5:
        hints.append("Most of the time goes into reading frames from the trajectory file; an uncompressed format "
                     "(.dcd, .xtc, .nc) on a local disk reads faster.")
    if max(per_frame, default=0.0) > 3 * max(min(per_frame, default=0.0), 1e-6) and len(per_frame) > 1:
        hints.append("The sampled frames differ a lot in cost (frames with many water bridges are slower), "
                     "so the real time may fall anywhere in the range.")

    result = {
        "n_atoms": len(u.atoms),
        "n_residues": len(u.residues),
        "n_waters": len(water.residues),
        "n_trajectory_frames": n_traj,
        "stride": stride,
        "n_analysed": n_analysed,
        "max_depth": max_depth,
        "sampled_frames": samples,
        "sample_endpoints": endpoints,
        "load_s": load_s,
        "per_frame_s": mean,
        "stage_share": stage_share,
        "estimate_s": estimate,
        "low_s": low,
        "high_s": high,
        "hints": hints,
    }
    rng = (f" (between {format_duration(low)[6:]} and {format_duration(high)[6:]})"
           if high > 1.5 * max(low, 1e-6) and high - low > 30 else "")
    result["summary"] = (f"Estimated runtime: {format_duration(estimate)}{rng} for {n_analysed} of {n_traj} frames "
                         f"(stride {stride}), {mean:.2f} s per frame measured on {len(samples)} sample frame"
                         f"{'s' if len(samples) != 1 else ''}. System: {len(u.atoms):,} atoms, "
                         f"{len(u.residues):,} residues, {len(water.residues):,} water molecules.")
    logger.info(result["summary"])
    for hint in hints:
        logger.info("Tip: " + hint)
    return result
