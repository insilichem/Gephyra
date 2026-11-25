"""
Hydration-site analysis: where do waters around the root sit persistently?

This is a frequency analysis, not a clustering. For every analysed frame:

1. Water oxygens are assigned to hydrogen-bond shells around the root, with
   the same shell-by-shell search as the water-bridge analysis: shell 1 =
   waters within hydrogen-bond distance (`shell_cutoff`) of a root atom,
   shell k = waters within that distance of a shell k-1 water. Waters up to
   `max_shell` are kept.
2. Their positions are put in the periodic image next to the root and
   superposed on the protein (the same superposition as `calculate`), so a
   spot has the same coordinates in every frame.

Over the whole trajectory the positions are counted on a 3D grid. Spots are
the highest counts, taken greedily from the top and at least
`min_site_distance` apart (two waters cannot share a spot). Each spot then
gets:

- occupancy: the fraction of analysed frames in which any water oxygen is
  within `site_radius` of its centre. It belongs to the spot, not to a
  molecule: different waters take turns at the same spot.
- enrichment: occupancy relative to bulk water, where a sphere of
  `site_radius` holds an oxygen with probability 1 - exp(-rho V)
  (rho = 0.0334 per cubic angstrom), about 0.13 for 1 A.
- mean and longest residence: runs of consecutive analysed frames.
- shell: the shell its waters most often belong to.

Outputs: a JSON file with the sites and the analysed system (read by
`visualize --mode sites`), an optional CSV table, an optional OpenDX map of
the water density relative to bulk, and a PDB of the site centres.
"""

import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

BULK_WATER_DENSITY = 0.0334   # oxygens per cubic angstrom at 300 K


def bulk_occupancy(radius):
    """Probability that a sphere of `radius` in bulk water contains an oxygen."""
    return 1.0 - np.exp(-BULK_WATER_DENSITY * 4.0 / 3.0 * np.pi * radius ** 3)


def water_oxygens(water_atoms):
    """
    One oxygen (the heaviest real atom) per water residue in the selection;
    massless virtual sites are never chosen.
    """
    picks = []
    for residue in water_atoms.residues:
        atoms = residue.atoms & water_atoms
        try:
            masses = atoms.masses
            heavy = atoms[masses > 0.1]
            pick = heavy[int(np.argmax(heavy.masses))] if len(heavy) else atoms[0]
        except Exception:
            names = [a for a in atoms if a.name.upper().startswith("O")]
            pick = names[0] if names else atoms[0]
        picks.append(pick.index)
    return water_atoms.universe.atoms[picks]


def assign_shells(root_positions, oxygen_positions, box, max_shell, cutoff):
    """
    Shell number (1..max_shell) for each oxygen, 0 when outside the shells.
    Shells follow hydrogen-bond-distance contacts outward from the root.
    """
    from MDAnalysis.lib.distances import capped_distance
    shells = np.zeros(len(oxygen_positions), dtype=int)
    if not len(oxygen_positions):
        return shells
    frontier = np.asarray(root_positions, dtype=np.float32)
    for shell in range(1, max_shell + 1):
        if not len(frontier):
            break
        pairs = capped_distance(frontier, oxygen_positions, max_cutoff=cutoff, box=box, return_distances=False)
        if not len(pairs):
            break
        new = np.unique(pairs[:, 1])
        new = new[shells[new] == 0]
        shells[new] = shell
        frontier = oxygen_positions[new]
    return shells


def _nearest_image(positions, centre, box):
    """Positions moved to the periodic image nearest to `centre`."""
    positions = np.asarray(positions, dtype=np.float64)
    if box is None or not np.all(np.asarray(box[:3]) > 0):
        return positions
    from MDAnalysis.lib.distances import minimize_vectors
    shift = minimize_vectors((positions - centre).astype(np.float32), np.asarray(box, dtype=np.float32))
    return centre + shift


def _runs(frame_positions):
    """Lengths of runs of consecutive analysed frames (frame positions are 0, 1, 2, ...)."""
    if not len(frame_positions):
        return []
    runs, current = [], 1
    for a, b in zip(frame_positions[:-1], frame_positions[1:]):
        if b == a + 1:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return runs


def find_sites(points, frame_of_point, shell_of_point, n_frames, site_radius=1.0, min_site_distance=2.5,
               min_occupancy=0.2, grid_spacing=0.5, max_shell=1, partners=None):
    """
    Hydration sites from the pooled oxygen positions (see the module docstring).

    `partners`, if given, is (partner_of_point, distance_of_point, labels):
    each observation's contact partner (see contact_partners) and the atom
    labels; every site then lists its most frequent partners.

    Returns (sites, grid, best_occupancy): grid = (counts, origin, spacing)
    of the occupancy map, counts being the number of oxygen observations per
    cell; best_occupancy is that of the most occupied spot, kept or not.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    from scipy.spatial import cKDTree

    if not len(points):
        return [], None, 0.0
    lo = points.min(axis=0) - 2 * site_radius
    hi = points.max(axis=0) + 2 * site_radius
    edges = [np.arange(lo[i], hi[i] + grid_spacing, grid_spacing) for i in range(3)]
    counts, _ = np.histogramdd(points, bins=edges)
    origin = np.array([e[0] for e in edges])

    # Smooth over about the site radius so a spot gives one peak, not a speckle.
    smooth = gaussian_filter(counts, sigma=smoothing_sigma(site_radius, grid_spacing))
    window = max(3, int(round(min_site_distance / grid_spacing)) | 1)
    peaks = (smooth == maximum_filter(smooth, size=window, mode="constant")) & (smooth > 0)
    idx = np.argwhere(peaks)
    order = np.argsort(-smooth[peaks])
    candidates = origin + (idx[order] + 0.5) * grid_spacing

    tree = cKDTree(points)
    bulk = bulk_occupancy(site_radius)
    scored = []
    for centre in candidates[:20000]:
        members = tree.query_ball_point(centre, site_radius)
        if not members:
            continue
        centre = points[members].mean(axis=0)            # refine on the actual oxygens
        members = tree.query_ball_point(centre, site_radius)
        frames = np.unique(frame_of_point[members])
        scored.append((len(frames) / n_frames, centre, frames, np.asarray(members)))
    scored.sort(key=lambda item: -item[0])
    best_occupancy = scored[0][0] if scored else 0.0
    scored = [item for item in scored if item[0] >= min_occupancy]

    sites = []
    for occupancy, centre, frames, members in scored:
        if any(np.linalg.norm(centre - np.array(s["position"])) < min_site_distance for s in sites):
            continue
        runs = _runs(frames)
        shells = shell_of_point[members]
        shell_counts = np.bincount(shells, minlength=max_shell + 1)[1:]
        sites.append({
            "site_id": len(sites) + 1,
            "position": [round(float(c), 3) for c in centre],
            "occupancy": round(float(occupancy), 4),
            "enrichment": round(float(occupancy / bulk), 2),
            "mean_residence_frames": round(float(np.mean(runs)), 2),
            "max_residence_frames": int(max(runs)),
            "shell": int(np.argmax(shell_counts) + 1),
            "shell_fractions": [round(float(c / shell_counts.sum()), 3) for c in shell_counts],
            "n_observations": int(len(members)),
        })
        if partners is not None:
            partner_of_point, distance_of_point, labels = partners
            p, d = partner_of_point[members], distance_of_point[members]
            found = p >= 0
            ids, n_contacts = np.unique(p[found], return_counts=True)   # (not `counts`: that is the grid)
            sites[-1]["contacts"] = [
                {"atom": labels[i], "fraction": round(float(c / len(members)), 3),
                 "mean_distance": round(float(np.nanmean(d[found][p[found] == i])), 2)}
                for i, c in sorted(zip(ids, n_contacts), key=lambda ic: -ic[1])[:3]]
    return sites, (counts, origin, grid_spacing), best_occupancy


def smoothing_sigma(site_radius, grid_spacing):
    """Gaussian width (in grid cells) used both for peak picking and for the written map."""
    return max(site_radius / (2 * grid_spacing), 0.5)


def density_map(counts, spacing, n_frames, site_radius=1.0):
    """
    Water density relative to bulk (1 = bulk water), smoothed like the peak
    picking so that single-frame visits do not show as speckle.
    """
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(counts.astype(np.float64), sigma=smoothing_sigma(site_radius, spacing))
    return smooth / (n_frames * spacing ** 3) / BULK_WATER_DENSITY


def map_level(density, origin, spacing, sites, floor=1.2):
    """
    Contour level for drawing the map: just below the density at the least
    occupied reported site (so every site is enclosed), never below `floor`
    times bulk.
    """
    if not sites:
        return floor
    values = []
    for site in sites:
        idx = np.floor((np.asarray(site["position"]) - origin) / spacing).astype(int)
        idx = np.clip(idx, 0, np.array(density.shape) - 1)
        values.append(density[tuple(idx)])
    return round(float(max(floor, 0.9 * min(values))), 2)


def _write_dx(path, density, origin, spacing):
    from gridData import Grid
    # gridData's origin is the centre of the first cell.
    Grid(density, origin=origin + spacing / 2.0, delta=spacing).export(path)

def _write_sites_pdb(path, sites):
    with open(path, "w") as f:
        f.write("REMARK   Gephyra hydration sites; occupancy column = site occupancy, B-factor = enrichment\n")
        for site in sites:
            x, y, z = site["position"]
            f.write(f"HETATM{site['site_id']:5d}  O   HOH S{site['site_id']:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{min(site['occupancy'], 9.99):6.2f}{min(site['enrichment'], 999.99):6.2f}           O\n")
        f.write("END\n")


PARTNER_CUTOFF = 4.0   # A: a water's contact partner is its nearest solute heavy atom within this distance


def contact_partners(oxygen_positions, solute_positions, box, cutoff=PARTNER_CUTOFF):
    """
    For each oxygen, the index (into solute_positions) of its nearest solute
    heavy atom within `cutoff`, and that distance; -1 / nan when none.
    Measured in the frame itself, so flexible residues are handled correctly.
    """
    from MDAnalysis.lib.distances import capped_distance
    n = len(oxygen_positions)
    partner, distance = np.full(n, -1), np.full(n, np.nan)
    if not n or not len(solute_positions):
        return partner, distance
    pairs, dist = capped_distance(oxygen_positions, solute_positions, max_cutoff=cutoff, box=box)
    if len(pairs):
        order = np.argsort(dist)
        oxygen, first = np.unique(pairs[order, 0], return_index=True)
        partner[oxygen] = pairs[order, 1][first]
        distance[oxygen] = dist[order][first]
    return partner, distance


def run_hydration_sites(topo_file, traj_file, root_sel, water_sel="resname SOL or resname WAT or resname HOH",
                        stride=1, max_shell=1, shell_cutoff=3.5, align_sel="auto", site_radius=1.0,
                        min_site_distance=2.5, min_occupancy=0.2, grid_spacing=0.5,
                        output_file="hydration_sites.json", csv_file=None, dx_file=None, pdb_file=None,
                        progress_callback=None):
    """
    Hydration-site analysis of the waters in the first `max_shell` hydrogen-bond
    shells around `root_sel` (see the module docstring). Returns a summary dict.

    `progress_callback(frames_done, frames_total)` is called after every
    analysed frame and may raise to abort.
    """
    import MDAnalysis as mda
    from .estimate import format_duration
    from .scene import FrameAligner, resolve_alignment

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    if not 1 <= int(max_shell) <= 3:
        raise ValueError(f"max_shell must be 1, 2 or 3, got {max_shell}.")
    for name, value in (("shell_cutoff", shell_cutoff), ("site_radius", site_radius),
                        ("min_site_distance", min_site_distance), ("grid_spacing", grid_spacing)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if not 0 <= min_occupancy <= 1:
        raise ValueError(f"min_occupancy must be between 0 and 1, got {min_occupancy}.")
    if not traj_file:
        raise ValueError("A trajectory file is required (e.g. .xtc, .dcd, .nc).")
    for label, path in (("Topology", topo_file), ("Trajectory", traj_file)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    u = mda.Universe(topo_file, traj_file)
    root = u.select_atoms(root_sel)
    if len(root) == 0:
        raise ValueError(f"Root selection '{root_sel}' returned no atoms. Check your resname/resid.")
    water = u.select_atoms(water_sel)
    if len(water) == 0:
        raise ValueError(f"Water selection '{water_sel}' returned no atoms. Check the solvent residue names.")
    oxygens = water_oxygens(water)
    from .scene import default_context_selection
    solute = u.select_atoms(f"({default_context_selection(water_sel)}) or ({root_sel})")
    if hasattr(solute, "masses") and len(solute[solute.masses > 1.5]):
        solute = solute[solute.masses > 1.5]          # heavy atoms: hydrogens are not useful partners
    labels = [f"{a.resname}{a.resid}:{a.name}" for a in solute]

    frames = range(0, len(u.trajectory), stride)
    n_frames = len(frames)
    alignment_sel = resolve_alignment(u, align_sel)
    aligner = FrameAligner(u, alignment_sel, frames[0]) if alignment_sel else None
    logger.info(f"Hydration sites around '{root_sel}': shells 1-{max_shell} (hydrogen-bond distance "
                f"{shell_cutoff} A), {n_frames} of {len(u.trajectory)} frames (stride {stride}), "
                + (f"superposed on '{alignment_sel}'." if aligner else "not superposed."))

    points, frame_of_point, shell_of_point, partner_of_point, distance_of_point = [], [], [], [], []
    started = time.perf_counter()
    for i, ts in enumerate(u.trajectory[::stride]):
        if i:
            left = (time.perf_counter() - started) / i * (n_frames - i)
            logger.info(f"Frame {ts.frame} ({i + 1}/{n_frames}, {format_duration(left)} left)...")
        o_pos = oxygens.positions
        shells = assign_shells(root.positions, o_pos, ts.dimensions, int(max_shell), shell_cutoff)
        keep = shells > 0
        if keep.any():
            pos = _nearest_image(o_pos[keep], root.positions.mean(axis=0), ts.dimensions)
            if aligner is not None:
                pos = aligner.apply(pos, aligner.current_transform())
            points.append(pos)
            frame_of_point.append(np.full(len(pos), i))
            shell_of_point.append(shells[keep])
            partner, distance = contact_partners(o_pos[keep], solute.positions, ts.dimensions)
            partner_of_point.append(partner)
            distance_of_point.append(distance)
        if progress_callback is not None:
            progress_callback(i + 1, n_frames)

    points = np.vstack(points) if points else np.zeros((0, 3))
    frame_of_point = np.concatenate(frame_of_point) if frame_of_point else np.zeros(0, int)
    shell_of_point = np.concatenate(shell_of_point) if shell_of_point else np.zeros(0, int)
    partners = (np.concatenate(partner_of_point) if partner_of_point else np.zeros(0, int),
                np.concatenate(distance_of_point) if distance_of_point else np.zeros(0),
                labels)

    sites, grid, best_occupancy = find_sites(points, frame_of_point, shell_of_point, n_frames, site_radius=site_radius,
                             min_site_distance=min_site_distance, min_occupancy=min_occupancy,
                             grid_spacing=grid_spacing, max_shell=int(max_shell), partners=partners)
    alignment = {"selection": alignment_sel, "reference_frame": frames[0]} if aligner else None

    level = None
    if dx_file and grid is not None:
        counts, origin, spacing = grid
        density = density_map(counts, spacing, n_frames, site_radius=site_radius)
        level = map_level(density, origin, spacing, sites)
        _write_dx(dx_file, density, origin, spacing)
    if pdb_file:
        _write_sites_pdb(pdb_file, sites)
    if csv_file:
        import csv
        with open(csv_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Site", "X", "Y", "Z", "Occupancy", "Enrichment_vs_bulk", "Mean_residence_frames",
                        "Max_residence_frames", "Shell", "Main_contact", "Contact_fraction", "Contact_mean_distance"])
            for s in sites:
                contact = (s.get("contacts") or [{"atom": "", "fraction": "", "mean_distance": ""}])[0]
                w.writerow([s["site_id"], *s["position"], s["occupancy"], s["enrichment"], s["mean_residence_frames"],
                            s["max_residence_frames"], s["shell"], contact["atom"], contact["fraction"],
                            contact["mean_distance"]])

    metadata = {
        "topo_file": os.path.abspath(topo_file), "traj_file": os.path.abspath(traj_file),
        "n_frames_analyzed": n_frames, "n_trajectory_frames": len(u.trajectory),
        "alignment": alignment, "dx_file": os.path.abspath(dx_file) if dx_file and grid is not None else None,
        "map_level": level,
        "bulk_occupancy": round(float(bulk_occupancy(site_radius)), 4),
        "parameters": {"root_sel": root_sel, "water_sel": water_sel, "stride": stride, "max_shell": int(max_shell),
                       "shell_cutoff": shell_cutoff, "site_radius": site_radius,
                       "min_site_distance": min_site_distance, "min_occupancy": min_occupancy,
                       "grid_spacing": grid_spacing},
    }
    with open(output_file, "w") as f:
        json.dump({"type": "gephyra_hydration_sites", "metadata": metadata, "sites": sites}, f, indent=2)

    warning = None
    if not len(points):
        warning = (f"No water was found within {shell_cutoff} A of the root in any of the {n_frames} analysed frames. "
                   "Check the root and water selections.")
    elif not sites:
        warning = (f"No spot is occupied in at least {min_occupancy:.0%} of the analysed frames: the most occupied "
                   f"one reaches {best_occupancy:.0%} (bulk water would give {bulk_occupancy(site_radius):.0%}). "
                   "Waters around the root exchange or move with it. Lower the minimum occupancy to list such "
                   "spots, or, if the ligand moves within the pocket, superpose on the ligand residue instead of "
                   f"the protein. A larger site radius (now {site_radius} A) also raises occupancies.")
    if warning:
        logger.warning(warning)
    logger.info(f"Found {len(sites)} hydration sites (occupancy >= {min_occupancy:.2f}; bulk water would give "
                f"{bulk_occupancy(site_radius):.2f}). Results saved to {output_file}")
    for s in sites[:10]:
        near = (s.get("contacts") or [{}])[0]
        logger.info(f"  Site {s['site_id']}: occupancy {s['occupancy']:.2f} ({s['enrichment']:.1f}x bulk), shell "
                    f"{s['shell']}, longest residence {s['max_residence_frames']} frames"
                    + (f", mostly {near.get('mean_distance')} A from {near.get('atom')} "
                       f"({near.get('fraction', 0):.0%} of the time)" if near else ""))

    return {
        "output_file": os.path.abspath(output_file),
        "csv_file": os.path.abspath(csv_file) if csv_file else None,
        "dx_file": metadata["dx_file"],
        "pdb_file": os.path.abspath(pdb_file) if pdb_file else None,
        "n_frames": n_frames,
        "n_sites": len(sites),
        "top_occupancy": sites[0]["occupancy"] if sites else 0.0,
        "best_occupancy": round(float(best_occupancy), 4),
        "warning": warning,
    }
