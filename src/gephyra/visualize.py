"""
Visualization scripts for VMD, PyMOL, UCSF Chimera and UCSF ChimeraX.

Each export happens in two steps:

1. `build_plan` turns the data (a `calculate` .jsonl or a cluster file) into
   a viewer-independent `Plan`: coloured groups of cylinders/spheres (cluster
   medoids, density overlay, ...) and, when the analysed topology/trajectory
   are known, a `scene.Scene` - a PDB snapshot of the protein, ligand and
   bridging waters for the frame being drawn.
2. One renderer per viewer writes a script from the plan.

With a scene, every script is self-contained: it loads the snapshot, shows
the protein as cartoon, the root residue, cofactors and bridging waters as
sticks, and the water bridges as dashed bonds or coloured tubes. Without one
(old result files, or the topology is no longer available) the scripts draw
the network geometry only and the user loads a matching structure first.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field

from .results import detect_data_kind, load_clusters, structure_info  # noqa: F401  (detect_data_kind re-exported)

logger = logging.getLogger(__name__)

VISUALIZE_FORMATS = ["vmd", "pymol", "chimera", "chimerax"]
VISUALIZE_MODES = ["density", "frame", "cluster"]
SCRIPT_EXTENSIONS = {"vmd": ".tcl", "pymol": ".py", "chimera": ".py", "chimerax": ".cxc"}

# How to run each script, shown in script headers, logs and the interface.
USAGE = {
    "vmd": "vmd -e {script}   (or, in VMD's Tk Console: source {script})",
    "pymol": "pymol {script}   (or, inside PyMOL: run {script})",
    "chimera": "chimera {script}   (or File > Open... in Chimera)",
    "chimerax": "chimerax {script}   (or drag it onto ChimeraX, or: open {script})",
}

# Categorical palette for distinguishing clusters (RGB in [0, 1]). Cycled when
# there are more clusters than colours.
CLUSTER_PALETTE = [
    (1.00, 0.50, 0.00),  # orange
    (0.12, 0.47, 0.71),  # blue
    (0.17, 0.63, 0.17),  # green
    (0.84, 0.15, 0.16),  # red
    (0.58, 0.40, 0.74),  # purple
    (0.55, 0.34, 0.29),  # brown
    (0.89, 0.47, 0.76),  # pink
    (0.50, 0.50, 0.50),  # grey
    (0.74, 0.74, 0.13),  # olive
    (0.09, 0.75, 0.81),  # cyan
]
# VMD's graphics colours are a fixed named set; these are the closest to the palette.
VMD_PALETTE = ["orange", "blue", "green", "red", "purple", "ochre", "pink", "gray", "lime", "cyan"]
DENSITY_RGB = (0.0, 0.8, 0.8)
FRAME_RGB = (1.0, 0.5, 0.0)

DEFAULT_CHIMERAX_SCENE_ID = "#101"


# --------------------------------------------------------------------- data

def read_cluster_json(data_file):
    """The list of clusters in a cluster file (old or new format)."""
    return load_clusters(data_file)[0]

def read_jsonl(data_file):
    frames = {}
    with open(data_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get('type') == 'frame':
                f_idx = str(obj['frame_idx'])
                if f_idx not in frames:
                    frames[f_idx] = []
                frames[f_idx].extend(obj['paths'])
    return frames


def _bild_path(output_file):
    """BILD geometry file that accompanies a script: same stem, .bild suffix."""
    return os.path.abspath(os.path.splitext(output_file)[0] + ".bild")

def _resolve_frame(frames, frame_idx):
    """
    Returns the key of the frame to draw, or None if there are no frames.

    A requested frame that is not in the data (e.g. skipped by --stride) falls
    back to the first available frame with a warning, rather than silently.
    """
    if not frames:
        return None
    if frame_idx is not None and str(frame_idx) in frames:
        return str(frame_idx)
    first = next(iter(frames))
    if frame_idx is not None:
        available = sorted(int(k) for k in frames)
        preview = ", ".join(str(k) for k in available[:10]) + (", ..." if len(available) > 10 else "")
        logger.warning(f"Frame {frame_idx} not found in data (available: {preview}). Using frame {first} instead.")
    return first

def _bridge_frames_hint(with_paths, n_frames, shown=12):
    """Which analysed frames have water bridges, so the user can pick one."""
    listed = ", ".join(with_paths[:shown]) + (", ..." if len(with_paths) > shown else "")
    return (f"Water bridges were found in {len(with_paths)} of {n_frames} analysed frames: {listed}. "
            "Choose one of these frame numbers; they are also listed in the first column ('Frame') of the "
            "CSV summary written by step 1.")


def _no_bridges_message(data_file, frames):
    return (f"No water bridges were found in any of the {len(frames)} frames of {os.path.basename(data_file)}, "
            "so there is nothing to draw. See the warning from step 1 (calculate): try a smaller stride, and "
            "check the root and water selections.")


def _rgb_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c * 255)) for c in rgb))

def _dist(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

def _segments(coords, max_bond_draw_dist):
    """Consecutive (p1, p2) pairs of a path, skipping PBC-artifact segments."""
    for p1, p2 in zip(coords[:-1], coords[1:]):
        if _dist(p1, p2) > max_bond_draw_dist:
            continue
        yield p1, p2

_MODEL_SPEC_RE = re.compile(r"^#\d+(\.\d+)*$")

def validate_chimerax_model(model_spec):
    """Raises ValueError unless model_spec is a ChimeraX model id such as '#1' or '#1.2'."""
    if not isinstance(model_spec, str) or not _MODEL_SPEC_RE.match(model_spec.strip()):
        raise ValueError(f"ChimeraX model must look like '#1' or '#1.2', got {model_spec!r}.")
    return model_spec.strip()


# --------------------------------------------------------------------- plan

@dataclass
class Group:
    """Geometry drawn in one colour: tubes between consecutive path atoms, plus spheres."""
    label: str
    rgb: tuple
    segments: list = field(default_factory=list)
    spheres: list = field(default_factory=list)
    cyl_radius: float = 0.2
    sphere_radius: float = 0.3
    transparency: float = 0.0          # 0 opaque .. 1 invisible
    comment: str = ""


@dataclass
class Plan:
    mode: str
    title: str
    source: str
    groups: list = field(default_factory=list)
    legend: list = field(default_factory=list)       # (text, rgb)
    scene: object = None                             # scene.Scene or None
    frame: object = None                             # trajectory frame for 'frame' mode
    # Geometry-only 'frame' mode: atoms of a structure the user loads themselves.
    atom_indices: list = field(default_factory=list)   # 0-based (VMD)
    atom_serials: list = field(default_factory=list)   # topology atom ids (Chimera/ChimeraX)
    atom_pairs: list = field(default_factory=list)     # serial pairs along paths


def _group_for_path_coords(label, rgb, coords_list, max_bond_draw_dist, cyl_radius, sphere_radius=None,
                           transparency=0.0, comment=""):
    g = Group(label, rgb, cyl_radius=cyl_radius, sphere_radius=sphere_radius or 0.0,
              transparency=transparency, comment=comment)
    for coords in coords_list:
        g.segments.extend(_segments(coords, max_bond_draw_dist))
        if sphere_radius:
            g.spheres.extend(coords)
    return g


def build_plan(data_file, mode, frame_idx=None, cluster_id=None, max_bond_draw_dist=6.0,
               structure=None, snapshot_path=None, top_clusters=None):
    """
    Viewer-independent description of what to draw, or None if there is
    nothing to draw (a warning has been logged).

    `structure`, if given, is a dict with topo_file, traj_file, root_sel,
    water_sel and optional context_sel; a snapshot is then written to
    `snapshot_path` and attached as `plan.scene`.
    """
    source = os.path.basename(data_file)

    if mode == "frame":
        frames = read_jsonl(data_file)
        if not frames:
            logger.error("No frames available for visualization.")
            return None
        with_paths = [k for k in frames if frames[k]]
        if not with_paths:
            logger.warning(_no_bridges_message(data_file, frames))
            return None
        if frame_idx is None:
            key = with_paths[0]
            logger.info(f"No frame given: drawing frame {key}, the first one with water bridges.")
        elif str(frame_idx) in frames:
            key = str(frame_idx)
        else:
            key = with_paths[0]
            logger.warning(f"Frame {frame_idx} was not analysed (see the stride used in step 1). "
                           f"Drawing frame {key} instead. {_bridge_frames_hint(with_paths, len(frames))}")
        paths = frames[key]
        if not paths:
            logger.warning(f"No paths found for frame {key}. Nothing to draw. "
                           f"{_bridge_frames_hint(with_paths, len(frames))}")
            return None
        plan = Plan("frame", f"frame {key}", source, frame=int(key))
        plan.legend.append((f"Frame {key}: {len(paths)} paths", FRAME_RGB))
        if structure:
            plan.scene = _snapshot(structure, int(key), snapshot_path, [p["nodes"] for p in paths])
        else:
            plan.groups.append(_group_for_path_coords(f"Frame {key}", FRAME_RGB, [p["coords"] for p in paths],
                                                      max_bond_draw_dist, 0.2, 0.3,
                                                      comment=f"Gephyra pathways, frame {key}"))
            seen = set()
            for p in paths:
                serials = [int(i) for i in p["atom_ids"]] if "atom_ids" in p else [int(i) + 1 for i in p["nodes"]]
                plan.atom_indices.extend(int(i) for i in p["nodes"])
                plan.atom_serials.extend(serials)
                for (a, b), (p1, p2) in zip(zip(serials[:-1], serials[1:]), zip(p["coords"][:-1], p["coords"][1:])):
                    key2 = (min(a, b), max(a, b))
                    if key2 not in seen and _dist(p1, p2) <= max_bond_draw_dist:
                        seen.add(key2)
                        plan.atom_pairs.append(key2)
            plan.atom_indices = sorted(set(plan.atom_indices))
            plan.atom_serials = sorted(set(plan.atom_serials))
        return plan

    if mode == "density":
        frames = read_jsonl(data_file)
        if not frames:
            logger.error("No frames available for visualization.")
            return None
        if not any(frames.values()):
            logger.warning(_no_bridges_message(data_file, frames))
            return None
        coords = [p["coords"] for paths in frames.values() for p in paths]
        plan = Plan("density", f"density over {len(frames)} frames", source)
        plan.groups.append(_group_for_path_coords("Density", DENSITY_RGB, coords, max_bond_draw_dist, 0.1,
                                                  transparency=0.7,
                                                  comment="Gephyra density overlay (all frames)"))
        plan.legend.append((f"All paths, {len(frames)} frames", DENSITY_RGB))
        if structure:
            plan.scene = _snapshot(structure, int(next(iter(frames))), snapshot_path, [])
        return plan

    if mode == "cluster":
        clusters = read_cluster_json(data_file)
        if cluster_id is not None:
            clusters = [c for c in clusters if c.get("cluster_id") == cluster_id]
            if not clusters:
                logger.warning(f"Cluster ID {cluster_id} not found in data.")
                return None
        elif top_clusters is not None:
            clusters = clusters[:top_clusters]
        clusters = [c for c in clusters if len(c.get("medoid_coords", [])) >= 2]
        if not clusters:
            logger.warning(f"{os.path.basename(data_file)} contains no clusters, so there is nothing to draw: "
                           "no pathway recurred in at least two frames. See the warning from step 2 (cluster) "
                           "for what to change, or draw single frames from the step-1 results with mode 'frame'.")
            return None

        def legend_text(c, n):
            occ, size = c.get("occupancy"), c.get("size")
            occ_txt = f"occ {occ:.2f}" if isinstance(occ, (int, float)) else ""
            size_txt = f"n={size}" if size is not None else ""
            return f"Cluster {c.get('cluster_id', n + 1)}  {occ_txt}  {size_txt}".strip()

        single = len(clusters) == 1
        title = f"cluster {clusters[0].get('cluster_id')}" if single else f"{len(clusters)} clusters"
        plan = Plan("cluster", title, source)

        # A single cluster with known medoid atoms is rebuilt as real molecules
        # at its medoid frame: protein conformation, waters and dashed bridges.
        if structure and single and clusters[0].get("medoid_nodes") and clusters[0].get("medoid_frame") is not None:
            c = clusters[0]
            plan.frame = int(c["medoid_frame"])
            plan.scene = _snapshot(structure, plan.frame, snapshot_path, [c["medoid_nodes"]])
            plan.legend.append((legend_text(c, 0) + f"  (frame {plan.frame})", CLUSTER_PALETTE[0]))
            return plan

        for n, c in enumerate(clusters):
            rgb = CLUSTER_PALETTE[n % len(CLUSTER_PALETTE)]
            cid = c.get("cluster_id", n + 1)
            plan.groups.append(_group_for_path_coords(
                f"Cluster {cid}", rgb, [c["medoid_coords"]], max_bond_draw_dist, 0.2, 0.3,
                comment=f"Cluster {cid}: medoid extracted from frame {c.get('medoid_frame', 'unknown')}"))
            plan.legend.append((legend_text(c, n), rgb))
        if structure:
            ref = clusters[0].get("medoid_frame")
            plan.scene = _snapshot(structure, int(ref) if ref is not None else 0, snapshot_path, [])
        return plan

    raise ValueError(f"Unknown mode {mode!r}; expected one of {', '.join(VISUALIZE_MODES)}.")


def _snapshot(structure, frame, snapshot_path, paths):
    from .scene import load_universe, write_snapshot
    u = load_universe(structure["topo_file"], structure.get("traj_file"))
    return write_snapshot(u, frame, snapshot_path, structure["root_sel"], structure["water_sel"],
                          paths=paths, context_sel=structure.get("context_sel"),
                          alignment=structure.get("alignment"))


def _header(comment, plan, fmt, output_file):
    script = os.path.basename(output_file)
    lines = [f"Gephyra {fmt} script: {plan.title} ({plan.mode} mode), from {plan.source}",
             "Run it with: " + USAGE[fmt].format(script=script)]
    if fmt == "vmd":
        lines.append(f"The -e is required: 'vmd {script}' makes VMD read this file as a molecule and nothing is drawn.")
    if plan.scene:
        lines.append(f"Loads the structure snapshot itself: {plan.scene.pdb_path} (frame {plan.scene.frame}).")
    else:
        lines.append("Geometry only: load the analysed structure in the viewer first.")
    return "".join(f"{comment} {line}\n" for line in lines)


# ---------------------------------------------------------------------- VMD

def _tcl_vec(p):
    return f"{{{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}}}"

def _tcl_path(path):
    return "{" + path.replace("\\", "/") + "}"

def _vmd_colour(rgb):
    if rgb in CLUSTER_PALETTE:
        return VMD_PALETTE[CLUSTER_PALETTE.index(rgb)]
    return "cyan" if rgb == DENSITY_RGB else "orange"

def write_vmd(plan, output_file):
    out = [_header("#", plan, "vmd", output_file)]
    s = plan.scene
    if s:
        out.append(f"set gmol [mol new {_tcl_path(s.pdb_path)} type pdb waitfor all]\n")
        out.append("mol rename $gmol gephyra_structure\nmol top $gmol\nmol delrep 0 $gmol\n")
        out.append("mol representation NewCartoon 0.30 12.0 4.50 0\nmol color ColorID 6\n"
                   "mol selection {protein or nucleic}\nmol material AOChalky\nmol addrep $gmol\n")
        out.append("mol representation Licorice 0.25 12.0 12.0\nmol color Name\n"
                   "mol selection {not (protein or nucleic)}\nmol material Opaque\nmol addrep $gmol\n")
        out.append(f"mol representation Licorice 0.30 12.0 12.0\nmol color ColorID 7\n"
                   f"mol selection {{chain {s.root_chain} and carbon}}\nmol addrep $gmol\n")
        if s.bridge_coords:
            out.append("graphics $gmol color orange\n")
            for p1, p2 in s.bridge_coords:
                out.append(f"graphics $gmol line {_tcl_vec(p1)} {_tcl_vec(p2)} width 4 style dashed\n")
    else:
        # VMD can only hold graphics inside a molecule: create an empty one if
        # nothing is loaded, instead of failing on every 'graphics top' line.
        out.append("if {[molinfo num] == 0} { mol new }\n")
        if plan.mode == "frame" and plan.atom_indices:
            out.append("mol color Name\nmol representation Licorice 0.3 12 12\n")
            out.append(f"mol selection \"index {' '.join(str(i) for i in plan.atom_indices)}\"\n")
            out.append("mol addrep top\n")
        out.append("graphics top delete all\n")
    target = "$gmol" if s else "top"
    # Drawing commands go inside a procedure: at top level VMD prints the id
    # returned by every 'graphics' call, flooding the console.
    body_start = len(out)
    for g in plan.groups:
        if g.transparency > 0:
            out.append(f"material change opacity Ghost {max(0.05, 1 - g.transparency):.2f}\n"
                       f"graphics {target} material Ghost\n")
        else:
            out.append(f"graphics {target} material Opaque\n")
        out.append(f"graphics {target} color {_vmd_colour(g.rgb)}\n")
        for p1, p2 in g.segments:
            out.append(f"graphics {target} cylinder {_tcl_vec(p1)} {_tcl_vec(p2)} radius {g.cyl_radius} resolution 20 filled yes\n")
        for p in g.spheres:
            out.append(f"graphics {target} sphere {_tcl_vec(p)} radius {g.sphere_radius}\n")
    body = out[body_start:]
    del out[body_start:]
    if s and s.bridge_coords:
        # bridges were emitted before body_start; move them into the procedure too
        bridges = [l for l in out if l.startswith("graphics $gmol")]
        out = [l for l in out if not l.startswith("graphics $gmol")]
        body = bridges + body
    if body:
        out.append("proc gephyra_draw {gmol} {\n")
        out.extend("    " + line for line in body)
        out.append("}\n")
        out.append(f"gephyra_draw {'$gmol' if s else '[molinfo top]'}\n")
    if plan.mode == "frame" and not s and plan.frame is not None:
        out.append(f"animate goto {plan.frame}\n")
    out.append("display resetview\naxes location off\n")
    if s:
        # Centre on the ligand and waters rather than the whole protein.
        out.append(f"set gfocus [atomselect $gmol {{chain {s.root_chain} {s.water_chain}}}]\n"
                   "if {[$gfocus num] > 0} { molinfo $gmol set center [list [measure center $gfocus]]; "
                   "translate to 0 0 0; scale by 2.2 }\n$gfocus delete\n")
    for text, rgb in plan.legend[:30]:
        out.append(f'puts "Gephyra: {text} ({_vmd_colour(rgb)})"\n')
    with open(output_file, "w") as f:
        f.writelines(out)
    return [output_file]


# -------------------------------------------------------------------- PyMOL

def write_pymol(plan, output_file):
    out = [_header("#", plan, "pymol", output_file),
           "from pymol import cmd\nfrom pymol.cgo import CYLINDER, SPHERE, COLOR, ALPHA\n\n",
           "cmd.delete('gephyra_*')  # re-running replaces the previous Gephyra scene\n"]
    s = plan.scene
    if s:
        out.append(f"cmd.load(r'{s.pdb_path}', 'gephyra_structure')\n")
        out.append("cmd.hide('everything', 'gephyra_structure')\n"
                   "cmd.show('cartoon', 'gephyra_structure and polymer')\n"
                   "cmd.set('cartoon_color', 'lightblue', 'gephyra_structure')\n"
                   "cmd.show('sticks', 'gephyra_structure and not polymer')\n"
                   "cmd.color('grey70', 'gephyra_structure and not polymer and elem C')\n"
                   f"cmd.color('green', 'gephyra_structure and chain {s.root_chain} and elem C')\n"
                   f"cmd.set('stick_radius', 0.18, 'gephyra_structure and chain {s.water_chain}')\n"
                   # PyMOL guesses bonds by distance and joins the two water H atoms
                   f"cmd.unbond('gephyra_structure and chain {s.water_chain} and elem H', "
                   f"'gephyra_structure and chain {s.water_chain} and elem H')\n")
        for a, b in s.bridges:
            out.append(f"cmd.distance('gephyra_bridges', 'gephyra_structure and id {a}', 'gephyra_structure and id {b}')\n")
        if s.bridges:
            out.append("cmd.hide('labels', 'gephyra_bridges')\ncmd.color('orange', 'gephyra_bridges')\n"
                       "cmd.set('dash_radius', 0.08, 'gephyra_bridges')\ncmd.set('dash_gap', 0.25, 'gephyra_bridges')\n")
    # One CGO object per group (toggle them in the object panel); merged
    # beyond 40 groups to keep the panel usable.
    groups = plan.groups
    named = len(groups) <= 40
    for i, g in enumerate(groups):
        name = f"gephyra_{re.sub(r'[^A-Za-z0-9]+', '_', g.label).strip('_').lower()}" if named else "gephyra_network"
        r, gg, b = g.rgb
        out.append(f"obj = [ALPHA, {1 - g.transparency:.2f}, COLOR, {r:.2f}, {gg:.2f}, {b:.2f}]\n")
        for p1, p2 in g.segments:
            out.append(f"obj += [CYLINDER, {p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f}, {p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f}, "
                       f"{g.cyl_radius:.3f}, {r:.2f}, {gg:.2f}, {b:.2f}, {r:.2f}, {gg:.2f}, {b:.2f}]\n")
        for p in g.spheres:
            out.append(f"obj += [SPHERE, {p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, {g.sphere_radius:.3f}]\n")
        out.append(f"cmd.load_cgo(obj, '{name}', state=1)\n" if named or i == 0 else
                   f"cmd.load_cgo(obj, '{name}_{i}', state=1)\n")
    if s and s.bridges:
        out.append(f"cmd.orient('gephyra_structure and (chain {s.root_chain} or chain {s.water_chain})')\n")
    elif s and groups:
        # CGO objects cannot be used in selections: zoom on a temporary
        # pseudoatom at the centre of the drawn network instead.
        pts = [p for g in groups for seg in g.segments for p in seg] + [p for g in groups for p in g.spheres]
        cx, cy, cz = (sum(p[i] for p in pts) / len(pts) for i in range(3))
        radius = max(_dist(p, (cx, cy, cz)) for p in pts) + 4.0
        out.append(f"cmd.orient('gephyra_structure and chain {s.root_chain}')\n"
                   f"cmd.pseudoatom('gephyra_tmp_view', pos=[{cx:.3f}, {cy:.3f}, {cz:.3f}])\n"
                   f"cmd.zoom('gephyra_tmp_view', {radius:.1f})\ncmd.delete('gephyra_tmp_view')\n")
    else:
        out.append("cmd.zoom('gephyra_*')\n")
    for text, rgb in plan.legend[:30]:
        out.append(f"print('Gephyra: {text} ({_rgb_hex(rgb)})')\n")
    with open(output_file, "w") as f:
        f.writelines(out)
    return [output_file]


# ------------------------------------------------------------------ Chimera

def _write_bild(path, plan):
    with open(path, "w") as bf:
        for g in plan.groups:
            if g.comment:
                bf.write(f".comment {g.comment}\n")
            bf.write(f".color {g.rgb[0]:.2f} {g.rgb[1]:.2f} {g.rgb[2]:.2f}\n")
            if g.transparency > 0:
                bf.write(f".transparency {g.transparency:.1f}\n")
            for p1, p2 in g.segments:
                bf.write(f".cylinder {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f} {p2[0]:.3f} {p2[1]:.3f} {p2[2]:.3f} {g.cyl_radius}\n")
            for p in g.spheres:
                bf.write(f".sphere {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {g.sphere_radius}\n")
    return path

def write_chimera(plan, output_file):
    """
    UCSF Chimera (1.x) Python script. Files are opened through
    chimera.openModels.open, not the 'open' command, whose parser does not
    accept quoted paths (the cause of "MidasError: No such file or directory").
    """
    written = [output_file]
    bild = None
    if plan.groups:
        bild = _write_bild(_bild_path(output_file), plan)
        written.append(bild)
    s = plan.scene
    out = [_header("#", plan, "chimera", output_file),
           "import chimera\nfrom chimera import runCommand as rc, openModels, selection\n\n",
           "# Re-running replaces the previous Gephyra scene.\n",
           "openModels.close([m for m in openModels.list() if getattr(m, 'name', '').startswith('gephyra_')])\n"]
    if s:
        out.append(f"mol = openModels.open(r'{s.pdb_path}')[0]\nmol.name = 'gephyra_structure'\n"
                   "spec = '#%d' % mol.id\n"
                   "rc('~display ' + spec)\nrc('ribbon ' + spec)\nrc('ribcolor 0.69,0.77,0.87 ' + spec)\n"
                   f"het = [a for r in mol.residues if r.isHet or r.id.chainId in ('{s.root_chain}', '{s.water_chain}') "
                   "for a in r.atoms]\n"
                   "selection.setCurrent(het)\n"
                   "rc('display sel')\nrc('represent stick sel')\nrc('color byhet sel')\nselection.setCurrent([])\n"
                   f"rc('color 0.13,0.55,0.13,a %s:.{s.root_chain}@C=' % spec)\n")
        if s.bridges:
            out.append("from chimera.misc import getPseudoBondGroup\n"
                       "by_serial = dict((a.serialNumber, a) for a in mol.atoms)\n"
                       "grp = getPseudoBondGroup('gephyra_bridges', associateWith=[mol])\n"
                       "grp.color = chimera.MaterialColor(1.0, 0.5, 0.0, 1.0)\n"
                       "grp.lineType = chimera.Dash\ngrp.lineWidth = 3\n"
                       f"for a, b in {s.bridges!r}:\n"
                       "    grp.newPseudoBond(by_serial[a], by_serial[b])\n")
    elif plan.mode == "frame" and plan.atom_serials:
        out.append(f"serial_set = set({plan.atom_serials!r})\n"
                   "atoms = [a for m in openModels.list() if hasattr(m, 'atoms') for a in m.atoms "
                   "if a.serialNumber in serial_set]\n"
                   "selection.setCurrent(atoms)\nrc('display sel')\nrc('represent stick sel')\nselection.setCurrent([])\n")
    if bild:
        out.append(f"geo = openModels.open(r'{bild}', type='Bild')[0]\ngeo.name = 'gephyra_network'\n")
        # Align with the structure's coordinate frame, whichever model that is.
        out.append("ref = mol if 'mol' in globals() else next((m for m in openModels.list() "
                   "if hasattr(m, 'atoms') and m is not geo), None)\n"
                   "if ref is not None:\n    geo.openState.xform = ref.openState.xform\n")
    # 2D labels need a graphics context; keep going (and print the legend)
    # when Chimera runs without one, e.g. with --nogui.
    if plan.legend:
        out.append("try:\n")
        for row, (text, rgb) in enumerate(plan.legend[:20]):
            out.append(f"    rc('2dlabels create gephyra_l{row} text \"{text}\" color {rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f} "
                       f"xpos 0.02 ypos {0.95 - 0.04 * row:.2f} size 14')\n")
        out.append("except Exception:\n")
        out.append(f"    print({chr(10).join(t for t, _ in plan.legend[:20])!r})\n")
    if s and s.bridges:
        out.append(f"rc('focus %s:.{s.water_chain} %s:.{s.root_chain}' % (spec, spec))\n")
    else:
        out.append("rc('focus')\n")
    with open(output_file, "w") as f:
        f.writelines(out)
    return written


# ----------------------------------------------------------------- ChimeraX

def _cxc_quote(path):
    """Quote a file path for a ChimeraX command line."""
    return '"' + path.replace('\\', '/').replace('"', '\\"') + '"'

def _serial_spec(serials, model_spec):
    """ChimeraX atom spec matching any of the given PDB serial numbers in one model."""
    return " | ".join(f"{model_spec}@@serial_number={int(s)}" for s in serials)

def _chimerax_legend(plan):
    out = ["2dlabels delete\n"]
    for row, (text, rgb) in enumerate(plan.legend[:20]):
        out.append(f'2dlabels text "{text}" color {_rgb_hex(rgb)} xpos 0.02 ypos {0.95 - 0.04 * row:.2f} size 14\n')
    if len(plan.legend) > 20:
        out.append(f"# Legend truncated: {len(plan.legend) - 20} more clusters are drawn but not labelled.\n")
    return out

def write_chimerax(plan, output_file, model_spec=None):
    """
    UCSF ChimeraX command script (.cxc) plus .bild geometry and, for bridges,
    a .pb pseudobond file.

    With a scene the snapshot is opened at a fixed model id (default #101;
    geometry at #102, bridges at #103), which the script first closes so that
    re-running it replaces the previous Gephyra scene. Without a scene, atom
    specs refer to the structure the user opened, in `model_spec` (default #1).
    """
    stem = os.path.splitext(output_file)[0]
    s = plan.scene
    written = [output_file]
    out = [_header("#", plan, "chimerax", output_file)]

    if s:
        base = validate_chimerax_model(model_spec or DEFAULT_CHIMERAX_SCENE_ID)
        if "." in base:
            raise ValueError("The ChimeraX scene needs a top-level model id such as #101, not a sub-model.")
        n = int(base[1:])
        mid, gid = f"#{n}", f"#{n + 1}"
        out.append(f"close {mid}-{n + 2}\n")
        out.append(f"open {_cxc_quote(s.pdb_path)} id {n}\n")
        # Explicit chain specs, not 'protein'/'solvent' categories: ChimeraX's
        # residue classification varies (e.g. MCPB-renamed metal-site residues).
        sticks = f"{mid}/{s.root_chain},{s.water_chain},H"
        out.append(f"hide {mid} target a\ncartoon {mid}\ncolor {mid} lightsteelblue target c\n"
                   f"show {sticks} target a\nstyle {sticks} stick\n"
                   f"color {sticks} byhetero\ncolor {mid}/{s.root_chain} & C forestgreen\n"
                   f"size {mid}/{s.water_chain} stickRadius 0.15\n")
        if plan.groups:
            bild = _write_bild(stem + ".bild", plan)
            written.append(os.path.abspath(bild))
            out.append(f"open {_cxc_quote(os.path.abspath(bild))} id {n + 1}\n")
        if s.bridges:
            pb = os.path.abspath(stem + ".pb")
            with open(pb, "w") as pf:
                pf.write("; halfbond = false\n; color = orange\n; radius = 0.12\n; dashes = 8\n")
                for a, b in s.bridges:
                    pf.write(f"{mid}@@serial_number={a} {mid}@@serial_number={b}\n")
            written.append(pb)
            out.append(f"open {_cxc_quote(pb)} id {n + 2}\n")
        out.extend(_chimerax_legend(plan))
        focus = f"{mid}/{s.water_chain} | {mid}/{s.root_chain}" if s.bridges else f"{mid} {gid}" if plan.groups else mid
        out.append(f"view {focus}\nlighting soft\ngraphics silhouettes true\n")
    else:
        model_spec = validate_chimerax_model(model_spec or "#1")
        out.append(f"# Atoms are addressed in model {model_spec}; regenerate with --chimerax_model if your structure is elsewhere.\n"
                   "# For a multi-model PDB trajectory use 'open traj.pdb coordset true' so it loads as one model.\n")
        bild = os.path.abspath(stem + ".bild")
        _write_bild(bild, plan)
        written.append(bild)
        if plan.mode == "frame":
            out.append(f"# Frame {plan.frame}: {len(plan.atom_serials)} atoms.\n")
            chunk = 50
            for i in range(0, len(plan.atom_serials), chunk):
                verb = "select" if i == 0 else "select add"
                out.append(f"{verb} {_serial_spec(plan.atom_serials[i:i + chunk], model_spec)}\n")
            out.append("show sel atoms\nstyle sel stick\ncolor sel orange\nselect clear\n")
            out.append(f"open {_cxc_quote(bild)} name gephyra_frame_{plan.frame}\n")
            pb = os.path.abspath(stem + ".pb")
            with open(pb, "w") as pf:
                pf.write("; halfbond = false\n; color = orange\n; radius = 0.15\n; dashes = 6\n")
                for a, b in plan.atom_pairs:
                    pf.write(f"{model_spec}@@serial_number={a} {model_spec}@@serial_number={b}\n")
            written.append(pb)
            out.append("# Dynamic pseudobonds between the same atoms (they follow a loaded trajectory).\n"
                       "# Requires the structure to be open; comment out if it is not.\n")
            out.append(f"open {_cxc_quote(pb)}\n")
        elif plan.mode == "density":
            out.append(f"open {_cxc_quote(bild)} name gephyra_density\n")
        else:
            out.append(f"open {_cxc_quote(bild)} name gephyra_clusters\n")
            out.extend(_chimerax_legend(plan))
    with open(output_file, "w") as f:
        f.writelines(out)
    return written


# ------------------------------------------------------------------ exports

RENDERERS = {"vmd": write_vmd, "pymol": write_pymol, "chimera": write_chimera, "chimerax": write_chimerax}

def _export(fmt, data_file, output_file, mode, frame_idx, cluster_id, max_bond_draw_dist, structure, top_clusters,
            **renderer_kwargs):
    snapshot = os.path.splitext(output_file)[0] + "_structure.pdb"
    plan = build_plan(data_file, mode, frame_idx=frame_idx, cluster_id=cluster_id,
                      max_bond_draw_dist=max_bond_draw_dist, structure=structure,
                      snapshot_path=snapshot, top_clusters=top_clusters)
    if plan is None:
        return None
    written = RENDERERS[fmt](plan, output_file, **renderer_kwargs)
    if plan.scene:
        written.append(plan.scene.pdb_path)
    logger.info(f"{fmt.upper() if fmt == 'vmd' else fmt.capitalize()} script for {plan.title} written to "
                f"{os.path.abspath(output_file)}" + (" (with molecular scene)" if plan.scene else " (geometry only)"))
    return [os.path.abspath(p) for p in written]

def export_vmd_script(data_file, output_file="draw_pathways.tcl", mode="frame", frame_idx=None, cluster_id=None,
                      max_bond_draw_dist=6.0, structure=None, top_clusters=None):
    """VMD Tcl script; run with `vmd -e script.tcl`. Returns the files written, or None."""
    return _export("vmd", data_file, output_file, mode, frame_idx, cluster_id, max_bond_draw_dist, structure, top_clusters)

def export_pymol_script(data_file, output_file="draw_pathways.py", mode="frame", frame_idx=None, cluster_id=None,
                        max_bond_draw_dist=6.0, structure=None, top_clusters=None):
    """PyMOL Python script; run with `pymol script.py`. Returns the files written, or None."""
    return _export("pymol", data_file, output_file, mode, frame_idx, cluster_id, max_bond_draw_dist, structure, top_clusters)

def export_chimera_script(data_file, output_file="draw_pathways.py", mode="frame", frame_idx=None, cluster_id=None,
                          max_bond_draw_dist=6.0, structure=None, top_clusters=None):
    """UCSF Chimera Python script (+ .bild). Returns the files written, or None."""
    return _export("chimera", data_file, output_file, mode, frame_idx, cluster_id, max_bond_draw_dist, structure, top_clusters)

def export_chimerax_script(data_file, output_file="draw_pathways.cxc", mode="frame", frame_idx=None, cluster_id=None,
                           max_bond_draw_dist=6.0, model_spec=None, structure=None, top_clusters=None):
    """UCSF ChimeraX command script (+ .bild, .pb). Returns the files written, or None."""
    return _export("chimerax", data_file, output_file, mode, frame_idx, cluster_id, max_bond_draw_dist, structure,
                   top_clusters, model_spec=model_spec)


def resolve_structure(data_file, topo_file=None, traj_file=None, context_sel=None, root_sel=None, water_sel=None):
    """
    The `structure` dict for build_plan, from explicit arguments and the
    metadata stored in the data file; None (with a warning) when the analysed
    topology is not known, in which case scripts draw geometry only.
    """
    info = structure_info(data_file) if os.path.exists(data_file) else {}
    topo = topo_file or info.get("topo_file")
    traj = traj_file or info.get("traj_file")
    root = root_sel or info.get("root_sel")
    water = water_sel or info.get("water_sel")
    if not topo:
        logger.warning("The data file does not record the analysed topology (it was written by an older version of Gephyra). "
                       "Writing geometry only; pass --topo and --traj to include the protein and waters.")
        return None
    if not root or not water:
        logger.warning("The root/water selections used by 'calculate' are unknown for this file; "
                       "writing geometry only.")
        return None
    return {"topo_file": topo, "traj_file": traj, "root_sel": root, "water_sel": water, "context_sel": context_sel,
            "alignment": info.get("alignment")}


def run_visualization(data_file, format="vmd", mode="density", frame_idx=None, cluster_id=None, output_file="pathways_viz",
                      max_bond_draw_dist=6.0, chimerax_model=None, with_structure=True, topo_file=None, traj_file=None,
                      context_sel=None, top_clusters=None):
    """
    Writes a visualization script and returns the list of files written
    (script first), or None if nothing was written (the reason is logged).

    With `with_structure` (the default) the analysed topology/trajectory,
    taken from the data file or from `topo_file`/`traj_file`, are used to
    write a structure snapshot that the script loads itself.
    """
    if not os.path.exists(data_file):
        logger.error(f"Data file {data_file} not found.")
        return None

    # Catch the most common mistake (passing results.jsonl to --mode cluster,
    # or clustered_pathways.json to density/frame) with an actionable message
    # instead of a JSON decode traceback or an empty script.
    kind = detect_data_kind(data_file)
    if mode == "cluster" and kind == "jsonl":
        logger.error(f"--mode cluster expects the JSON written by 'gephyra cluster', but {data_file} "
                     "looks like the .jsonl written by 'gephyra calculate'. Run 'gephyra cluster' first.")
        return None
    if mode in ("frame", "density") and kind == "cluster":
        logger.error(f"--mode {mode} expects the .jsonl written by 'gephyra calculate', but {data_file} "
                     "looks like a clustered_pathways.json. Use --mode cluster for that file.")
        return None
    if format not in RENDERERS:
        logger.error(f"Unknown format: {format}")
        return None

    ext = SCRIPT_EXTENSIONS[format]
    out = output_file if output_file.endswith(ext) else f"{output_file}{ext}"
    out_dir = os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Output folder does not exist: {out_dir}")

    structure = resolve_structure(data_file, topo_file, traj_file, context_sel) if with_structure else None

    kwargs = dict(output_file=out, mode=mode, frame_idx=frame_idx, cluster_id=cluster_id,
                  max_bond_draw_dist=max_bond_draw_dist, structure=structure, top_clusters=top_clusters)
    if format == "chimerax":
        kwargs["model_spec"] = chimerax_model
    exporter = {"vmd": export_vmd_script, "pymol": export_pymol_script,
                "chimera": export_chimera_script, "chimerax": export_chimerax_script}[format]
    written = exporter(data_file, **kwargs)
    if written:
        logger.info("Run it with: " + USAGE[format].format(script=written[0]))
    return written
