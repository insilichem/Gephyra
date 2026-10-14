import json
import logging
import os

logger = logging.getLogger(__name__)

def read_cluster_json(data_file):
    with open(data_file, 'r') as f:
        return json.load(f)

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
    """BILD geometry file that accompanies a Chimera script: same stem, .bild suffix."""
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

def detect_data_kind(data_file):
    """
    'cluster' for a clustered_pathways.json (a JSON array), 'jsonl' for the
    line-delimited output of `calculate`, or None if it cannot be told apart.
    """
    with open(data_file, 'r') as f:
        while True:
            ch = f.read(1)
            if not ch:
                return None
            if not ch.isspace():
                break
    if ch == '[':
        return 'cluster'
    if ch == '{':
        return 'jsonl'
    return None

def export_vmd_script(data_file, output_file="draw_pathways.tcl", mode="frame", frame_idx=None, cluster_id=None, max_bond_draw_dist=6.0):
    """
    Generates a Tcl script to visualize the network in VMD using dynamic selections.
    """
    if mode in ["frame", "density"]:
        frames = read_jsonl(data_file)
        
    if mode == "frame":
        frame_idx = _resolve_frame(frames, frame_idx)

        if frame_idx is None:
            logger.error("No frames available for visualization.")
            return

        paths = frames[str(frame_idx)]

        # Collect all atom indices
        all_indices = set()
        for path in paths:
            all_indices.update(path["nodes"])

        indices_str = " ".join(str(i) for i in sorted(all_indices))

        if not indices_str:
            logger.warning(f"No paths found for frame {frame_idx}. Skipping VMD script generation to prevent Tcl crash.")
            return

        with open(output_file, 'w') as f:
            f.write("mol color Name\n")
            f.write("mol representation Licorice 0.3 12 12\n")
            f.write(f"mol selection \"index {indices_str}\"\n")
            f.write("mol addrep top\n")
            f.write("mol delrep 0 top\n")

            # Append static graphics geometry for connected wires
            f.write("\ngraphics top material Opaque\n")
            f.write("graphics top color orange\n")
            for path in paths:
                coords = path["coords"]
                if len(coords) < 2:
                    continue
                for i in range(len(coords) - 1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    if sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5 > max_bond_draw_dist:
                        continue  # PBC artifact: segment spans > max_bond_draw_dist Å, skip
                    f.write(f"graphics top cylinder {{{p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}}} {{{p2[0]:.3f} {p2[1]:.3f} {p2[2]:.3f}}} radius 0.2 resolution 20 filled yes\n")
                    f.write(f"graphics top sphere {{{p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}}} radius 0.3\n")

                # write the last sphere
                p_last = coords[-1]
                f.write(f"graphics top sphere {{{p_last[0]:.3f} {p_last[1]:.3f} {p_last[2]:.3f}}} radius 0.3\n")

            # Optional: Move to the frame
            f.write(f"animate goto {frame_idx}\n")

        logger.info(f"VMD script written to {output_file} for frame {frame_idx}")

    elif mode == "density":
        with open(output_file, 'w') as f:
            f.write("graphics top delete all\n")
            f.write("material change opacity Ghost 0.1\n")
            f.write("graphics top material Ghost\n")
            f.write("graphics top color cyan\n\n")

            for f_idx, paths in frames.items():
                for path in paths:
                    coords = path["coords"]
                    for i in range(len(coords) - 1):
                        p1 = coords[i]
                        p2 = coords[i+1]
                        if sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5 > max_bond_draw_dist:
                            continue  # PBC artifact: segment spans > max_bond_draw_dist Å, skip
                        f.write(f"graphics top cylinder {{{p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}}} {{{p2[0]:.3f} {p2[1]:.3f} {p2[2]:.3f}}} radius 0.1 resolution 20 filled yes\n")
                        f.write(f"graphics top sphere {{{p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}}} radius 0.3\n")

        logger.info(f"VMD density script written to {output_file}")
        
    elif mode == "cluster":
        clusters = read_cluster_json(data_file)
        if cluster_id is not None:
            clusters = [c for c in clusters if c.get("cluster_id") == cluster_id]
            if not clusters:
                logger.warning(f"Cluster ID {cluster_id} not found in data.")
                return

        with open(output_file, 'w') as f:
            f.write("graphics top delete all\n")
            f.write("material change opacity Opaque 1.0\n")
            f.write("graphics top material Opaque\n")
            f.write("graphics top color orange\n\n")

            for cluster in clusters:
                coords = cluster.get("medoid_coords", [])
                if len(coords) < 2:
                    continue
                for i in range(len(coords) - 1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    if sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5 > max_bond_draw_dist:
                        continue  # PBC artifact: segment spans > max_bond_draw_dist Å, skip
                    f.write(f"graphics top cylinder {{{p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}}} {{{p2[0]:.3f} {p2[1]:.3f} {p2[2]:.3f}}} radius 0.2 resolution 20 filled yes\n")
                    f.write(f"graphics top sphere {{{p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}}} radius 0.3\n")
                
                # write the last sphere
                p_last = coords[-1]
                f.write(f"graphics top sphere {{{p_last[0]:.3f} {p_last[1]:.3f} {p_last[2]:.3f}}} radius 0.3\n")

        logger.info(f"VMD cluster script written to {output_file}")


def export_pymol_script(data_file, output_file="draw_pathways.py", mode="frame", frame_idx=None, cluster_id=None, max_bond_draw_dist=6.0):
    """
    Generates a Python script to visualize the network in PyMOL as CGO.
    """
    if mode in ["frame", "density"]:
        frames = read_jsonl(data_file)

    with open(output_file, 'w') as f:
        f.write("from pymol import cmd\n")
        f.write("from pymol.cgo import *\n\n")
        f.write("obj = []\n")

        def write_cylinder(p1, p2, radius=0.2, color=[0.0, 1.0, 1.0], alpha=1.0):
            f.write(f"obj.extend([CYLINDER, {p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f}, {p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f}, {radius:.3f}, ")
            f.write(f"{color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f}, {color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f}])\n")

        def write_sphere(p, radius=0.3, color=[0.0, 1.0, 1.0]):
            f.write(f"obj.extend([COLOR, {color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f}])\n")
            f.write(f"obj.extend([SPHERE, {p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, {radius:.3f}])\n")

        if mode == "frame":
            frame_idx = _resolve_frame(frames, frame_idx)

            if frame_idx is None:
                logger.error("No frames available for visualization.")
                f.write("\ncmd.load_cgo(obj, 'water_network')\n")
                return

            paths = frames[str(frame_idx)]
            for path in paths:
                coords = path["coords"]
                if len(coords) < 2:
                    continue
                for i in range(len(coords) - 1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    if sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5 > max_bond_draw_dist:
                        continue  # PBC artifact filter
                    write_cylinder(p1, p2, radius=0.2)
                    write_sphere(p1, radius=0.3)
                write_sphere(coords[-1], radius=0.3)

            logger.info(f"PyMOL script written to {output_file} for frame {frame_idx}")

        elif mode == "density":
            for f_idx, paths in frames.items():
                for path in paths:
                    coords = path["coords"]
                    for i in range(len(coords) - 1):
                        p1 = coords[i]
                        p2 = coords[i+1]
                        if sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5 > max_bond_draw_dist:
                            continue  # PBC artifact filter
                        write_cylinder(p1, p2, radius=0.05, color=[0.0, 0.8, 0.8])
            logger.info(f"PyMOL density script written to {output_file}")
            
        elif mode == "cluster":
            clusters = read_cluster_json(data_file)
            if cluster_id is not None:
                clusters = [c for c in clusters if c.get("cluster_id") == cluster_id]
                if not clusters:
                    logger.warning(f"Cluster ID {cluster_id} not found in data.")
                    return

            for cluster in clusters:
                coords = cluster.get("medoid_coords", [])
                if len(coords) < 2:
                    continue
                for i in range(len(coords) - 1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    if sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5 > max_bond_draw_dist:
                        continue  # PBC artifact filter
                    write_cylinder(p1, p2, radius=0.2, color=[1.0, 0.5, 0.0], alpha=1.0)
                    write_sphere(p1, radius=0.3, color=[1.0, 0.5, 0.0])
                write_sphere(coords[-1], radius=0.3, color=[1.0, 0.5, 0.0])
            logger.info(f"PyMOL cluster script written to {output_file}")

        f.write("\ncmd.load_cgo(obj, 'water_network')\n")
        if mode == "density":
            f.write("cmd.set('cgo_transparency', 0.8, 'water_network')\n")


def export_chimera_script(data_file, output_file="draw_pathways.py", mode="frame", frame_idx=None, cluster_id=None, max_bond_draw_dist=6.0):
    """
    Generates a Python script to visualize the network in UCSF Chimera.
    """
    if mode in ["frame", "density"]:
        frames = read_jsonl(data_file)

    if mode == "frame":
        frame_idx = _resolve_frame(frames, frame_idx)

        if frame_idx is None:
            logger.error("No frames available for visualization.")
            return

        paths = frames[str(frame_idx)]

        all_ids = set()
        for path in paths:
            # Use atom_ids if available (from analysis.py upgrade), otherwise fallback to indices+1
            if "atom_ids" in path:
                all_ids.update(path["atom_ids"])
            else:
                all_ids.update(i+1 for i in path["nodes"])

        bild_file = _bild_path(output_file)
        with open(bild_file, 'w') as bild_f:
            bild_f.write('.color orange\n')
            for path in paths:
                coords = path["coords"]
                if len(coords) < 2:
                    continue
                for i in range(len(coords) - 1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    if sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5 > max_bond_draw_dist:
                        continue  # PBC artifact filter
                    bild_f.write(f".cylinder {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f} {p2[0]:.3f} {p2[1]:.3f} {p2[2]:.3f} 0.2\n")
                    bild_f.write(f".sphere {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f} 0.3\n")

                # write the last sphere
                p_last = coords[-1]
                bild_f.write(f".sphere {p_last[0]:.3f} {p_last[1]:.3f} {p_last[2]:.3f} 0.3\n")

        with open(output_file, 'w') as f:
            f.write("import chimera\n")
            f.write("from chimera import runCommand\n")
            sel_str = ", ".join(str(i) for i in sorted(all_ids))
            f.write(f"serial_set = set([{sel_str}])\n")
            f.write("atoms = [a for m in chimera.openModels.list() for a in m.atoms if a.serialNumber in serial_set]\n")
            f.write("chimera.selection.setCurrent(atoms)\n")
            f.write("runCommand('show sel')\n")
            f.write("runCommand('repr stick sel')\n")
            f.write(f"runCommand(\"open '{bild_file}'\")\n")
            f.write("print('Linking BILD geometry to Model #0 (ensure your protein is loaded first)')\n")
            f.write("try:\n")
            f.write("    runCommand('matrixset ~0 #last')\n")
            f.write("except Exception as e:\n")
            f.write("    print('Warning: Failed to link BILD geometry to Model #0. Ensure a protein structure is loaded first.')\n")

        logger.info(f"Chimera script written to {output_file} for frame {frame_idx} (BILD geometry: {bild_file})")

    elif mode == "density":
        bild_file = _bild_path(output_file)
        with open(bild_file, 'w') as bild_f:
            bild_f.write('.color cyan\n')
            
            for f_idx, paths in frames.items():
                for path in paths:
                    coords = path["coords"]
                    for i in range(len(coords) - 1):
                        p1 = coords[i]
                        p2 = coords[i+1]
                        if sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5 > max_bond_draw_dist:
                            continue  # PBC artifact filter
                        bild_f.write(f".cylinder {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f} {p2[0]:.3f} {p2[1]:.3f} {p2[2]:.3f} 0.1\n")
        with open(output_file, 'w') as f:
            f.write("from chimera import runCommand, openModels\n")
            f.write("before_models = set(openModels.list())\n")
            f.write(f"runCommand(\"open '{bild_file}'\")\n")
            f.write("new_models = set(openModels.list()) - before_models\n")
            f.write("for m in new_models:\n")
            f.write("    runCommand('transparency 50 models #{}'.format(m.id))\n")
        logger.info(f"Chimera density script written to {output_file} (BILD geometry: {bild_file})")
        
    elif mode == "cluster":
        clusters = read_cluster_json(data_file)
        if cluster_id is not None:
            clusters = [c for c in clusters if c.get("cluster_id") == cluster_id]
            if not clusters:
                logger.warning(f"Cluster ID {cluster_id} not found in data.")
                return

        bild_file = _bild_path(output_file)
        with open(bild_file, 'w') as bild_f:
            bild_f.write('.color orange\n')

            for cluster in clusters:
                medoid_frame = cluster.get("medoid_frame", "Unknown")
                bild_f.write(f".comment Medoid extracted from Frame {medoid_frame}\n")
                coords = cluster.get("medoid_coords", [])
                if len(coords) < 2:
                    continue
                for i in range(len(coords) - 1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    if sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5 > max_bond_draw_dist:
                        continue  # PBC artifact filter
                    bild_f.write(f".cylinder {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f} {p2[0]:.3f} {p2[1]:.3f} {p2[2]:.3f} 0.2\n")
                    bild_f.write(f".sphere {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f} 0.3\n")

                # write the last sphere
                p_last = coords[-1]
                bild_f.write(f".sphere {p_last[0]:.3f} {p_last[1]:.3f} {p_last[2]:.3f} 0.3\n")

        with open(output_file, 'w') as f:
            f.write("from chimera import runCommand, openModels\n")
            f.write(f"runCommand(\"open '{bild_file}'\")\n")
            f.write("print('Linking BILD geometry to Model #0 (ensure your protein is loaded first)')\n")
            f.write("try:\n")
            f.write("    runCommand('matrixset ~0 #last')\n")
            f.write("except Exception as e:\n")
            f.write("    print('Warning: Failed to link BILD geometry to Model #0. Ensure a protein structure is loaded first.')\n")
        logger.info(f"Chimera cluster script written to {output_file} (BILD geometry: {bild_file})")


def run_visualization(data_file, format="vmd", mode="density", frame_idx=None, cluster_id=None, output_file="pathways_viz", max_bond_draw_dist=6.0):
    if not os.path.exists(data_file):
        logger.error(f"Data file {data_file} not found.")
        return

    # Catch the most common mistake (passing results.jsonl to --mode cluster,
    # or clustered_pathways.json to density/frame) with an actionable message
    # instead of a JSON decode traceback or an empty script.
    kind = detect_data_kind(data_file)
    if mode == "cluster" and kind == "jsonl":
        logger.error(f"--mode cluster expects the JSON written by 'gephyra cluster', but {data_file} "
                     "looks like the .jsonl written by 'gephyra calculate'. Run 'gephyra cluster' first.")
        return
    if mode in ("frame", "density") and kind == "cluster":
        logger.error(f"--mode {mode} expects the .jsonl written by 'gephyra calculate', but {data_file} "
                     "looks like a clustered_pathways.json. Use --mode cluster for that file.")
        return

    # Append appropriate extension based on format if not provided by user
    if format == "vmd" and not output_file.endswith(".tcl"):
        out = f"{output_file}.tcl"
    elif format in ["pymol", "chimera"] and not output_file.endswith(".py"):
        out = f"{output_file}.py"
    else:
        out = output_file

    if format == "vmd":
        export_vmd_script(data_file, output_file=out, mode=mode, frame_idx=frame_idx, cluster_id=cluster_id, max_bond_draw_dist=max_bond_draw_dist)
    elif format == "pymol":
        export_pymol_script(data_file, output_file=out, mode=mode, frame_idx=frame_idx, cluster_id=cluster_id, max_bond_draw_dist=max_bond_draw_dist)
    elif format == "chimera":
        export_chimera_script(data_file, output_file=out, mode=mode, frame_idx=frame_idx, cluster_id=cluster_id, max_bond_draw_dist=max_bond_draw_dist)
    else:
        logger.error(f"Unknown format: {format}")
