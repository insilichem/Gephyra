import argparse
import logging
import sys

from ._version import __version__
from .analysis import run_analysis, cluster_pathways
from .visualize import run_visualization, validate_chimerax_model, VISUALIZE_FORMATS, VISUALIZE_MODES

logger = logging.getLogger("gephyra")

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def _positive_int(value):
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {ivalue}")
    return ivalue

def _positive_float(value):
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}")
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {fvalue}")
    return fvalue

def _cooperativity(value):
    fvalue = _positive_float(value)
    if fvalue > 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {fvalue}")
    return fvalue

def _chimerax_model(value):
    try:
        return validate_chimerax_model(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))

def build_parser():
    parser = argparse.ArgumentParser(prog="gephyra", description="Gephyra: A network-graph and Fréchet clustering framework for water network detection.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Only log warnings and errors.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    subparsers.required = True

    # --- Calculate Subcommand ---
    calc_parser = subparsers.add_parser("calculate", help="Analyze trajectory to find water networks.")
    calc_parser.add_argument("--topo", required=True, help="Topology file (e.g., .pdb, .tpr).")
    calc_parser.add_argument("--traj", required=True, help="Trajectory file (e.g., .xtc, .dcd, .nc). To analyse a single structure, pass a coordinate file such as a .pdb.")
    calc_parser.add_argument("--root", required=True, help="MDAnalysis selection string for the root atoms (e.g., 'resname LIG').")
    calc_parser.add_argument("--water", default="resname SOL or resname WAT or resname HOH", help="MDAnalysis selection string for water.")
    calc_parser.add_argument("--stride", type=_positive_int, default=1, help="Frame stride for trajectory processing.")
    calc_parser.add_argument("--max_depth", type=_positive_int, default=10, help="Maximum number of consecutive waters in a path.")
    calc_parser.add_argument("--min_depth", type=_positive_int, default=1, help="Minimum number of consecutive waters in a path to save in results.")
    calc_parser.add_argument("--cooperativity", type=_cooperativity, default=0.92, help="Depth-dependent cooperativity factor for hydrogen bonds, in (0, 1]. 1.0 disables the effect.")
    calc_parser.add_argument("--align", default="auto",
        help="MDAnalysis selection used to superpose every frame onto the first analysed one before "
             "path coordinates are stored, so that clustering compares channels in a common frame "
             "even if the solute drifts or rotates. Default 'auto': protein C-alpha atoms "
             "(or nucleic-acid P).")
    calc_parser.add_argument("--no_align", action="store_true",
        help="Store path coordinates exactly as in the trajectory (only for trajectories already fitted).")
    calc_parser.add_argument("--coarse_cutoff", type=_positive_float, default=4.5, help="Coarse distance cutoff (Angstroms) for initial graph building.")
    calc_parser.add_argument("--output", default="results.jsonl", help="Output JSON Lines file for storing the raw network data.")
    calc_parser.add_argument("--csv", default=None, help="Optional output CSV file for storing human-readable summary data.")

    # --cluster / --cluster_threshold are kept for backward compatibility but are deprecated in favour of the dedicated 'cluster' sub-command below.
    # Their defaults are kept identical to the 'cluster' sub-command so that the two routes give the same result.
    calc_parser.add_argument("--cluster", action="store_true", help="Deprecated. Enable temporal clustering of pathways post-analysis. Prefer the 'cluster' sub-command.")
    calc_parser.add_argument("--cluster_threshold", type=float, default=6.0, help="Deprecated. Frechet threshold for the inline clustering. Same default as 'gephyra cluster --threshold'.")
    calc_parser.add_argument("--cluster_output", default="clustered_pathways.json", help="Deprecated. Output file for the inline clustering.")
    
    # cluster sub-command  (NEW – decoupled from 'calculate')
    cluster_parser = subparsers.add_parser("cluster", help="Cluster pathways from a finished calculation without re-running the MD analysis.",)
    cluster_parser.add_argument("--data", required=True, help="Input JSONL file generated by the 'calculate' sub-command.")
    cluster_parser.add_argument("--threshold", type=_positive_float, default=6.0,
        help="Discrete Fréchet distance threshold (Å) between 3D path coordinates "
             "for average-link clustering. Lower values produce finer clusters. "
             "Recommended range: 3.0–8.0 Å depending on system flexibility.")
    cluster_parser.add_argument("--coarse_trigger", type=_positive_int, default=1000,
        help="Number of paths below which the 9D coarse screening pass is bypassed "
             "and Fréchet distance is computed directly on all paths. "
             "Below this threshold, the Fréchet computation is fast enough that "
             "the coarse filter adds risk without benefit. "
             "Default: 1000 (≈500,000 pairs, completes in seconds on a modern CPU). "
             "Safe range: 500–2000 depending on available CPU time.")
    cluster_parser.add_argument("--coarse_threshold", type=_positive_float, default=None,
        help="9D feature-vector distance threshold (Å) for the coarse screening pass. "
         "Because it combines three 3D displacements, its numerical scale is up to "
         "sqrt(3) ≈ 1.73x larger than the Fréchet threshold. By default, it is "
         "calculated as threshold / sqrt(3).")
    cluster_parser.add_argument("--max_paths", type=_positive_int, default=60000, help="Hard cap on number of paths sent to the distance matrix. "
         "If exceeded, the top N paths by occupancy are used. "
         "Prevents out-of-memory crashes on large trajectories.")
    cluster_parser.add_argument("--output", default="clustered_pathways.json", help="Output JSON file for the cluster summary.")
    
    # --- Visualize Subcommand ---
    vis_parser = subparsers.add_parser("visualize", help="Generate visualization scripts from calculation results.")
    vis_parser.add_argument("--data", required=True, help="Input file: the .jsonl from 'calculate' for --mode density/frame, or the JSON from 'cluster' for --mode cluster.")
    vis_parser.add_argument("--format", choices=VISUALIZE_FORMATS, default="vmd",
        help="Target visualization software: vmd (.tcl), pymol (.py), chimera (legacy UCSF Chimera, .py + .bild) "
             "or chimerax (UCSF ChimeraX, .cxc + .bild, plus .pb pseudobonds in frame mode).")
    vis_parser.add_argument("--mode", choices=VISUALIZE_MODES, default="density", help="Visualization mode: 'density' (all frames), 'frame' (single frame), or 'cluster' (clustered medoids).")
    vis_parser.add_argument("--frame", type=int, default=None, help="Frame index to visualize (only used if mode='frame').")
    vis_parser.add_argument("--cluster_id", type=int, default=None, help="Specific cluster ID to visualize (only used if mode='cluster'). If None, visualizes all clusters.")
    vis_parser.add_argument("--output", default="pathways_viz", help="Output script file name or prefix.")
    vis_parser.add_argument("--chimerax_model", type=_chimerax_model, default=None,
        help="ChimeraX model id (only used with --format chimerax). With a molecular scene, the id the "
             "script opens the structure at (default #101; geometry and bridges use the next two ids). "
             "Geometry-only scripts address the structure you opened yourself (default #1).")
    vis_parser.add_argument("--topo", default=None,
        help="Topology of the analysed system. Defaults to the one recorded in the data file "
             "(current versions); needed for older result files to include the protein and waters.")
    vis_parser.add_argument("--traj", default=None, help="Trajectory of the analysed system (see --topo).")
    vis_parser.add_argument("--context", default=None,
        help="MDAnalysis selection of the solute shown around the network. "
             "Default: everything except solvent and common counter-ions.")
    vis_parser.add_argument("--top_clusters", type=_positive_int, default=None,
        help="Cluster mode: draw only the N most occupied clusters (default: all).")
    vis_parser.add_argument("--no_structure", action="store_true",
        help="Write geometry only (no structure snapshot); load the structure in the viewer yourself.")
    vis_parser.add_argument("--max_bond_draw_dist", type=_positive_float, default=6.0,
        help="Maximum Euclidean distance (Å) between two connected path atoms "
             "before the cylinder is treated as a PBC artifact and skipped. "
             "Should be slightly larger than your --coarse_cutoff value. "
             "Default: 6.0 Å (safe for the default coarse_cutoff of 4.5 Å).")

    # --- GUI Subcommand ---
    subparsers.add_parser("gui", help="Open the graphical interface (tkinter) for the whole workflow.")

    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        setup_logging(logging.DEBUG)
    elif args.quiet:
        setup_logging(logging.WARNING)
    else:
        setup_logging()

    if args.command == "calculate" and args.min_depth > args.max_depth:
        parser.error(f"--min_depth ({args.min_depth}) cannot exceed --max_depth ({args.max_depth}).")
    if args.command == "visualize" and args.mode == "frame" and args.frame is None:
        logger.warning("--mode frame given without --frame; the first frame with water bridges will be drawn.")

    if args.command == "gui":
        from .gui import main as gui_main
        return gui_main()

    try:
        ok = _dispatch(args)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(str(exc))
        return 1
    return 0 if ok is not False else 1

def _dispatch(args):
    if args.command == "calculate":
        run_analysis(
            topo_file=args.topo,
            traj_file=args.traj,
            root_sel=args.root,
            water_sel=args.water,
            stride=args.stride,
            max_depth=args.max_depth,
            min_depth=args.min_depth,
            cooperativity=args.cooperativity,
            coarse_cutoff=args.coarse_cutoff,
            align_sel=None if args.no_align else args.align,
            output_file=args.output,
            csv_file=args.csv,
            cluster=args.cluster,
            cluster_threshold=args.cluster_threshold,
            cluster_output=args.cluster_output
        )

    elif args.command == "cluster":
        cluster_pathways(
            data_file=args.data,
            threshold=args.threshold,
            coarse_threshold=args.coarse_threshold,
            coarse_trigger=args.coarse_trigger,
            max_paths=args.max_paths,
            output_file=args.output,
        )
        
    elif args.command == "visualize":
        return run_visualization(
            data_file=args.data,
            format=args.format,
            mode=args.mode,
            frame_idx=args.frame,
            cluster_id=args.cluster_id,
            output_file=args.output,
            max_bond_draw_dist=args.max_bond_draw_dist,
            chimerax_model=args.chimerax_model,
            with_structure=not args.no_structure,
            topo_file=args.topo,
            traj_file=args.traj,
            context_sel=args.context,
            top_clusters=args.top_clusters,
        ) is not None

if __name__ == "__main__":
    sys.exit(main())
