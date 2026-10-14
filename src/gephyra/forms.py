"""
Toolkit-independent form handling for the Gephyra interface.

Each pipeline stage (calculate, cluster, visualize) is described by a set of
string-valued form fields, as an entry widget would hold them. The functions
here turn those strings into keyword arguments for the library functions,
applying the same constraints as the command line, and raise `FormError`
naming the offending field so an interface can point the user at it.

Nothing in this module imports tkinter, so it is fully testable headless and
reusable by any other front end.
"""

import os
import shlex

from .visualize import VISUALIZE_FORMATS, VISUALIZE_MODES, detect_data_kind, validate_chimerax_model

STAGES = ("calculate", "cluster", "visualize")

DEFAULT_WATER = "resname SOL or resname WAT or resname HOH"

# Defaults mirror the argparse defaults in cli.py.
DEFAULTS = {
    "calculate": {
        "topo": "", "traj": "", "root": "", "water": DEFAULT_WATER,
        "stride": "1", "max_depth": "10", "min_depth": "1",
        "cooperativity": "0.92", "coarse_cutoff": "4.5",
        "output": "results.jsonl", "csv": "",
    },
    "cluster": {
        "data": "", "threshold": "6.0", "coarse_threshold": "",
        "coarse_trigger": "1000", "max_paths": "60000",
        "output": "clustered_pathways.json",
    },
    "visualize": {
        "data": "", "format": "chimerax", "mode": "cluster", "frame": "",
        "cluster_id": "", "max_bond_draw_dist": "6.0", "chimerax_model": "#1",
        "output": "pathways_viz",
    },
}


class FormError(ValueError):
    """A form value failed validation. `field` names the offending field."""

    def __init__(self, field, message):
        super().__init__(message)
        self.field = field
        self.message = message


def _text(values, field, required=False, label=None):
    value = str(values.get(field, "") or "").strip()
    if required and not value:
        raise FormError(field, f"{label or field} is required.")
    return value


def _int(values, field, label, minimum=None, optional=False):
    raw = _text(values, field)
    if not raw:
        if optional:
            return None
        raise FormError(field, f"{label} is required.")
    try:
        value = int(raw)
    except ValueError:
        raise FormError(field, f"{label} must be a whole number, got {raw!r}.")
    if minimum is not None and value < minimum:
        raise FormError(field, f"{label} must be >= {minimum}, got {value}.")
    return value


def _float(values, field, label, positive=True, maximum=None, optional=False):
    raw = _text(values, field)
    if not raw:
        if optional:
            return None
        raise FormError(field, f"{label} is required.")
    try:
        value = float(raw)
    except ValueError:
        raise FormError(field, f"{label} must be a number, got {raw!r}.")
    if positive and value <= 0:
        raise FormError(field, f"{label} must be greater than 0, got {value}.")
    if maximum is not None and value > maximum:
        raise FormError(field, f"{label} must be at most {maximum}, got {value}.")
    return value


def _existing_file(values, field, label, required=True):
    path = _text(values, field, required=required, label=label)
    if path and not os.path.isfile(path):
        raise FormError(field, f"{label} not found: {path}")
    return path or None


def _output_path(values, field, label):
    path = _text(values, field, required=True, label=label)
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise FormError(field, f"Folder for {label.lower()} does not exist: {parent}")
    return path


def parse_calculate(values):
    """Form strings -> keyword arguments for analysis.run_analysis."""
    kwargs = {
        "topo_file": _existing_file(values, "topo", "Topology file"),
        "traj_file": _existing_file(values, "traj", "Trajectory file", required=False),
        "root_sel": _text(values, "root", required=True, label="Root selection"),
        "water_sel": _text(values, "water", required=True, label="Water selection"),
        "stride": _int(values, "stride", "Stride", minimum=1),
        "max_depth": _int(values, "max_depth", "Max depth", minimum=1),
        "min_depth": _int(values, "min_depth", "Min depth", minimum=1),
        "cooperativity": _float(values, "cooperativity", "Cooperativity", maximum=1.0),
        "coarse_cutoff": _float(values, "coarse_cutoff", "Coarse cutoff"),
        "output_file": _output_path(values, "output", "Output JSONL"),
        "csv_file": None,
    }
    if kwargs["min_depth"] > kwargs["max_depth"]:
        raise FormError("min_depth", f"Min depth ({kwargs['min_depth']}) cannot exceed max depth ({kwargs['max_depth']}).")
    if _text(values, "csv"):
        kwargs["csv_file"] = _output_path(values, "csv", "CSV summary")
    return kwargs


def parse_cluster(values):
    """Form strings -> keyword arguments for analysis.cluster_pathways."""
    data = _existing_file(values, "data", "Pathway data (.jsonl)")
    if detect_data_kind(data) == "cluster":
        raise FormError("data", "This looks like a clustered_pathways.json; clustering needs the .jsonl written by step 1.")
    return {
        "data_file": data,
        "threshold": _float(values, "threshold", "Fréchet threshold"),
        "coarse_threshold": _float(values, "coarse_threshold", "Coarse threshold", optional=True),
        "coarse_trigger": _int(values, "coarse_trigger", "Coarse trigger", minimum=1),
        "max_paths": _int(values, "max_paths", "Max paths", minimum=1),
        "output_file": _output_path(values, "output", "Output JSON"),
    }


def parse_visualize(values):
    """Form strings -> keyword arguments for visualize.run_visualization."""
    fmt = _text(values, "format", required=True, label="Format")
    if fmt not in VISUALIZE_FORMATS:
        raise FormError("format", f"Format must be one of {', '.join(VISUALIZE_FORMATS)}.")
    mode = _text(values, "mode", required=True, label="Mode")
    if mode not in VISUALIZE_MODES:
        raise FormError("mode", f"Mode must be one of {', '.join(VISUALIZE_MODES)}.")

    data = _existing_file(values, "data", "Input data")
    kind = detect_data_kind(data)
    if mode == "cluster" and kind == "jsonl":
        raise FormError("data", "Mode 'cluster' needs the clustered_pathways.json from step 2, not the .jsonl from step 1.")
    if mode in ("frame", "density") and kind == "cluster":
        raise FormError("data", f"Mode '{mode}' needs the .jsonl from step 1, not a clustered_pathways.json.")

    kwargs = {
        "data_file": data,
        "format": fmt,
        "mode": mode,
        "frame_idx": _int(values, "frame", "Frame", minimum=0, optional=True) if mode == "frame" else None,
        "cluster_id": _int(values, "cluster_id", "Cluster ID", minimum=1, optional=True) if mode == "cluster" else None,
        "output_file": _output_path(values, "output", "Output script"),
        "max_bond_draw_dist": _float(values, "max_bond_draw_dist", "Max bond draw distance"),
    }
    if fmt == "chimerax":
        try:
            kwargs["chimerax_model"] = validate_chimerax_model(_text(values, "chimerax_model") or "#1")
        except ValueError as exc:
            raise FormError("chimerax_model", str(exc))
    return kwargs


PARSERS = {"calculate": parse_calculate, "cluster": parse_cluster, "visualize": parse_visualize}


def parse_stage(stage, values):
    if stage not in PARSERS:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {', '.join(STAGES)}.")
    return PARSERS[stage](values)


# kwarg name -> CLI flag, per stage. Used to show users the equivalent command.
_CLI_FLAGS = {
    "calculate": [("topo_file", "--topo"), ("traj_file", "--traj"), ("root_sel", "--root"),
                  ("water_sel", "--water"), ("stride", "--stride"), ("max_depth", "--max_depth"),
                  ("min_depth", "--min_depth"), ("cooperativity", "--cooperativity"),
                  ("coarse_cutoff", "--coarse_cutoff"), ("output_file", "--output"), ("csv_file", "--csv")],
    "cluster": [("data_file", "--data"), ("threshold", "--threshold"), ("coarse_threshold", "--coarse_threshold"),
                ("coarse_trigger", "--coarse_trigger"), ("max_paths", "--max_paths"), ("output_file", "--output")],
    "visualize": [("data_file", "--data"), ("format", "--format"), ("mode", "--mode"), ("frame_idx", "--frame"),
                  ("cluster_id", "--cluster_id"), ("chimerax_model", "--chimerax_model"),
                  ("max_bond_draw_dist", "--max_bond_draw_dist"), ("output_file", "--output")],
}


def build_command_line(stage, kwargs):
    """The `gephyra ...` command equivalent to running `stage` with `kwargs`."""
    parts = ["gephyra", stage]
    for key, flag in _CLI_FLAGS[stage]:
        value = kwargs.get(key)
        if value is None or value == "":
            continue
        parts += [flag, str(value)]
    return " ".join(shlex.quote(p) for p in parts)


def run_stage(stage, kwargs, progress_callback=None):
    """
    Runs one stage with already-parsed kwargs.

    `progress_callback(done, total)` is forwarded to the stages that report
    progress (calculate and cluster). Imports are deferred so that merely
    opening the interface does not load MDAnalysis.
    """
    if stage == "calculate":
        from .analysis import run_analysis
        return run_analysis(**kwargs, progress_callback=progress_callback)
    if stage == "cluster":
        from .analysis import cluster_pathways
        return cluster_pathways(**kwargs, progress_callback=progress_callback)
    if stage == "visualize":
        from .visualize import run_visualization
        return run_visualization(**kwargs)
    raise ValueError(f"Unknown stage {stage!r}; expected one of {', '.join(STAGES)}.")


def visualize_output_path(kwargs):
    """Final script path run_visualization will write (it appends the extension)."""
    out = kwargs["output_file"]
    ext = {"vmd": ".tcl", "pymol": ".py", "chimera": ".py", "chimerax": ".cxc"}[kwargs["format"]]
    return out if out.endswith(ext) else out + ext
