"""
Readers for Gephyra output files. Kept free of heavy imports (no
MDAnalysis), so the interface and the visualization exporters can inspect
results cheaply.
"""

import json

JSONL_TYPES = ("metadata", "frame", "path_statistics", "notice")


def load_clusters(path):
    """
    (clusters, metadata) from a cluster file.

    Current files are {"type": "gephyra_clusters", "metadata": ..., "clusters": [...]};
    older ones are a bare list, for which metadata is {}.
    """
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj, {}
    if isinstance(obj, dict) and isinstance(obj.get("clusters"), list):
        return obj["clusters"], obj.get("metadata") or {}
    raise ValueError(f"{path} is not a Gephyra cluster file.")


def load_sites(path):
    """(sites, metadata) from a hydration-site file written by `gephyra hydration`."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict) and obj.get("type") == "gephyra_hydration_sites" and isinstance(obj.get("sites"), list):
        return obj["sites"], obj.get("metadata") or {}
    raise ValueError(f"{path} is not a Gephyra hydration-site file.")


def read_jsonl_metadata(path):
    """The metadata line of a `calculate` .jsonl file, or {} if there is none."""
    with open(path) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                return obj if obj.get("type") == "metadata" else {}
    return {}


def detect_data_kind(path):
    """
    'jsonl' for the line-delimited output of `calculate`, 'cluster' for a
    cluster file (old list or new envelope format), 'sites' for a
    hydration-site file, None otherwise.
    """
    with open(path) as f:
        first = ""
        for line in f:
            if line.strip():
                first = line.strip()
                break
    if not first:
        return None
    if first.startswith("["):
        return "cluster"
    if first.startswith("{"):
        try:
            obj = json.loads(first)
        except ValueError:
            obj = None  # a multi-line JSON document: the cluster envelope
        if isinstance(obj, dict):
            if obj.get("type") in JSONL_TYPES:
                return "jsonl"
            return "cluster" if "clusters" in obj else ("sites" if "sites" in obj else None)
        # A multi-line JSON document: a cluster or hydration-site envelope.
        for kind, loader in (("cluster", load_clusters), ("sites", load_sites)):
            try:
                loader(path)
                return kind
            except (ValueError, OSError):
                continue
        return None
    return None


def structure_info(path):
    """
    What is known about the analysed system from a result file:
    {'topo_file', 'traj_file', 'root_sel', 'water_sel', 'alignment'} (values may be None).
    """
    kind = detect_data_kind(path)
    meta = {}
    if kind == "jsonl":
        meta = read_jsonl_metadata(path)
    elif kind == "cluster":
        meta = load_clusters(path)[1]
    elif kind == "sites":
        meta = load_sites(path)[1]
    params = meta.get("parameters", {}) if meta else {}
    return {
        "topo_file": meta.get("topo_file") if meta else None,
        "traj_file": meta.get("traj_file") if meta else None,
        "root_sel": params.get("root_sel"),
        "water_sel": params.get("water_sel"),
        "alignment": meta.get("alignment") if meta else None,
    }
