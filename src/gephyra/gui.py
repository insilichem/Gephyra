"""
Desktop interface for the Gephyra workflow.

Built on the standard-library tkinter/ttk toolkit with a custom flat theme
(dark or light), so it adds no dependencies and stays light on memory.

A sidebar lists the three steps of the pipeline - 1. Pathway Analysis
(`calculate`), 2. Clustering (`cluster`) and 3. Visualization (`visualize`) -
and the output folder all results are written to. Form values are parsed and
validated by `gephyra.forms`, the same rules the command line applies; this
module only handles widgets, threading and dialogs.

Stages run on a worker thread. Log records and progress updates reach the Tk
main loop through a queue, so the window stays responsive and the log updates
in real time. Launch with `gephyra gui` or `gephyra-gui`.
"""

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import traceback

from . import forms
from ._version import __version__
from .visualize import USAGE

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
    from tkinter import font as tkfont
except ImportError:  # pragma: no cover - depends on the Python build
    tk = None

logger = logging.getLogger(__name__)

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".config", "gephyra", "gui.json")

# ----------------------------------------------------------------- palettes

THEMES = {
    "dark": {
        "bg": "#0d1117", "sidebar": "#0a0e14", "surface": "#151b23", "surface2": "#1c2430",
        "border": "#2a3441", "text": "#e6edf3", "muted": "#8b98a6", "accent": "#22d3ee",
        "accent_hover": "#67e8f9", "accent_text": "#06202a", "entry": "#0b0f15",
        "success": "#34d399", "error": "#f87171", "warning": "#fbbf24", "select": "#155e75",
        "log_bg": "#0a0e14", "log_text": "#c9d4df",
    },
    "light": {
        "bg": "#f4f6f9", "sidebar": "#ffffff", "surface": "#ffffff", "surface2": "#eef1f5",
        "border": "#d6dde5", "text": "#0f172a", "muted": "#5b6b7c", "accent": "#0891b2",
        "accent_hover": "#0e7490", "accent_text": "#ffffff", "entry": "#ffffff",
        "success": "#059669", "error": "#dc2626", "warning": "#b45309", "select": "#cffafe",
        "log_bg": "#f8fafc", "log_text": "#1e293b",
    },
}

UI_FONTS = ("Inter", "Cantarell", "Ubuntu", "Noto Sans", "Segoe UI", "SF Pro Text", "Helvetica Neue",
            "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica")
MONO_FONTS = ("JetBrains Mono", "Fira Code", "Ubuntu Mono", "Cascadia Mono", "Menlo", "Consolas",
              "DejaVu Sans Mono", "Liberation Mono", "Courier New")
# Tk builds without Xft (e.g. conda-forge's "noxft" tk) only see X11 core
# fonts, almost all fixed bitmaps; Charter is the one smooth, scalable family
# usually available there.
CORE_UI_FONTS = ("bitstream charter", "charter", "helvetica")
CORE_MONO_FONTS = ("fixed", "courier 10 pitch", "courier")
XFT_HINT = ("Fonts look blocky? This Python's Tk was built without anti-aliased fonts (Xft). Use the conda "
            "environment from the README (environment.yml), or run:  conda install -c conda-forge \"tk=*=xft*\"")

# ------------------------------------------------------------------- fields

TOPOLOGY_TYPES = [("Topology", "*.pdb *.gro *.tpr *.psf *.prmtop *.parm7 *.mol2 *.pqr *.crd"), ("All files", "*")]
TRAJECTORY_TYPES = [("Trajectory", "*.xtc *.trr *.dcd *.nc *.netcdf *.pdb *.xyz *.crd *.mdcrd"), ("All files", "*")]
JSONL_TYPES = [("Gephyra pathways", "*.jsonl"), ("All files", "*")]
JSON_TYPES = [("Gephyra clusters", "*.json"), ("All files", "*")]
VIS_INPUT_TYPES = [("Gephyra output", "*.jsonl *.json"), ("All files", "*")]
CSV_TYPES = [("CSV", "*.csv"), ("All files", "*")]

# Per stage: list of (section title, [(field, label, kind, extra, tooltip)]).
# Kinds: "text", "open" (file picker), "save" (file name in the output
# folder), "choice" (read-only combobox), "check" (on/off).
SECTIONS = {
    "calculate": [
        ("System", [
            ("topo", "Topology", "open", TOPOLOGY_TYPES, "Structure/topology (.prmtop, .pdb, .tpr, .psf, ...). Bond information enables the full angular H-bond criterion."),
            ("traj", "Trajectory", "open", TRAJECTORY_TYPES, "Required. Trajectory frames to analyse (.dcd, .xtc, .nc, ...). For a single structure, choose a .pdb."),
            ("root", "Root selection", "text", None, "MDAnalysis selection for the root atom(s), e.g. resname LIG and name O1"),
            ("water", "Water selection", "text", None, "MDAnalysis selection for bridging solvent atoms."),
        ]),
        ("Parameters", [
            ("stride", "Stride", "text", None, "Analyse every Nth frame (integer >= 1)."),
            ("max_depth", "Max depth", "text", None, "Maximum number of consecutive waters in a path (integer >= 1)."),
            ("min_depth", "Min depth", "text", None, "Shorter paths are not written (1 <= min depth <= max depth)."),
            ("cooperativity", "Cooperativity", "text", None, "Per-depth discount in (0, 1]; 1.0 disables cooperativity."),
            ("coarse_cutoff", "Coarse cutoff (Å)", "text", None, "Neighbour search radius for building the graph (> 0)."),
        ]),
        ("Output", [
            ("output", "Results file", "save", JSONL_TYPES, "Frame-by-frame path data (.jsonl), written to the output folder; input for clustering and visualization."),
            ("csv", "CSV summary", "save", CSV_TYPES, "Optional human-readable table of all paths."),
        ]),
    ],
    "cluster": [
        ("Input", [
            ("data", "Pathway data", "open", JSONL_TYPES, "The .jsonl written by step 1."),
        ]),
        ("Parameters", [
            ("threshold", "Fréchet threshold (Å)", "text", None, "Lower values give finer clusters; typical range 3-8 Å."),
            ("coarse_threshold", "Coarse threshold (Å)", "text", None, "Optional. 9D pre-screening threshold; default threshold/√3."),
            ("coarse_trigger", "Coarse trigger", "text", None, "Above this many unique paths the 9D pre-screening runs first."),
            ("max_paths", "Max paths", "text", None, "Memory cap: only the most frequent paths are clustered beyond this."),
        ]),
        ("Output", [
            ("output", "Cluster file", "save", JSON_TYPES, "Cluster summary (.json), written to the output folder; input for visualization."),
        ]),
    ],
    "visualize": [
        ("Input", [
            ("data", "Input data", "open", VIS_INPUT_TYPES, "Step-1 .jsonl for density/frame modes; step-2 cluster file for cluster mode."),
            ("format", "Viewer", "choice", forms.VISUALIZE_FORMATS, "chimerax writes a .cxc script; chimera/pymol a .py; vmd a .tcl (run with vmd -e)."),
            ("mode", "Mode", "choice", forms.VISUALIZE_MODES, "cluster: cluster medoids. frame: one frame with its real water molecules. density: every path of every frame."),
            ("frame", "Frame", "text", None, "Trajectory frame index (frame mode). Empty = first stored frame."),
            ("cluster_id", "Cluster ID", "text", None, "Cluster mode: draw a single cluster as real molecules at its medoid frame. Empty = several clusters."),
            ("top_clusters", "Top clusters", "text", None, "Cluster mode: number of most-occupied clusters to draw (empty = all)."),
        ]),
        ("Molecular scene", [
            ("with_structure", "Include protein, ligand and waters", "check", None, "The script loads a structure snapshot written by Gephyra, so the network is shown in its molecular context."),
            ("topo", "Topology", "open", TOPOLOGY_TYPES, "Only needed when the data file does not record it (files from Gephyra < 0.4)."),
            ("traj", "Trajectory", "open", TRAJECTORY_TYPES, "Only needed when the data file does not record it."),
            ("context", "Context selection", "text", None, "Optional MDAnalysis selection for the solute shown. Default: everything but solvent and counter-ions."),
        ]),
        ("Output", [
            ("max_bond_draw_dist", "Max segment length (Å)", "text", None, "Longer segments are treated as periodic-boundary artifacts and skipped."),
            ("chimerax_model", "ChimeraX model id", "text", None, "Optional. Scene scripts open the structure at #101 by default."),
            ("output", "Script name", "save", None, "File name in the output folder; the viewer's extension is added if missing."),
        ]),
    ],
}
FIELDS = {stage: [f for _, fields in SECTIONS[stage] for f in fields] for stage in SECTIONS}
# Short fields laid out two per row (consecutive ones are paired).
HALF_FIELDS = {"stride", "max_depth", "min_depth", "cooperativity", "coarse_cutoff",
               "threshold", "coarse_threshold", "coarse_trigger", "max_paths",
               "format", "mode", "frame", "cluster_id", "max_bond_draw_dist", "chimerax_model"}

STEP_TITLES = {"calculate": "Pathway Analysis", "cluster": "Clustering", "visualize": "Visualization"}
TAB_TITLES = {s: f"{i}. {t}" for i, (s, t) in enumerate(STEP_TITLES.items(), start=1)}
STEP_SUBTITLES = {
    "calculate": "Trace water-bridge chains from the root atoms through the trajectory.",
    "cluster": "Group recurring pathways into spatial channels (Fréchet distance).",
    "visualize": "Write a ready-to-run scene for ChimeraX, Chimera, PyMOL or VMD.",
}
RUN_LABELS = {"calculate": "Run analysis", "cluster": "Run clustering", "visualize": "Write script"}


class Cancelled(Exception):
    """Raised from the progress callback to stop a running stage."""


class _QueueLogHandler(logging.Handler):
    def __init__(self, q):
        super().__init__(level=logging.INFO)
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.q.put(("log", self.format(record), record.levelno, record.getMessage()))
        except Exception:  # pragma: no cover
            self.handleError(record)


def _pick_font(candidates, fallback):
    try:
        available = set(tkfont.families())
    except Exception:
        return fallback
    for name in candidates:
        if name in available:
            return name
    return fallback


def load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except OSError:
        pass


def open_in_file_manager(path):
    """Opens a folder (or a file's folder) in the system file manager."""
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if sys.platform.startswith("win"):
        os.startfile(folder)  # noqa: S606 - user-requested action on their own folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class _Tooltip:
    """Hover tooltip in the current theme."""

    def __init__(self, app, widget, text):
        self.app, self.widget, self.text, self.tip = app, widget, text, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event=None):
        if self.tip or not self.text:
            return
        p = self.app.palette
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background=p["surface2"], foreground=p["text"],
                 font=self.app.fonts["small"], wraplength=360, padx=10, pady=6,
                 highlightthickness=1, highlightbackground=p["border"]).pack()

    def _hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class GephyraApp:
    """
    The main window. `runner(stage, kwargs, progress_callback)` executes a
    stage; it defaults to `forms.run_stage` and can be replaced in tests.
    """

    POLL_MS = 100

    def __init__(self, master, runner=None, theme=None, settings_enabled=True):
        self.master = master
        self.runner = runner or forms.run_stage
        self.queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker = None
        self.current_stage = None
        self.error_records = []
        self.last_dir = os.getcwd()
        self.settings_enabled = settings_enabled
        self.settings = load_settings() if settings_enabled else {}
        self.theme_name = theme or self.settings.get("theme", "dark")
        if self.theme_name not in THEMES:
            self.theme_name = "dark"
        self.palette = THEMES[self.theme_name]

        self.vars = {stage: {} for stage in forms.STAGES}
        self.widgets = {stage: {} for stage in forms.STAGES}
        self.run_buttons = {}
        self.pages = {}
        self.canvases = []
        self.nav_buttons = {}
        self.active_stage = "calculate"
        self.outdir_var = tk.StringVar(value=os.getcwd())
        self.status_var = tk.StringVar(value="Ready.")
        self.log_visible = True

        self.log_handler = _QueueLogHandler(self.queue)
        pkg_logger = logging.getLogger("gephyra")
        pkg_logger.addHandler(self.log_handler)
        if pkg_logger.getEffectiveLevel() > logging.INFO:
            pkg_logger.setLevel(logging.INFO)  # keep a more verbose level (gephyra -v gui) if set

        master.title(f"Gephyra {__version__}")
        master.geometry("1120x780")
        master.minsize(980, 680)
        master.protocol("WM_DELETE_WINDOW", self.close)
        self._init_fonts()
        self.style = ttk.Style(master)
        self.style.theme_use("clam")
        self._apply_theme()
        self._build()
        self.show_stage("calculate")
        if self.core_fonts_only:
            self._append_log(XFT_HINT, logging.WARNING)
        master.bind("<Control-Return>", lambda _e: self.start(self.active_stage))
        self._poll_id = master.after(self.POLL_MS, self._poll)

    # ------------------------------------------------------------- theming
    def _init_fonts(self):
        ui = _pick_font(UI_FONTS, None)
        mono = _pick_font(MONO_FONTS, None)
        self.core_fonts_only = ui is None
        if self.core_fonts_only:
            ui = _pick_font(CORE_UI_FONTS, "TkDefaultFont")
            mono = _pick_font(CORE_MONO_FONTS, "TkFixedFont")
        # Negative sizes are pixels: crisp on bitmap fonts, consistent across DPI settings.
        self.fonts = {
            "base": (ui, -13), "small": (ui, -12), "label": (ui, -13), "section": (ui, -11, "bold"),
            "title": (ui, -24, "bold"), "subtitle": (ui, -13), "brand": (ui, -22, "bold"),
            "nav": (ui, -14), "button": (ui, -13, "bold"), "mono": (mono, -12), "dialog_title": (ui, -18, "bold"),
        }
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                tkfont.nametofont(name).configure(family=ui, size=-13)
            except tk.TclError:
                pass

    def _apply_theme(self):
        p = self.palette = THEMES[self.theme_name]
        st, f = self.style, self.fonts
        self.master.configure(background=p["bg"])
        st.configure(".", background=p["bg"], foreground=p["text"], font=f["base"], bordercolor=p["border"],
                     lightcolor=p["border"], darkcolor=p["border"], troughcolor=p["surface2"],
                     focuscolor=p["accent"], selectbackground=p["select"], selectforeground=p["text"],
                     insertcolor=p["text"])
        st.configure("TFrame", background=p["bg"])
        st.configure("Sidebar.TFrame", background=p["sidebar"])
        st.configure("Card.TFrame", background=p["surface"], bordercolor=p["border"], relief="solid", borderwidth=1)
        st.configure("Inner.TFrame", background=p["surface"])
        st.configure("TLabel", background=p["bg"], foreground=p["text"])
        st.configure("Title.TLabel", font=f["title"])
        st.configure("Subtitle.TLabel", foreground=p["muted"], font=f["subtitle"])
        st.configure("Status.TLabel", foreground=p["muted"], font=f["small"])
        st.configure("Brand.TLabel", background=p["sidebar"], foreground=p["text"], font=f["brand"])
        st.configure("BrandAccent.TLabel", background=p["sidebar"], foreground=p["accent"], font=f["brand"])
        st.configure("SideMuted.TLabel", background=p["sidebar"], foreground=p["muted"], font=f["small"])
        st.configure("SideSection.TLabel", background=p["sidebar"], foreground=p["muted"], font=f["section"])
        st.configure("SidePath.TLabel", background=p["sidebar"], foreground=p["text"], font=f["small"])
        st.configure("Field.TLabel", background=p["surface"], foreground=p["text"], font=f["label"])
        st.configure("Hint.TLabel", background=p["surface"], foreground=p["muted"], font=f["small"])
        st.configure("Section.TLabel", background=p["surface"], foreground=p["accent"], font=f["section"])

        st.configure("TEntry", fieldbackground=p["entry"], foreground=p["text"], bordercolor=p["border"],
                     lightcolor=p["border"], darkcolor=p["border"], padding=(8, 5), insertcolor=p["text"])
        st.map("TEntry", bordercolor=[("focus", p["accent"])], lightcolor=[("focus", p["accent"])],
               fieldbackground=[("disabled", p["surface2"])], foreground=[("disabled", p["muted"])])
        st.configure("TCombobox", fieldbackground=p["entry"], background=p["surface2"], foreground=p["text"],
                     arrowcolor=p["muted"], bordercolor=p["border"], lightcolor=p["border"], darkcolor=p["border"],
                     padding=(8, 4))
        st.map("TCombobox", fieldbackground=[("readonly", p["entry"])], foreground=[("readonly", p["text"])],
               bordercolor=[("focus", p["accent"])], arrowcolor=[("active", p["accent"])],
               selectbackground=[("readonly", p["entry"])], selectforeground=[("readonly", p["text"])])
        self.master.option_add("*TCombobox*Listbox.background", p["surface2"])
        self.master.option_add("*TCombobox*Listbox.foreground", p["text"])
        self.master.option_add("*TCombobox*Listbox.selectBackground", p["select"])
        self.master.option_add("*TCombobox*Listbox.selectForeground", p["text"])
        self.master.option_add("*TCombobox*Listbox.font", f["base"])
        st.configure("TCheckbutton", background=p["surface"], foreground=p["text"], indicatorbackground=p["entry"],
                     indicatorforeground=p["accent"], focuscolor=p["surface"])
        st.map("TCheckbutton", background=[("active", p["surface"])],
               indicatorbackground=[("selected", p["accent"]), ("active", p["surface2"])],
               indicatorforeground=[("selected", p["accent_text"])])

        st.configure("TButton", background=p["surface2"], foreground=p["text"], bordercolor=p["border"],
                     lightcolor=p["surface2"], darkcolor=p["surface2"], padding=(14, 7), font=f["base"],
                     focuscolor=p["surface2"])
        st.map("TButton", background=[("disabled", p["surface"]), ("active", p["border"])],
               foreground=[("disabled", p["muted"])], bordercolor=[("active", p["muted"])])
        st.configure("Accent.TButton", background=p["accent"], foreground=p["accent_text"], bordercolor=p["accent"],
                     lightcolor=p["accent"], darkcolor=p["accent"], font=f["button"], padding=(18, 8),
                     focuscolor=p["accent"])
        st.map("Accent.TButton", background=[("disabled", p["surface2"]), ("active", p["accent_hover"])],
               foreground=[("disabled", p["muted"])], bordercolor=[("active", p["accent_hover"])],
               lightcolor=[("active", p["accent_hover"])], darkcolor=[("active", p["accent_hover"])])
        st.configure("Small.TButton", padding=(10, 4), font=f["small"])
        st.configure("Side.TButton", background=p["sidebar"], bordercolor=p["border"], lightcolor=p["sidebar"],
                     darkcolor=p["sidebar"], padding=(10, 4), font=f["small"], focuscolor=p["sidebar"])
        st.map("Side.TButton", background=[("active", p["surface2"])])
        st.configure("Nav.TButton", background=p["sidebar"], foreground=p["muted"], bordercolor=p["sidebar"],
                     lightcolor=p["sidebar"], darkcolor=p["sidebar"], anchor="w", padding=(16, 10),
                     font=f["nav"], focuscolor=p["sidebar"])
        st.map("Nav.TButton", background=[("active", p["surface"])], foreground=[("active", p["text"])])
        st.configure("NavActive.TButton", background=p["surface2"], foreground=p["text"], bordercolor=p["surface2"],
                     lightcolor=p["accent"], darkcolor=p["surface2"], anchor="w", padding=(16, 10),
                     font=f["nav"], focuscolor=p["surface2"])
        st.map("NavActive.TButton", background=[("active", p["surface2"])])

        st.configure("Horizontal.TProgressbar", background=p["accent"], troughcolor=p["surface2"],
                     bordercolor=p["bg"], lightcolor=p["accent"], darkcolor=p["accent"], thickness=6)
        st.configure("Vertical.TScrollbar", background=p["surface2"], troughcolor=p["log_bg"],
                     bordercolor=p["log_bg"], arrowcolor=p["muted"], lightcolor=p["surface2"], darkcolor=p["surface2"])
        st.map("Vertical.TScrollbar", background=[("active", p["border"])])

        # Plain tk widgets do not follow ttk styles; recolour them by hand.
        for canvas in getattr(self, "canvases", []):
            canvas.configure(background=p["surface"])
        if hasattr(self, "log"):
            self._style_log()
        for nav in getattr(self, "nav_badges", {}).values():
            nav.configure(background=p["sidebar"])
        if hasattr(self, "nav_buttons") and self.nav_buttons:
            self._refresh_nav()

    def _style_log(self):
        p = self.palette
        self.log.configure(background=p["log_bg"], foreground=p["log_text"], insertbackground=p["text"],
                           selectbackground=p["select"], highlightthickness=0, borderwidth=0,
                           font=self.fonts["mono"], padx=12, pady=8)
        self.log.tag_configure("warning", foreground=p["warning"])
        self.log.tag_configure("error", foreground=p["error"])
        self.log.tag_configure("command", foreground=p["accent"])
        self.log.tag_configure("success", foreground=p["success"])

    def toggle_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self._apply_theme()
        self.theme_button.configure(text=self._theme_label())
        if self.settings_enabled:
            self.settings["theme"] = self.theme_name
            save_settings(self.settings)

    def _theme_label(self):
        return "Switch to light mode" if self.theme_name == "dark" else "Switch to dark mode"

    # -------------------------------------------------------------- layout
    def _build(self):
        root = ttk.Frame(self.master)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # Sidebar ---------------------------------------------------------
        side = ttk.Frame(root, style="Sidebar.TFrame", width=268)
        side.grid(row=0, column=0, sticky="nsw")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        brand = ttk.Frame(side, style="Sidebar.TFrame")
        brand.grid(row=0, column=0, sticky="ew", padx=20, pady=(22, 2))
        ttk.Label(brand, text="◆ ", style="BrandAccent.TLabel").pack(side="left")
        ttk.Label(brand, text="Gephyra", style="Brand.TLabel").pack(side="left")
        ttk.Label(side, text="Water-bridge network analysis", style="SideMuted.TLabel").grid(
            row=1, column=0, sticky="w", padx=22, pady=(0, 22))
        ttk.Label(side, text="WORKFLOW", style="SideSection.TLabel").grid(row=2, column=0, sticky="w", padx=22, pady=(0, 6))
        for i, stage in enumerate(forms.STAGES):
            b = ttk.Button(side, text=f"{i + 1}    {STEP_TITLES[stage]}", style="Nav.TButton",
                           command=lambda s=stage: self.show_stage(s))
            b.grid(row=3 + i, column=0, sticky="ew", padx=10, pady=1)
            self.nav_buttons[stage] = b

        side.rowconfigure(6, weight=1)
        ttk.Label(side, text="OUTPUT FOLDER", style="SideSection.TLabel").grid(row=7, column=0, sticky="w", padx=22, pady=(0, 4))
        self.outdir_label = ttk.Label(side, style="SidePath.TLabel", wraplength=224, justify="left")
        self.outdir_label.grid(row=8, column=0, sticky="ew", padx=22)
        self.outdir_var.trace_add("write", lambda *_: self._update_outdir_label())
        self._update_outdir_label()
        _Tooltip(self, self.outdir_label, "All results are saved here. Output names in the forms are relative to this "
                                          "folder; an absolute path in a field overrides it.")
        od = ttk.Frame(side, style="Sidebar.TFrame")
        od.grid(row=9, column=0, sticky="ew", padx=20, pady=(8, 18))
        ttk.Button(od, text="Change…", style="Side.TButton", command=self.choose_outdir).pack(side="left")
        ttk.Button(od, text="Open", style="Side.TButton", command=lambda: self._open_folder(self.outdir())).pack(side="left", padx=6)
        self.theme_button = ttk.Button(side, text=self._theme_label(), style="Side.TButton", command=self.toggle_theme)
        self.theme_button.grid(row=10, column=0, sticky="w", padx=20)
        ttk.Label(side, text=f"Version {__version__}   ·   Ctrl+Enter runs a step", style="SideMuted.TLabel").grid(
            row=11, column=0, sticky="w", padx=22, pady=(10, 18))

        # Main column -----------------------------------------------------
        main = ttk.Frame(root)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        pages = ttk.Frame(main)
        pages.grid(row=0, column=0, sticky="nsew", padx=28, pady=(22, 8))
        pages.columnconfigure(0, weight=1)
        pages.rowconfigure(0, weight=1)
        for stage in forms.STAGES:
            page = self._build_page(pages, stage)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[stage] = page

        # Activity bar + log ---------------------------------------------
        activity = ttk.Frame(main)
        activity.grid(row=1, column=0, sticky="ew", padx=28)
        activity.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(activity, mode="determinate", maximum=1.0)
        self.progress.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Label(activity, textvariable=self.status_var, style="Status.TLabel").grid(row=1, column=0, sticky="w")
        self.log_toggle = ttk.Button(activity, text="Hide log", style="Small.TButton", command=self.toggle_log)
        self.log_toggle.grid(row=1, column=1, padx=4)
        ttk.Button(activity, text="Clear", style="Small.TButton", command=self.clear_log).grid(row=1, column=2, padx=4)
        self.cancel_button = ttk.Button(activity, text="Cancel", style="Small.TButton", command=self.cancel, state="disabled")
        self.cancel_button.grid(row=1, column=3, padx=(4, 0))

        self.log_frame = ttk.Frame(main)
        self.log_frame.grid(row=2, column=0, sticky="nsew", padx=28, pady=(8, 20))
        self.log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(self.log_frame, height=7, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        self._style_log()

        v = self.vars["visualize"]
        for name in ("mode", "format", "with_structure", "data"):
            v[name].trace_add("write", lambda *_: self._update_visualize_state())
        self._update_visualize_state()

    def _build_page(self, parent, stage):
        page = ttk.Frame(parent)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(2, weight=1)
        ttk.Label(page, text=STEP_TITLES[stage], style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(page, text=STEP_SUBTITLES[stage], style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 14))

        # The card scrolls when the window is too short for the whole form.
        holder = ttk.Frame(page, style="Card.TFrame")
        holder.grid(row=2, column=0, sticky="nsew")
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        canvas = tk.Canvas(holder, highlightthickness=0, borderwidth=0, background=self.palette["surface"])
        canvas.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        bar = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        card = ttk.Frame(canvas, style="Inner.TFrame", padding=(24, 18))
        window = canvas.create_window(0, 0, window=card, anchor="nw")
        self.canvases.append(canvas)

        def refresh(_event=None):
            canvas.configure(scrollregion=(0, 0, card.winfo_reqwidth(), card.winfo_reqheight()))
            canvas.itemconfigure(window, width=canvas.winfo_width())
            if card.winfo_reqheight() > canvas.winfo_height() + 2:
                bar.grid(row=0, column=1, sticky="ns")
            else:
                bar.grid_remove()
                canvas.yview_moveto(0)
        card.bind("<Configure>", refresh)
        canvas.bind("<Configure>", refresh)

        def wheel(event):
            if bar.winfo_ismapped():
                step = -1 if (getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0) else 1
                canvas.yview_scroll(step, "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            card.bind_all(seq, lambda e, w=wheel, s=stage: w(e) if self.active_stage == s else None, add="+")

        # Grid: 0 label | 1 entry | 2 label | 3 entry | 4 browse
        card.columnconfigure(1, weight=1)
        card.columnconfigure(3, weight=1)
        defaults = forms.DEFAULTS[stage]
        row = 0
        for title, fields in SECTIONS[stage]:
            ttk.Label(card, text=title.upper(), style="Section.TLabel").grid(
                row=row, column=0, columnspan=5, sticky="w", pady=(14 if row else 0, 6))
            row += 1
            col = 0
            for i, (field, label, kind, extra, tip) in enumerate(fields):
                half = field in HALF_FIELDS
                if not half or col == 0:
                    if col:
                        row += 1
                    col = 0
                var = tk.StringVar(value=defaults.get(field, ""))
                self.vars[stage][field] = var
                lc, ec = (0, 1) if col == 0 else (2, 3)
                if kind == "check":
                    widget = ttk.Checkbutton(card, text=label, variable=var, onvalue="1", offvalue="0")
                    widget.grid(row=row, column=1, columnspan=4, sticky="w", pady=5)
                else:
                    ttk.Label(card, text=label, style="Field.TLabel").grid(
                        row=row, column=lc, sticky="w", padx=(0 if lc == 0 else 20, 14), pady=5)
                    if kind == "choice":
                        widget = ttk.Combobox(card, textvariable=var, values=list(extra), state="readonly", width=14)
                    else:
                        widget = ttk.Entry(card, textvariable=var, width=14 if half else 40)
                    span = 1 if half else 3
                    widget.grid(row=row, column=ec, columnspan=span, sticky="ew", pady=5)
                    if kind in ("open", "save"):
                        ttk.Button(card, text="Browse", style="Small.TButton",
                                   command=lambda s=stage, f=field, k=kind, t=extra: self._browse(s, f, k, t)
                                   ).grid(row=row, column=4, padx=(10, 0), pady=5)
                self.widgets[stage][field] = widget
                _Tooltip(self, widget, tip)
                if half and col == 0:
                    col = 1
                else:
                    col = 0
                    row += 1
            if col:
                row += 1
            if stage == "visualize" and title == "Molecular scene":
                self.structure_hint = ttk.Label(card, text="", style="Hint.TLabel", wraplength=620, justify="left")
                self.structure_hint.grid(row=row, column=1, columnspan=4, sticky="w", pady=(0, 2))
                row += 1

        buttons = ttk.Frame(page)
        buttons.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(buttons, text="Reset defaults", command=lambda s=stage: self.reset(s)).pack(side="left")
        run = ttk.Button(buttons, text=RUN_LABELS[stage] + "  →", style="Accent.TButton", command=lambda s=stage: self.start(s))
        run.pack(side="right")
        self.run_buttons[stage] = run
        ttk.Button(buttons, text="Copy CLI command", command=lambda s=stage: self.copy_command(s)).pack(side="right", padx=8)
        return page

    def show_stage(self, stage):
        self.active_stage = stage
        self.pages[stage].tkraise()
        self._refresh_nav()

    def _refresh_nav(self):
        for stage, b in self.nav_buttons.items():
            b.configure(style="NavActive.TButton" if stage == self.active_stage else "Nav.TButton")

    def _update_outdir_label(self):
        path = self.outdir_var.get()
        home = os.path.expanduser("~")
        shown = "~" + path[len(home):] if path.startswith(home) else path
        self.outdir_label.configure(text=shown)

    def _update_visualize_state(self):
        v, w = self.vars["visualize"], self.widgets["visualize"]
        mode = v["mode"].get()
        w["frame"].configure(state="normal" if mode == "frame" else "disabled")
        w["cluster_id"].configure(state="normal" if mode == "cluster" else "disabled")
        w["top_clusters"].configure(state="normal" if mode == "cluster" else "disabled")
        w["chimerax_model"].configure(state="normal" if v["format"].get() == "chimerax" else "disabled")
        scene = v["with_structure"].get() == "1"
        for name in ("topo", "traj", "context"):
            w[name].configure(state="normal" if scene else "disabled")
        data = v["data"].get().strip()
        hint = forms.describe_structure(data) if scene and data and os.path.isfile(data) else ""
        if not scene:
            hint = "Geometry only: load the analysed structure in the viewer before running the script."
        if hasattr(self, "structure_hint"):
            self.structure_hint.configure(text=hint)

    # ------------------------------------------------------------- actions
    def outdir(self):
        return forms.resolve_outdir(self.outdir_var.get())

    def choose_outdir(self):
        path = filedialog.askdirectory(parent=self.master, initialdir=self.outdir(), title="Choose the output folder")
        if path:
            self.outdir_var.set(path)
            self._append_log(f"Output folder: {path}", logging.INFO)

    def _browse(self, stage, field, kind, filetypes):
        var = self.vars[stage][field]
        current = var.get().strip()
        if kind == "save":
            initialdir = os.path.dirname(current) if os.path.isabs(current) else self.outdir()
            path = filedialog.asksaveasfilename(parent=self.master, initialdir=initialdir,
                                                initialfile=os.path.basename(current) if current else "",
                                                filetypes=filetypes or [("All files", "*")])
            if path:
                # Inside the output folder -> keep just the name, so it follows the folder.
                rel = os.path.relpath(path, self.outdir())
                var.set(rel if not rel.startswith("..") and not os.path.isabs(rel) else path)
            return
        initialdir = os.path.dirname(os.path.abspath(current)) if current else self.last_dir
        path = filedialog.askopenfilename(parent=self.master, initialdir=initialdir,
                                          filetypes=filetypes or [("All files", "*")])
        if path:
            var.set(path)
            self.last_dir = os.path.dirname(path)

    def values(self, stage):
        return {field: var.get() for field, var in self.vars[stage].items()}

    def collect(self, stage):
        """Parsed kwargs for `stage`, or None after showing a validation dialog."""
        try:
            return forms.parse_stage(stage, self.values(stage), self.outdir_var.get())
        except forms.FormError as exc:
            if exc.code == "missing_dir":
                folder = self.outdir()
                if self.ask_yes_no("Create output folder?", f"{folder}\n\ndoes not exist. Create it?"):
                    try:
                        os.makedirs(folder, exist_ok=True)
                    except OSError as err:
                        self.show_error("Could not create folder", str(err))
                        return None
                    return self.collect(stage)
                return None
            self.show_stage(stage)
            widget = self.widgets[stage].get(exc.field)
            if widget is not None:
                widget.focus_set()
            self.show_error("Check this field", exc.message)
            return None

    def reset(self, stage):
        for field, var in self.vars[stage].items():
            var.set(forms.DEFAULTS[stage].get(field, ""))

    def copy_command(self, stage):
        kwargs = self.collect(stage)
        if kwargs is None:
            return
        cmd = forms.build_command_line(stage, kwargs)
        self.master.clipboard_clear()
        self.master.clipboard_append(cmd)
        self._append_log(f"Command copied to clipboard:\n  {cmd}", logging.INFO, tag="command")

    def start(self, stage):
        if self.worker is not None and self.worker.is_alive():
            self.show_info("Busy", "A task is already running. Wait for it to finish or cancel it.")
            return
        kwargs = self.collect(stage)
        if kwargs is None:
            return

        self.current_stage = stage
        self.error_records = []
        self.cancel_event.clear()
        self._set_running(True)
        self.progress.configure(mode="indeterminate" if stage == "visualize" else "determinate", value=0)
        if stage == "visualize":
            self.progress.start(15)
        self.status_var.set(f"Running {STEP_TITLES[stage].lower()}…")
        self._append_log(f"$ {forms.build_command_line(stage, kwargs)}", logging.INFO, tag="command")

        self.worker = threading.Thread(target=self._work, args=(stage, kwargs), daemon=True)
        self.worker.start()

    def cancel(self):
        if self.worker is not None and self.worker.is_alive():
            self.cancel_event.set()
            self.status_var.set("Cancelling after the current step…")

    def _work(self, stage, kwargs):
        def progress(done, total):
            self.queue.put(("progress", done, total))
            if self.cancel_event.is_set():
                raise Cancelled()

        try:
            result = self.runner(stage, kwargs, progress_callback=progress)
        except Cancelled:
            self.queue.put(("done", stage, kwargs, "cancelled", None, None))
        except Exception as exc:  # reported in a dialog, not as a traceback
            logger.debug("Stage %s failed:\n%s", stage, traceback.format_exc().rstrip())  # visible with gephyra -v gui
            message = str(exc) or exc.__class__.__name__
            self.queue.put(("done", stage, kwargs, "error", f"{exc.__class__.__name__}: {message}", None))
        else:
            self.queue.put(("done", stage, kwargs, "ok", None, result))

    # ---------------------------------------------------- main-thread side
    def _poll(self):
        try:
            while True:
                self._handle(self.queue.get_nowait())
        except queue.Empty:
            pass
        self._poll_id = self.master.after(self.POLL_MS, self._poll)

    def _handle(self, item):
        kind = item[0]
        if kind == "log":
            _, text, level, message = item
            if level >= logging.ERROR and self.current_stage is not None:
                self.error_records.append(message)
            if level >= logging.INFO:
                self._append_log(text, level)
        elif kind == "progress":
            _, done, total = item
            if total and self.current_stage:
                self.progress.configure(value=done / total)
                self.status_var.set(f"Running {STEP_TITLES[self.current_stage].lower()}…  {done} / {total}")
        elif kind == "done":
            self._finish(*item[1:])

    def _finish(self, stage, kwargs, outcome, message, result=None):
        self.progress.stop()
        self._set_running(False)
        self.current_stage = None

        if outcome == "cancelled":
            self.progress.configure(value=0)
            self.status_var.set("Cancelled. Output files keep what was written before cancelling.")
            self._append_log("Cancelled by user.", logging.WARNING)
            return
        if outcome == "error" or self.error_records or (stage == "visualize" and not result):
            self.progress.configure(value=0)
            detail = message or "\n".join(self.error_records) or "No script was written; see the log for details."
            self.status_var.set(f"{STEP_TITLES[stage]} failed.")
            self.show_error(f"{STEP_TITLES[stage]} failed", detail)
            return

        self.progress.configure(mode="determinate", value=1.0)
        self._on_success(stage, kwargs, result)

    def _on_success(self, stage, kwargs, result):
        files = forms.result_files(stage, kwargs, result)
        folder = os.path.dirname(files[0]) if files else self.outdir()
        for path in files:
            self._append_log(f"Saved: {path}", logging.INFO, tag="success")
        next_action = None
        if stage == "calculate":
            out = kwargs["output_file"]
            self.vars["cluster"]["data"].set(out)
            if self.vars["visualize"]["mode"].get() != "cluster":
                self.vars["visualize"]["data"].set(out)
            summary = result or {}
            headline = "Analysis complete"
            body = (f"{summary.get('n_paths', '?')} water-bridge paths found in {summary.get('n_frames', '?')} frames."
                    if summary else "The pathway data was written successfully.")
            next_action = ("Next: Clustering  →", lambda: self.show_stage("cluster"))
        elif stage == "cluster":
            out = kwargs["output_file"]
            self.vars["visualize"]["data"].set(out)
            self.vars["visualize"]["mode"].set("cluster")
            headline = "Clustering complete"
            n = (result or {}).get("n_clusters")
            body = f"{n} recurring water-bridge channels identified." if n is not None else "Clusters written successfully."
            next_action = ("Next: Visualization  →", lambda: self.show_stage("visualize"))
        else:
            script = files[0] if files else forms.visualize_output_path(kwargs)
            headline = "Script ready"
            body = "Open it with:\n" + USAGE[kwargs["format"]].format(script=os.path.basename(script))
            if kwargs["format"] == "vmd":
                body += ("\n\nThe -e is required: 'vmd " + os.path.basename(script) + "' without it makes VMD read "
                         "the script as a molecule and nothing is drawn.")
            if kwargs.get("with_structure") and len(files) > 1 and files[-1].endswith("_structure.pdb"):
                body += "\n\nThe script loads the protein, ligand and water network by itself."
            self._append_log("Open it with: " + USAGE[kwargs["format"]].format(script=script), logging.INFO, tag="command")
        self.status_var.set(f"{headline}. Saved in {folder}")
        self.show_success(headline, body, files, folder, next_action)

    def _set_running(self, running):
        for button in self.run_buttons.values():
            button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")

    def _append_log(self, text, level, tag=None):
        if tag is None:
            tag = "error" if level >= logging.ERROR else "warning" if level >= logging.WARNING else ""
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def toggle_log(self):
        self.log_visible = not self.log_visible
        if self.log_visible:
            self.log_frame.grid()
        else:
            self.log_frame.grid_remove()
        self.log_toggle.configure(text="Hide log" if self.log_visible else "Show log")

    def _open_folder(self, path):
        try:
            open_in_file_manager(path)
        except Exception as exc:
            self.show_error("Could not open folder", f"{path}\n\n{exc}")

    def _copy_files_to(self, files):
        target = filedialog.askdirectory(parent=self.master, initialdir=self.outdir(), title="Save a copy of the results to…")
        if not target:
            return
        copied = []
        for path in files:
            try:
                copied.append(shutil.copy2(path, target))
            except OSError as exc:
                self.show_error("Copy failed", f"{path}\n\n{exc}")
                return
        for path in copied:
            self._append_log(f"Copied: {path}", logging.INFO, tag="success")
        self.status_var.set(f"Copied {len(copied)} file(s) to {target}")

    # ------------------------------------------------------------- dialogs
    def _dialog(self, title, message, kind="info", files=None, buttons=(("OK", None, True),)):
        """
        Themed modal dialog. `buttons` are (label, callback, closes) tuples;
        returns the label of the button that closed it.
        """
        p, f = self.palette, self.fonts
        top = tk.Toplevel(self.master)
        top.title(title)
        top.configure(background=p["surface"])
        top.transient(self.master)
        top.resizable(False, False)
        colour = {"success": p["success"], "error": p["error"]}.get(kind, p["accent"])
        body = tk.Frame(top, background=p["surface"], padx=28, pady=22)
        body.pack(fill="both", expand=True)
        head = tk.Frame(body, background=p["surface"])
        head.pack(fill="x")
        # The icon is drawn, not a font glyph: core-font Tk builds lack check marks.
        icon = tk.Canvas(head, width=30, height=30, background=p["surface"], highlightthickness=0)
        icon.create_oval(2, 2, 28, 28, fill=colour, outline=colour)
        ink = p["surface"]
        if kind == "success":
            icon.create_line(9, 15, 13, 20, 21, 10, fill=ink, width=3, capstyle="round", joinstyle="round")
        elif kind == "error":
            icon.create_line(15, 8, 15, 17, fill=ink, width=3, capstyle="round")
            icon.create_oval(13, 20, 17, 24, fill=ink, outline=ink)
        else:
            icon.create_line(15, 13, 15, 22, fill=ink, width=3, capstyle="round")
            icon.create_oval(13, 6, 17, 10, fill=ink, outline=ink)
        icon.pack(side="left")
        tk.Label(head, text=title, background=p["surface"], foreground=p["text"], font=f["dialog_title"],
                 padx=12).pack(side="left")
        tk.Label(body, text=message, background=p["surface"], foreground=p["text"], font=f["base"],
                 justify="left", wraplength=520, anchor="w").pack(fill="x", pady=(14, 6))
        if files:
            tk.Label(body, text="SAVED FILES", background=p["surface"], foreground=p["muted"],
                     font=f["section"], anchor="w").pack(fill="x", pady=(8, 2))
            box = tk.Text(body, height=min(6, len(files)), width=60, background=p["log_bg"], foreground=p["log_text"],
                          font=f["mono"], borderwidth=0, highlightthickness=1, highlightbackground=p["border"],
                          padx=8, pady=6, wrap="none")
            box.insert("1.0", "\n".join(os.path.basename(x) for x in files))
            box.configure(state="disabled")
            box.pack(fill="x")
        result = {"value": None}
        row = tk.Frame(body, background=p["surface"])
        row.pack(fill="x", pady=(18, 0))

        def make(label, callback, closes):
            def handler():
                if callback:
                    callback()
                if closes:
                    result["value"] = label
                    top.destroy()
            return handler

        for i, (label, callback, closes) in enumerate(reversed(buttons)):
            style = "Accent.TButton" if i == 0 else "TButton"
            ttk.Button(row, text=label, style=style, command=make(label, callback, closes)).pack(side="right", padx=(8, 0))
        top.bind("<Escape>", lambda _e: top.destroy())
        top.update_idletasks()
        x = self.master.winfo_rootx() + (self.master.winfo_width() - top.winfo_width()) // 2
        y = self.master.winfo_rooty() + (self.master.winfo_height() - top.winfo_height()) // 3
        top.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        try:
            top.grab_set()
        except tk.TclError:
            pass
        self.master.wait_window(top)
        return result["value"]

    def show_error(self, title, message):
        return self._dialog(title, message, kind="error")

    def show_info(self, title, message):
        return self._dialog(title, message, kind="info")

    def ask_yes_no(self, title, message):
        return self._dialog(title, message, kind="question",
                            buttons=(("No", None, True), ("Yes", None, True))) == "Yes"

    def show_success(self, title, message, files, folder, next_action=None):
        message = f"{message}\n\nSaved in: {folder}"
        buttons = [("Open folder", lambda: self._open_folder(folder), False)]
        if files:
            buttons.append(("Save a copy to…", lambda: self._copy_files_to(files), False))
        buttons.append(("Close", None, True))
        if next_action:
            buttons.append((next_action[0], next_action[1], True))
        return self._dialog(title, message, kind="success", files=files, buttons=tuple(buttons))

    def close(self):
        if self.worker is not None and self.worker.is_alive():
            if not self.ask_yes_no("Task running", "A task is still running. Quit anyway?"):
                return
            self.cancel_event.set()
        logging.getLogger("gephyra").removeHandler(self.log_handler)
        try:
            self.master.after_cancel(self._poll_id)
        except Exception:
            pass
        self.master.destroy()


def main(argv=None):
    """Entry point for `gephyra gui` and `gephyra-gui`. Returns a process exit code."""
    if tk is None:
        print("The Gephyra interface needs tkinter, which this Python was built without. "
              "On Debian/Ubuntu install it with: sudo apt install python3-tk")
        return 1
    try:
        root = tk.Tk(className="Gephyra")
    except tk.TclError as exc:
        print(f"Could not open a window ({exc}). The interface needs a graphical display; "
              "use the 'gephyra' command line instead.")
        return 1
    GephyraApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
