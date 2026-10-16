"""
Desktop interface for the Gephyra workflow, built on the standard-library
tkinter/ttk toolkit so it adds no dependencies.

Three tabs mirror the pipeline: 1. Pathway Analysis (`calculate`),
2. Clustering (`cluster`) and 3. Visualization (`visualize`). Form values are
parsed and validated by `gephyra.forms`, the same rules the command line
applies; this module only handles widgets, threading and dialogs.

Stages run on a worker thread. Log records and progress updates are passed to
the Tk main loop through a queue, so the window stays responsive and the log
pane updates in real time. Launch with `gephyra gui` or `gephyra-gui`.
"""

import logging
import os
import queue
import threading
import traceback

from . import forms
from ._version import __version__

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError:  # pragma: no cover - depends on the Python build
    tk = None

logger = logging.getLogger(__name__)

TOPOLOGY_TYPES = [("Topology", "*.pdb *.gro *.tpr *.psf *.prmtop *.parm7 *.mol2 *.pqr *.crd"), ("All files", "*")]
TRAJECTORY_TYPES = [("Trajectory", "*.xtc *.trr *.dcd *.nc *.netcdf *.pdb *.xyz *.crd *.mdcrd"), ("All files", "*")]
JSONL_TYPES = [("Gephyra pathways", "*.jsonl"), ("All files", "*")]
JSON_TYPES = [("Gephyra clusters", "*.json"), ("All files", "*")]
VIS_INPUT_TYPES = [("Gephyra output", "*.jsonl *.json"), ("All files", "*")]
CSV_TYPES = [("CSV", "*.csv"), ("All files", "*")]

# (field, label, widget kind, file types or choices, tooltip)
# widget kinds: "text", "open", "save", "choice"
FIELDS = {
    "calculate": [
        ("topo", "Topology file", "open", TOPOLOGY_TYPES, "Structure/topology (.pdb, .tpr, .psf, ...). Bond information enables the full angular H-bond criterion."),
        ("traj", "Trajectory (optional)", "open", TRAJECTORY_TYPES, "Leave empty to analyse the topology coordinates only."),
        ("root", "Root selection", "text", None, "MDAnalysis selection for the root atom(s), e.g. resname LIG and name O1"),
        ("water", "Water selection", "text", None, "MDAnalysis selection for bridging solvent atoms."),
        ("stride", "Stride", "text", None, "Analyse every Nth frame (integer >= 1)."),
        ("max_depth", "Max depth", "text", None, "Maximum number of consecutive waters in a path (integer >= 1)."),
        ("min_depth", "Min depth", "text", None, "Shorter paths are not written (1 <= min depth <= max depth)."),
        ("cooperativity", "Cooperativity", "text", None, "Per-depth discount in (0, 1]; 1.0 disables cooperativity."),
        ("coarse_cutoff", "Coarse cutoff (Å)", "text", None, "Neighbour search radius for building the graph (> 0)."),
        ("output", "Output JSONL", "save", JSONL_TYPES, "Frame-by-frame path data; input for clustering and visualization."),
        ("csv", "CSV summary (optional)", "save", CSV_TYPES, "Optional human-readable table of all paths."),
    ],
    "cluster": [
        ("data", "Pathway data (.jsonl)", "open", JSONL_TYPES, "The .jsonl written by step 1."),
        ("threshold", "Fréchet threshold (Å)", "text", None, "Lower values give finer clusters; typical range 3-8 Å."),
        ("coarse_threshold", "Coarse threshold (Å)", "text", None, "Optional. 9D pre-screening threshold; default threshold/√3."),
        ("coarse_trigger", "Coarse trigger", "text", None, "Above this many unique paths the 9D pre-screening runs first."),
        ("max_paths", "Max paths", "text", None, "Memory cap: only the most frequent paths are clustered beyond this."),
        ("output", "Output JSON", "save", JSON_TYPES, "Cluster summary; input for visualization in 'cluster' mode."),
    ],
    "visualize": [
        ("data", "Input data", "open", VIS_INPUT_TYPES, "Step-1 .jsonl for density/frame modes; step-2 .json for cluster mode."),
        ("format", "Viewer", "choice", forms.VISUALIZE_FORMATS, "chimerax writes a .cxc command script; chimera/pymol a .py; vmd a .tcl."),
        ("mode", "Mode", "choice", forms.VISUALIZE_MODES, "cluster: medoid of each cluster. density: every path of every frame. frame: one frame."),
        ("frame", "Frame", "text", None, "Trajectory frame index (frame mode). Empty = first stored frame."),
        ("cluster_id", "Cluster ID", "text", None, "Draw a single cluster (cluster mode). Empty = all clusters."),
        ("chimerax_model", "ChimeraX model", "text", None, "Model id of your structure in ChimeraX, e.g. #1."),
        ("max_bond_draw_dist", "Max segment length (Å)", "text", None, "Longer segments are treated as periodic-boundary artifacts and skipped."),
        ("output", "Output script", "save", None, "File name or prefix; the viewer's extension is added if missing."),
    ],
}

TAB_TITLES = {"calculate": "1. Pathway Analysis", "cluster": "2. Clustering", "visualize": "3. Visualization"}
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


class _Tooltip:
    """Minimal hover tooltip for a widget."""

    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0", relief="solid",
                 borderwidth=1, wraplength=380, padx=6, pady=3).pack()

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

    def __init__(self, master, runner=None):
        self.master = master
        self.runner = runner or forms.run_stage
        self.queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker = None
        self.current_stage = None
        self.error_records = []
        self.last_dir = os.getcwd()

        self.vars = {stage: {} for stage in forms.STAGES}
        self.widgets = {stage: {} for stage in forms.STAGES}
        self.run_buttons = {}

        self.log_handler = _QueueLogHandler(self.queue)
        pkg_logger = logging.getLogger("gephyra")
        pkg_logger.addHandler(self.log_handler)
        if pkg_logger.getEffectiveLevel() > logging.INFO:
            pkg_logger.setLevel(logging.INFO)  # keep a more verbose level (gephyra -v gui) if set

        master.title(f"Gephyra {__version__}")
        style = ttk.Style(master)
        if style.theme_use() == "default" and "clam" in style.theme_names():
            style.theme_use("clam")  # the X11 default theme looks dated; Windows/macOS keep their native one
            master.update_idletasks()  # deliver <<ThemeChanged>> now, not after a quick destroy
        master.minsize(760, 620)
        master.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self._poll_id = master.after(self.POLL_MS, self._poll)

    # ------------------------------------------------------------------ layout
    def _build(self):
        outer = ttk.Frame(self.master, padding=8)
        outer.pack(fill="both", expand=True)

        paned = ttk.PanedWindow(outer, orient="vertical")
        paned.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(paned)
        for stage in forms.STAGES:
            self.notebook.add(self._build_tab(stage), text=TAB_TITLES[stage])
        paned.add(self.notebook, weight=3)

        bottom = ttk.Frame(paned)
        status_row = ttk.Frame(bottom)
        status_row.pack(fill="x", pady=(6, 4))
        self.status_var = tk.StringVar(value="Ready.")
        # Pack the fixed-size widgets first so a long status never pushes them out of view.
        self.cancel_button = ttk.Button(status_row, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="right")
        self.progress = ttk.Progressbar(status_row, length=220, mode="determinate", maximum=1.0)
        self.progress.pack(side="right", padx=8)
        ttk.Label(status_row, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)

        self.log = ScrolledText(bottom, height=10, state="disabled", wrap="word", font=("TkFixedFont", 9))
        self.log.tag_configure("warning", foreground="#a15c00")
        self.log.tag_configure("error", foreground="#b00020")
        self.log.pack(fill="both", expand=True)
        log_buttons = ttk.Frame(bottom)
        log_buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(log_buttons, text="Clear log", command=self.clear_log).pack(side="right")
        paned.add(bottom, weight=2)

        self.vars["visualize"]["mode"].trace_add("write", lambda *_: self._update_visualize_state())
        self.vars["visualize"]["format"].trace_add("write", lambda *_: self._update_visualize_state())
        self._update_visualize_state()

    def _build_tab(self, stage):
        frame = ttk.Frame(self.notebook, padding=10)
        frame.columnconfigure(1, weight=1)
        defaults = forms.DEFAULTS[stage]

        for row, (field, label, kind, extra, tip) in enumerate(FIELDS[stage]):
            var = tk.StringVar(value=defaults.get(field, ""))
            self.vars[stage][field] = var
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            if kind == "choice":
                widget = ttk.Combobox(frame, textvariable=var, values=list(extra), state="readonly", width=18)
                widget.grid(row=row, column=1, sticky="w", pady=3)
            else:
                widget = ttk.Entry(frame, textvariable=var)
                widget.grid(row=row, column=1, sticky="ew", pady=3)
            if kind in ("open", "save"):
                ttk.Button(frame, text="Browse…", width=9,
                           command=lambda s=stage, f=field, k=kind, t=extra: self._browse(s, f, k, t)
                           ).grid(row=row, column=2, padx=(6, 0), pady=3)
            self.widgets[stage][field] = widget
            _Tooltip(widget, tip)

        buttons = ttk.Frame(frame)
        buttons.grid(row=len(FIELDS[stage]), column=0, columnspan=3, sticky="ew", pady=(12, 0))
        run = ttk.Button(buttons, text=RUN_LABELS[stage], command=lambda s=stage: self.start(s))
        run.pack(side="right")
        self.run_buttons[stage] = run
        ttk.Button(buttons, text="Copy CLI command", command=lambda s=stage: self.copy_command(s)).pack(side="right", padx=6)
        ttk.Button(buttons, text="Reset defaults", command=lambda s=stage: self.reset(s)).pack(side="left")
        return frame

    def _update_visualize_state(self):
        v, w = self.vars["visualize"], self.widgets["visualize"]
        w["frame"].configure(state="normal" if v["mode"].get() == "frame" else "disabled")
        w["cluster_id"].configure(state="normal" if v["mode"].get() == "cluster" else "disabled")
        w["chimerax_model"].configure(state="normal" if v["format"].get() == "chimerax" else "disabled")

    # ---------------------------------------------------------------- actions
    def _browse(self, stage, field, kind, filetypes):
        var = self.vars[stage][field]
        current = var.get().strip()
        initialdir = os.path.dirname(os.path.abspath(current)) if current else self.last_dir
        options = {"parent": self.master, "initialdir": initialdir, "filetypes": filetypes or [("All files", "*")]}
        if kind == "open":
            path = filedialog.askopenfilename(**options)
        else:
            path = filedialog.asksaveasfilename(initialfile=os.path.basename(current) if current else "", **options)
        if path:
            var.set(path)
            self.last_dir = os.path.dirname(path)

    def values(self, stage):
        return {field: var.get() for field, var in self.vars[stage].items()}

    def collect(self, stage):
        """Parsed kwargs for `stage`, or None after showing a validation dialog."""
        try:
            return forms.parse_stage(stage, self.values(stage))
        except forms.FormError as exc:
            self.notebook.select(forms.STAGES.index(stage))
            widget = self.widgets[stage].get(exc.field)
            if widget is not None:
                widget.focus_set()
            messagebox.showerror("Invalid input", exc.message, parent=self.master)
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
        self._append_log(f"Command copied to clipboard:\n  {cmd}", logging.INFO)

    def start(self, stage):
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Busy", "A task is already running. Wait for it to finish or cancel it.", parent=self.master)
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
        self.status_var.set(f"Running {TAB_TITLES[stage]}…")
        self._append_log(f"$ {forms.build_command_line(stage, kwargs)}", logging.INFO)

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
            self.runner(stage, kwargs, progress_callback=progress)
        except Cancelled:
            self.queue.put(("done", stage, kwargs, "cancelled", None))
        except Exception as exc:  # reported in a dialog, not as a traceback
            logger.debug("Stage %s failed:\n%s", stage, traceback.format_exc().rstrip())  # visible with gephyra -v gui
            message = str(exc) or exc.__class__.__name__
            self.queue.put(("done", stage, kwargs, "error", f"{exc.__class__.__name__}: {message}"))
        else:
            self.queue.put(("done", stage, kwargs, "ok", None))

    # ------------------------------------------------------- main-thread side
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
            if total:
                self.progress.configure(value=done / total)
                self.status_var.set(f"Running {TAB_TITLES[self.current_stage]}… {done}/{total}")
        elif kind == "done":
            self._finish(*item[1:])

    def _finish(self, stage, kwargs, outcome, message):
        self.progress.stop()
        self._set_running(False)
        self.current_stage = None

        if outcome == "cancelled":
            self.progress.configure(value=0)
            self.status_var.set("Cancelled. Output files keep what was written before cancelling.")
            self._append_log("Cancelled by user.", logging.WARNING)
            return
        if outcome == "error" or self.error_records:
            self.progress.configure(value=0)
            detail = message or "\n".join(self.error_records)
            self.status_var.set(f"{TAB_TITLES[stage]} failed.")
            messagebox.showerror(f"{TAB_TITLES[stage]} failed", detail, parent=self.master)
            return

        self.progress.configure(mode="determinate", value=1.0)
        self._on_success(stage, kwargs)

    def _on_success(self, stage, kwargs):
        """Pre-fill the next stage with this stage's output."""
        if stage == "calculate":
            out = kwargs["output_file"]
            self.vars["cluster"]["data"].set(out)
            self.status_var.set(f"Analysis complete: {os.path.basename(out)}. Next: 2. Clustering.")
        elif stage == "cluster":
            out = kwargs["output_file"]
            self.vars["visualize"]["data"].set(out)
            self.vars["visualize"]["mode"].set("cluster")
            self.status_var.set(f"Clustering complete: {os.path.basename(out)}. Next: 3. Visualization.")
        else:
            out = forms.visualize_output_path(kwargs)
            hint = {"chimerax": "In ChimeraX: open your structure, then 'open {}'",
                    "chimera": "In Chimera: open your structure, then run {}",
                    "pymol": "In PyMOL: load your structure, then 'run {}'",
                    "vmd": "In VMD: load your structure, then 'source {}'"}[kwargs["format"]]
            self.status_var.set(f"Script written: {os.path.basename(out)}")
            self._append_log(hint.format(out), logging.INFO)

    def _set_running(self, running):
        for button in self.run_buttons.values():
            button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")

    def _append_log(self, text, level):
        tag = "error" if level >= logging.ERROR else "warning" if level >= logging.WARNING else ""
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def close(self):
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno("Task running", "A task is still running. Quit anyway?", parent=self.master):
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
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Could not open a window ({exc}). The interface needs a graphical display; "
              "use the 'gephyra' command line instead.")
        return 1
    GephyraApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
