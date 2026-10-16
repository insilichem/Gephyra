import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from gephyra import forms
from gephyra.cli import build_parser, main
from gephyra.visualize import CLUSTER_PALETTE, _rgb_hex, export_chimerax_script, run_visualization, validate_chimerax_model


def _write_jsonl(path, frames, n_frames=None):
    with open(path, "w") as f:
        f.write(json.dumps({"type": "metadata", "n_frames_analyzed": n_frames or len(frames), "parameters": {"stride": 1}}) + "\n")
        for idx, paths in frames:
            f.write(json.dumps({"type": "frame", "frame_idx": idx, "paths": paths}) + "\n")


PATH_A = {"nodes": [0, 1, 2], "atom_ids": [1, 2, 3], "coords": [[0, 0, 0], [2.8, 0, 0], [5.6, 0, 0]], "hquality": 0.5}
PATH_B = {"nodes": [0, 1, 3], "atom_ids": [1, 2, 4], "coords": [[0, 0, 0], [2.8, 0, 0], [2.8, 2.8, 0]], "hquality": 0.4}
CLUSTERS = [
    {"cluster_id": 1, "size": 3, "occupancy": 0.8, "medoid_frame": 4, "medoid_coords": [[0, 0, 0], [1, 0, 0]]},
    {"cluster_id": 2, "size": 1, "occupancy": 0.25, "medoid_frame": 7, "medoid_coords": [[0, 0, 0], [0, 1, 0], [0, 2, 0]]},
]


class _TmpDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def p(self, *parts):
        return os.path.join(self.tmp, *parts)

    def read(self, *parts):
        with open(self.p(*parts)) as f:
            return f.read()


class TestChimeraXExport(_TmpDir):
    def test_frame_mode_writes_scoped_selection_bild_and_pseudobonds(self):
        _write_jsonl(self.p("r.jsonl"), [(0, [PATH_A, PATH_B])])
        export_chimerax_script(self.p("r.jsonl"), self.p("f.cxc"), mode="frame", frame_idx=0)

        cxc = self.read("f.cxc")
        self.assertIn("select #1@@serial_number=1 | #1@@serial_number=2 | #1@@serial_number=3 | #1@@serial_number=4", cxc)
        self.assertIn("style sel stick", cxc)
        self.assertIn(f'open "{os.path.abspath(self.p("f.bild"))}" name gephyra_frame_0', cxc)
        self.assertIn(f'open "{os.path.abspath(self.p("f.pb"))}"', cxc)

        # The shared 1-2 bond appears once in the pseudobond file.
        pb_lines = [l for l in self.read("f.pb").splitlines() if not l.startswith(";")]
        self.assertEqual(pb_lines, ["#1@@serial_number=1 #1@@serial_number=2",
                                    "#1@@serial_number=2 #1@@serial_number=3",
                                    "#1@@serial_number=2 #1@@serial_number=4"])
        bild = self.read("f.bild")
        self.assertEqual(bild.count(".cylinder"), 4)
        self.assertIn(".color 1.0 0.5 0.0", bild)

    def test_frame_mode_custom_model_and_fallback_serials(self):
        path = {"nodes": [9, 10], "coords": [[0, 0, 0], [2.8, 0, 0]]}  # old output without atom_ids
        _write_jsonl(self.p("r.jsonl"), [(3, [path])])
        export_chimerax_script(self.p("r.jsonl"), self.p("f.cxc"), mode="frame", frame_idx=3, model_spec="#2.1")
        self.assertIn("select #2.1@@serial_number=10 | #2.1@@serial_number=11", self.read("f.cxc"))
        self.assertIn("#2.1@@serial_number=10 #2.1@@serial_number=11", self.read("f.pb"))

    def test_frame_mode_skips_pbc_artifacts(self):
        path = {"nodes": [0, 1, 2], "atom_ids": [1, 2, 3], "coords": [[0, 0, 0], [2.8, 0, 0], [40, 0, 0]]}
        _write_jsonl(self.p("r.jsonl"), [(0, [path])])
        export_chimerax_script(self.p("r.jsonl"), self.p("f.cxc"), mode="frame", frame_idx=0, max_bond_draw_dist=6.0)
        self.assertEqual(self.read("f.bild").count(".cylinder"), 1)
        self.assertNotIn("serial_number=3", self.read("f.pb"))

    def test_large_selection_is_chunked(self):
        path = {"nodes": list(range(120)), "atom_ids": list(range(1, 121)),
                "coords": [[i * 2.8, 0, 0] for i in range(120)]}
        _write_jsonl(self.p("r.jsonl"), [(0, [path])])
        export_chimerax_script(self.p("r.jsonl"), self.p("f.cxc"), mode="frame", frame_idx=0)
        lines = self.read("f.cxc").splitlines()
        self.assertEqual(sum(l.startswith("select #1") for l in lines), 1)
        self.assertEqual(sum(l.startswith("select add ") for l in lines), 2)

    def test_empty_frame_writes_nothing(self):
        _write_jsonl(self.p("r.jsonl"), [(0, [])])
        with self.assertLogs("gephyra.visualize", level="WARNING"):
            export_chimerax_script(self.p("r.jsonl"), self.p("f.cxc"), mode="frame", frame_idx=0)
        self.assertFalse(os.path.exists(self.p("f.cxc")))

    def test_density_mode_is_transparent_overlay(self):
        _write_jsonl(self.p("r.jsonl"), [(0, [PATH_A]), (1, [PATH_B])])
        export_chimerax_script(self.p("r.jsonl"), self.p("d.cxc"), mode="density")
        bild = self.read("d.bild")
        self.assertIn(".transparency 0.7", bild)
        self.assertEqual(bild.count(".cylinder"), 4)
        self.assertIn("name gephyra_density", self.read("d.cxc"))
        self.assertFalse(os.path.exists(self.p("d.pb")))

    def test_cluster_mode_colours_and_legend(self):
        with open(self.p("c.json"), "w") as f:
            json.dump(CLUSTERS, f)
        export_chimerax_script(self.p("c.json"), self.p("c.cxc"), mode="cluster")
        bild, cxc = self.read("c.bild"), self.read("c.cxc")
        for rgb in CLUSTER_PALETTE[:2]:
            self.assertIn(f".color {rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f}", bild)
        self.assertIn(".comment Cluster 2: medoid extracted from frame 7", bild)
        self.assertIn(f'2dlabels text "Cluster 1  occ 0.80  n=3" color {_rgb_hex(CLUSTER_PALETTE[0])}', cxc)
        self.assertIn("ypos 0.91", cxc)

    def test_cluster_id_filter(self):
        with open(self.p("c.json"), "w") as f:
            json.dump(CLUSTERS, f)
        export_chimerax_script(self.p("c.json"), self.p("c.cxc"), mode="cluster", cluster_id=2)
        self.assertNotIn("Cluster 1", self.read("c.cxc"))
        with self.assertLogs("gephyra.visualize", level="WARNING"):
            export_chimerax_script(self.p("c.json"), self.p("none.cxc"), mode="cluster", cluster_id=9)
        self.assertFalse(os.path.exists(self.p("none.cxc")))

    def test_path_with_spaces_and_dots_is_quoted(self):
        os.mkdir(self.p("my dir.py"))
        _write_jsonl(self.p("r.jsonl"), [(0, [PATH_A])])
        export_chimerax_script(self.p("r.jsonl"), self.p("my dir.py", "out.cxc"), mode="density")
        self.assertIn(f'open "{os.path.abspath(self.p("my dir.py", "out.bild"))}"', self.read("my dir.py", "out.cxc"))

    def test_model_spec_validation(self):
        self.assertEqual(validate_chimerax_model(" #3.2 "), "#3.2")
        for bad in ("1", "#", "#a", "#1;close", "", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_chimerax_model(bad)

    def test_run_visualization_routes_and_appends_extension(self):
        _write_jsonl(self.p("r.jsonl"), [(0, [PATH_A])])
        run_visualization(self.p("r.jsonl"), format="chimerax", mode="frame", frame_idx=0,
                          output_file=self.p("viz"), chimerax_model="#4")
        self.assertIn("#4@@serial_number=1", self.read("viz.cxc"))

    def test_cli_accepts_chimerax_and_rejects_bad_model(self):
        _write_jsonl(self.p("r.jsonl"), [(0, [PATH_A])])
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["-q", "visualize", "--data", self.p("r.jsonl"), "--format", "chimerax",
                         "--mode", "density", "--output", self.p("cli")])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(self.p("cli.cxc")))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            main(["visualize", "--data", "x", "--format", "chimerax", "--chimerax_model", "1"])
        self.assertEqual(cm.exception.code, 2)


@unittest.skipUnless(shutil.which("chimerax") and os.environ.get("GEPHYRA_TEST_CHIMERAX"),
                     "set GEPHYRA_TEST_CHIMERAX=1 with ChimeraX on PATH to run")
class TestChimeraXIntegration(_TmpDir):
    """Opens the generated scripts in a headless ChimeraX and checks for errors."""

    def test_scripts_open_cleanly(self):
        import numpy as np
        import MDAnalysis as mda
        u = mda.Universe.empty(4, trajectory=True, n_residues=4, atom_resindex=[0, 1, 2, 3])
        u.add_TopologyAttr("resname", ["LIG", "SOL", "SOL", "SOL"])
        u.add_TopologyAttr("name", ["O1", "OW", "OW", "OW"])
        u.add_TopologyAttr("resid", [1, 2, 3, 4])
        u.atoms.positions = np.array([[0.0, 0, 0], [2.8, 0, 0], [5.6, 0, 0], [2.8, 2.8, 0]])
        u.atoms.write(self.p("sys.pdb"))
        _write_jsonl(self.p("r.jsonl"), [(0, [PATH_A, PATH_B])])
        with open(self.p("c.json"), "w") as f:
            json.dump(CLUSTERS, f)
        for mode, data in (("frame", "r.jsonl"), ("density", "r.jsonl"), ("cluster", "c.json")):
            run_visualization(self.p(data), format="chimerax", mode=mode, frame_idx=0, output_file=self.p(mode))
        with open(self.p("check.cxc"), "w") as f:
            f.write(f'open "{self.p("sys.pdb")}"\n')
            for mode in ("frame", "density", "cluster"):
                f.write(f'open "{self.p(mode + ".cxc")}"\n')
        res = subprocess.run(["chimerax", "--nogui", "--exit", "--cmd", f'open "{self.p("check.cxc")}"'],
                             capture_output=True, text=True, timeout=300)
        out = res.stdout + res.stderr
        self.assertNotIn("Traceback", out)
        self.assertIn("Opened Pseudobonds frame.pb, 3 bonds", out)


class TestForms(_TmpDir):
    def setUp(self):
        super().setUp()
        for name in ("sys.pdb", "traj.xtc"):
            open(self.p(name), "w").close()
        _write_jsonl(self.p("r.jsonl"), [(0, [PATH_A])])
        with open(self.p("c.json"), "w") as f:
            json.dump(CLUSTERS, f)

    def calc(self, **overrides):
        v = dict(forms.DEFAULTS["calculate"], topo=self.p("sys.pdb"), traj=self.p("traj.xtc"),
                 root="resname LIG", output=self.p("out.jsonl"))
        v.update(overrides)
        return v

    def vis(self, **overrides):
        v = dict(forms.DEFAULTS["visualize"], data=self.p("c.json"), output=self.p("viz"))
        v.update(overrides)
        return v

    def assertFieldError(self, stage, values, field):
        with self.assertRaises(forms.FormError) as cm:
            forms.parse_stage(stage, values)
        self.assertEqual(cm.exception.field, field, cm.exception.message)

    def test_defaults_match_cli_defaults(self):
        parser = build_parser()
        calc = vars(parser.parse_args(["calculate", "--topo", "t", "--root", "r"]))
        for field in ("water", "stride", "max_depth", "min_depth", "cooperativity", "coarse_cutoff", "output"):
            self.assertEqual(str(calc[field]), forms.DEFAULTS["calculate"][field], field)
        clus = vars(parser.parse_args(["cluster", "--data", "d"]))
        for field in ("threshold", "coarse_trigger", "max_paths", "output"):
            self.assertEqual(str(clus[field]), forms.DEFAULTS["cluster"][field], field)

    def test_parse_calculate(self):
        kw = forms.parse_calculate(self.calc(csv=self.p("s.csv"), stride="5"))
        self.assertEqual(kw["stride"], 5)
        self.assertEqual(kw["cooperativity"], 0.92)
        self.assertEqual(kw["csv_file"], self.p("s.csv"))
        self.assertIsNone(forms.parse_calculate(self.calc(traj=""))["traj_file"])

    def test_calculate_errors_name_the_field(self):
        cases = [({"topo": ""}, "topo"), ({"topo": self.p("missing.pdb")}, "topo"),
                 ({"traj": self.p("missing.xtc")}, "traj"), ({"root": "  "}, "root"),
                 ({"stride": "0"}, "stride"), ({"stride": "1.5"}, "stride"), ({"max_depth": "x"}, "max_depth"),
                 ({"min_depth": "5", "max_depth": "3"}, "min_depth"), ({"cooperativity": "1.2"}, "cooperativity"),
                 ({"cooperativity": "0"}, "cooperativity"), ({"coarse_cutoff": "-1"}, "coarse_cutoff"),
                 ({"output": self.p("nope", "out.jsonl")}, "output"), ({"csv": self.p("nope", "s.csv")}, "csv")]
        for overrides, field in cases:
            with self.subTest(**overrides):
                self.assertFieldError("calculate", self.calc(**overrides), field)

    def test_parse_cluster(self):
        v = dict(forms.DEFAULTS["cluster"], data=self.p("r.jsonl"), output=self.p("c2.json"))
        kw = forms.parse_cluster(v)
        self.assertIsNone(kw["coarse_threshold"])
        self.assertEqual(kw["threshold"], 6.0)
        self.assertEqual(forms.parse_cluster(dict(v, coarse_threshold="2.5"))["coarse_threshold"], 2.5)
        self.assertFieldError("cluster", dict(v, threshold="0"), "threshold")
        self.assertFieldError("cluster", dict(v, max_paths="0"), "max_paths")
        self.assertFieldError("cluster", dict(v, data=self.p("c.json")), "data")  # wrong kind of file

    def test_parse_visualize(self):
        kw = forms.parse_visualize(self.vis(cluster_id="2", frame="7"))
        self.assertEqual(kw["cluster_id"], 2)
        self.assertIsNone(kw["frame_idx"])  # frame only applies to frame mode
        self.assertEqual(kw["chimerax_model"], "#1")
        kw = forms.parse_visualize(self.vis(data=self.p("r.jsonl"), mode="frame", frame="0", format="vmd"))
        self.assertEqual(kw["frame_idx"], 0)
        self.assertNotIn("chimerax_model", kw)

    def test_visualize_errors_name_the_field(self):
        cases = [({"mode": "cluster", "data": self.p("r.jsonl")}, "data"),
                 ({"mode": "density"}, "data"),
                 ({"mode": "frame", "data": self.p("r.jsonl"), "frame": "-1"}, "frame"),
                 ({"cluster_id": "0"}, "cluster_id"), ({"format": "rasmol"}, "format"),
                 ({"mode": "movie"}, "mode"), ({"chimerax_model": "1"}, "chimerax_model"),
                 ({"max_bond_draw_dist": "0"}, "max_bond_draw_dist")]
        for overrides, field in cases:
            with self.subTest(**overrides):
                self.assertFieldError("visualize", self.vis(**overrides), field)

    def test_command_line_round_trips_through_cli_parser(self):
        parser = build_parser()
        kw = forms.parse_calculate(self.calc(root="resname LIG and name 'O1'", csv=self.p("s.csv")))
        args = parser.parse_args(shlex.split(forms.build_command_line("calculate", kw))[1:])
        self.assertEqual(args.root, kw["root_sel"])
        self.assertEqual(args.traj, kw["traj_file"])
        self.assertEqual(args.csv, kw["csv_file"])
        self.assertEqual(args.cooperativity, kw["cooperativity"])

        kw = forms.parse_visualize(self.vis(cluster_id="2"))
        args = parser.parse_args(shlex.split(forms.build_command_line("visualize", kw))[1:])
        self.assertEqual((args.format, args.mode, args.cluster_id, args.chimerax_model), ("chimerax", "cluster", 2, "#1"))
        self.assertNotIn("--frame", forms.build_command_line("visualize", kw))

    def test_run_stage_dispatch(self):
        cb = object()
        with patch("gephyra.analysis.run_analysis") as ra, patch("gephyra.analysis.cluster_pathways") as cp, \
                patch("gephyra.visualize.run_visualization") as rv:
            forms.run_stage("calculate", {"a": 1}, progress_callback=cb)
            forms.run_stage("cluster", {"b": 2}, progress_callback=cb)
            forms.run_stage("visualize", {"c": 3}, progress_callback=cb)
        ra.assert_called_once_with(a=1, progress_callback=cb)
        cp.assert_called_once_with(b=2, progress_callback=cb)
        rv.assert_called_once_with(c=3)
        with self.assertRaises(ValueError):
            forms.run_stage("nope", {})

    def test_visualize_output_path(self):
        self.assertEqual(forms.visualize_output_path({"output_file": "a", "format": "chimerax"}), "a.cxc")
        self.assertEqual(forms.visualize_output_path({"output_file": "a.tcl", "format": "vmd"}), "a.tcl")


class TestProgressCallbacks(_TmpDir):
    def test_cluster_reports_progress_and_can_abort(self):
        from gephyra.analysis import cluster_pathways
        paths = [{"nodes": [0, i], "coords": [[0, 0, 0], [i, 0, 0]], "hquality": 1.0} for i in range(1, 6)]
        _write_jsonl(self.p("r.jsonl"), [(0, paths), (1, paths)])
        calls = []
        cluster_pathways(self.p("r.jsonl"), output_file=self.p("c.json"), progress_callback=lambda d, t: calls.append((d, t)))
        self.assertEqual(calls[-1], (10, 10))

        class Stop(Exception):
            pass

        def stop(d, t):
            raise Stop()
        with self.assertRaises(Stop):
            cluster_pathways(self.p("r.jsonl"), output_file=self.p("c2.json"), progress_callback=stop)


class TestGuiEntryPoints(unittest.TestCase):
    def test_cli_gui_subcommand_dispatches(self):
        with patch("gephyra.gui.main", return_value=0) as gm, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["gui"]), 0)
        gm.assert_called_once()

    def test_gui_main_without_display_returns_error(self):
        from gephyra import gui
        if gui.tk is None:
            self.skipTest("tkinter not available")
        with patch.object(gui.tk, "Tk", side_effect=gui.tk.TclError("no display name")), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(gui.main(), 1)
        self.assertIn("graphical display", out.getvalue())


def _make_root():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        return root
    except Exception:
        return None


class TestGuiApp(_TmpDir):
    """Exercises the real window when a display is available (skipped headless)."""

    def setUp(self):
        super().setUp()
        self.root = _make_root()
        if self.root is None:
            self.skipTest("no graphical display available")
        from gephyra.gui import GephyraApp
        self.calls = []
        self.app = GephyraApp(self.root, runner=self._runner)
        self.runner_behaviour = None
        open(self.p("sys.pdb"), "w").close()
        _write_jsonl(self.p("r.jsonl"), [(0, [PATH_A])])
        self.errors = []
        p = patch("gephyra.gui.messagebox.showerror", side_effect=lambda t, m, **k: self.errors.append((t, m)))
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        if getattr(self, "app", None) is not None:
            self.app.close()
        super().tearDown()

    def _runner(self, stage, kwargs, progress_callback=None):
        self.calls.append((stage, kwargs))
        if self.runner_behaviour:
            self.runner_behaviour(progress_callback)

    def _wait(self, timeout=10):
        t = time.time()
        while time.time() - t < timeout:
            self.root.update()
            if self.app.worker is not None and not self.app.worker.is_alive() and self.app.current_stage is None:
                return
            time.sleep(0.01)
        self.fail("worker did not finish")

    def _fill_calculate(self):
        c = self.app.vars["calculate"]
        c["topo"].set(self.p("sys.pdb"))
        c["root"].set("resname LIG")
        c["output"].set(self.p("out.jsonl"))

    def test_validation_error_shows_dialog_and_does_not_run(self):
        self._fill_calculate()
        self.app.vars["calculate"]["stride"].set("0")
        self.app.start("calculate")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.errors[0][0], "Invalid input")
        self.assertIn("Stride", self.errors[0][1])

    def test_successful_stages_prefill_the_next_tab(self):
        self._fill_calculate()
        self.app.start("calculate")
        self._wait()
        self.assertEqual(self.calls[0][0], "calculate")
        self.assertEqual(self.app.vars["cluster"]["data"].get(), self.p("out.jsonl"))

        self.app.vars["cluster"]["data"].set(self.p("r.jsonl"))
        self.app.vars["cluster"]["output"].set(self.p("c.json"))
        self.app.start("cluster")
        self._wait()
        self.assertEqual(self.app.vars["visualize"]["data"].get(), self.p("c.json"))
        self.assertEqual(self.app.vars["visualize"]["mode"].get(), "cluster")
        self.assertEqual(self.errors, [])

    def test_runtime_exception_becomes_error_dialog(self):
        def boom(_cb):
            raise ValueError("Root selection matched nothing")
        self.runner_behaviour = boom
        self._fill_calculate()
        self.app.start("calculate")
        self._wait()
        self.assertIn("Root selection matched nothing", self.errors[0][1])
        self.assertIn("failed", self.app.status_var.get())

    def test_logged_errors_are_reported_as_failure(self):
        import logging

        def log_error(_cb):
            logging.getLogger("gephyra.visualize").error("Cluster file is empty")
        self.runner_behaviour = log_error
        self._fill_calculate()
        self.app.start("calculate")
        self._wait()
        self.assertEqual(self.errors[0][1], "Cluster file is empty")

    def test_cancel_stops_at_next_progress_update(self):
        def slow(cb):
            for i in range(200):
                cb(i, 200)
                time.sleep(0.01)
        self.runner_behaviour = slow
        self._fill_calculate()
        self.app.start("calculate")
        self.app.cancel()
        self._wait()
        self.assertIn("Cancelled", self.app.status_var.get())
        self.assertEqual(self.errors, [])

    def test_mode_and_format_toggle_dependent_fields(self):
        v, w = self.app.vars["visualize"], self.app.widgets["visualize"]
        v["mode"].set("frame")
        v["format"].set("vmd")
        self.root.update()
        self.assertEqual(str(w["frame"].cget("state")), "normal")
        self.assertEqual(str(w["cluster_id"].cget("state")), "disabled")
        self.assertEqual(str(w["chimerax_model"].cget("state")), "disabled")

    def test_copy_command_puts_cli_on_clipboard(self):
        self._fill_calculate()
        self.app.copy_command("calculate")
        self.assertTrue(self.root.clipboard_get().startswith("gephyra calculate --topo"))


if __name__ == "__main__":
    unittest.main()
