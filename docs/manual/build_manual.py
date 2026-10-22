"""
Builds docs/Gephyra-GUI-Manual.pdf, the user manual of the Gephyra interface.

    python3 docs/manual/build_manual.py

Needs `reportlab` (pip install reportlab). Screenshots are read from
docs/manual/images/ by the file names listed in SCREENSHOTS; a missing image
leaves a captioned placeholder frame in its place, so the manual can be built
before every screenshot exists. Set VIDEO_URL to link a walkthrough video.
"""

import os
import re
import sys
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
IMAGES = os.path.join(HERE, "images")
OUTPUT = os.path.join(REPO, "docs", "Gephyra-GUI-Manual.pdf")

VIDEO_URL = ""   # e.g. "https://youtu.be/..." once the walkthrough is recorded

ACCENT = colors.HexColor("#0891b2")
DARK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#5b6b7c")
SOFT = colors.HexColor("#eef6f9")
LINE = colors.HexColor("#d6dde5")

# file name -> what the screenshot should show (also the placeholder text)
SCREENSHOTS = {
    "01_overview.png": "The main window: sidebar with the three steps and the output folder, the form, the progress bar and the log.",
    "01b_output_folder.png": "Choosing the output folder with Change... in the sidebar.",
    "02_analysis.png": "Step 1, Pathway Analysis, filled in with a topology, a trajectory and a root selection.",
    "02b_analysis_running.png": "While the analysis runs: the progress bar counts frames and the log streams messages.",
    "03_analysis_done.png": "The confirmation shown when the analysis finishes, listing the saved files and their folder.",
    "04_clustering.png": "Step 2, Clustering, with the pathway data filled in from step 1.",
    "04b_clustering_running.png": "While clustering runs: progress through the Frechet distance matrix.",
    "05_clustering_done.png": "The confirmation shown when clustering finishes.",
    "06_visualization.png": "Step 3, Visualization: viewer, mode and the molecular-scene section.",
    "07_script_ready.png": "The 'Script ready' confirmation with the command that opens the script.",
    "08_scene_frame.png": "A frame scene in ChimeraX: protein cartoon, ligand, water molecules and dashed water bridges (hypothetical example system).",
    "09_scene_clusters.png": "A cluster scene: colour-coded medoid paths around the protein, with the legend (hypothetical example system).",
    "10_light_theme.png": "The same window in the light theme.",
}


def version():
    with open(os.path.join(REPO, "src", "gephyra", "_version.py")) as f:
        return re.search(r'"([^"]+)"', f.read()).group(1)


# ------------------------------------------------------------------ styles
base = getSampleStyleSheet()
S = {
    "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=10, leading=14.5,
                           textColor=DARK, spaceAfter=6),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=19, leading=23,
                         textColor=DARK, spaceBefore=4, spaceAfter=10),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                         textColor=ACCENT, spaceBefore=12, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=MUTED),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=MUTED,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=9, leading=12.5, textColor=DARK,
                           backColor=SOFT, borderPadding=(5, 6, 5, 6), leftIndent=6, rightIndent=6,
                           spaceBefore=4, spaceAfter=9),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=12, textColor=DARK),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=DARK),
    "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=10, leading=14.5,
                             textColor=DARK, leftIndent=14, bulletIndent=3, spaceAfter=3),
    "tip": ParagraphStyle("tip", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=DARK,
                          backColor=colors.HexColor("#ecfeff"), borderColor=ACCENT, borderWidth=0.6,
                          borderPadding=(6, 8, 6, 8), spaceBefore=6, spaceAfter=10),
    "toc_title": ParagraphStyle("toc_title", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=DARK,
                                spaceAfter=14),
    "toc1": ParagraphStyle("toc1", fontName="Helvetica", fontSize=10.5, leading=17, textColor=DARK),
}


class Doc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "h1":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


def P(text, style="body"):
    return Paragraph(text, S[style])

def H1(text):
    return P(text, "h1")

def H2(text):
    return P(text, "h2")

def code(text):
    para = P(text.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"), "code")
    return KeepTogether([para])

def bullets(items):
    return [Paragraph(i, S["bullet"], bulletText="•") for i in items]

def tip(text):
    return P("<b>Tip.</b> " + text, "tip")

def table(rows, widths, header=True):
    data = [[Paragraph(str(c), S["cellb" if (header and r == 0) or c_i == 0 else "cell"]) for c_i, c in enumerate(row)]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(style))
    return t

def screenshot(name, width=16 * cm):
    """The image if present, otherwise a captioned placeholder frame."""
    caption = SCREENSHOTS[name]
    path = os.path.join(IMAGES, name)
    if os.path.exists(path):
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(path).getSize()
        height = width * ih / iw
        if height > 13 * cm:
            width, height = width * 13 * cm / height, 13 * cm
        img = Image(path, width=width, height=height)
        frame = Table([[img]], colWidths=[width + 4])
        frame.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, LINE), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                                   ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        return KeepTogether([frame, P(caption, "caption")])
    box = Table([[P(f"<b>Screenshot: {name}</b><br/>{caption}<br/><font color='#5b6b7c'>Add the image to "
                    f"docs/manual/images/{name} and rebuild.</font>", "cell")]], colWidths=[width], rowHeights=[4.2 * cm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, ACCENT), ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 14)]))
    return KeepTogether([box, P(caption, "caption")])


def on_page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.2 * cm, f"Gephyra {version()}  -  Graphical interface manual")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
        canvas.setStrokeColor(LINE)
        canvas.line(2 * cm, 1.55 * cm, A4[0] - 2 * cm, 1.55 * cm)
    canvas.restoreState()


def cover(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(colors.HexColor("#0d1117"))
    canvas.rect(0, h * 0.42, w, h * 0.58, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#22d3ee"))
    canvas.rect(2 * cm, h * 0.42 - 0.12 * cm, 3 * cm, 0.24 * cm, stroke=0, fill=1)
    # A small water-bridge motif: nodes joined by dashed bonds.
    canvas.setStrokeColor(colors.HexColor("#22d3ee"))
    canvas.setFillColor(colors.HexColor("#22d3ee"))
    pts = [(13.2, 22.8), (15.0, 24.3), (16.9, 23.2), (18.4, 25.0), (16.4, 26.4), (14.4, 26.0)]
    canvas.setDash(3, 3)
    canvas.setLineWidth(1.4)
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        canvas.line(x1 * cm, y1 * cm, x2 * cm, y2 * cm)
    canvas.setDash()
    for x, y in pts:
        canvas.circle(x * cm, y * cm, 0.22 * cm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 38)
    canvas.drawString(2 * cm, h - 7.2 * cm, "Gephyra")
    canvas.setFont("Helvetica", 16)
    canvas.drawString(2 * cm, h - 8.4 * cm, "Graphical interface - user manual")
    canvas.setFillColor(colors.HexColor("#8b98a6"))
    canvas.setFont("Helvetica", 11)
    canvas.drawString(2 * cm, h - 9.4 * cm, "Water-bridge network discovery, clustering and visualization")
    canvas.drawString(2 * cm, h * 0.42 + 1.2 * cm, f"Version {version()}   -   {date.today():%B %Y}")
    canvas.setFillColor(DARK)
    canvas.setFont("Helvetica", 10)
    canvas.drawString(2 * cm, 4.2 * cm, "Peralta-Morales, M. F., Sciortino, G. & Marechal, J.-D.")
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 3.6 * cm, "https://github.com/insilichem/Gephyra")
    canvas.restoreState()


# ----------------------------------------------------------------- content
def story():
    s = [Spacer(1, 1), PageBreak()]

    toc = TableOfContents()
    toc.levelStyles = [S["toc1"]]
    s += [P("Contents", "toc_title"), toc, PageBreak()]

    s += [H1("1. What the interface does"),
          P("Gephyra finds chains of water molecules that connect a chosen <i>root</i> atom (for example a ligand "
            "oxygen) to the surrounding solvent in a molecular dynamics trajectory, groups the chains that recur "
            "into spatial channels, and writes ready-to-run scenes for molecular viewers."),
          P("The graphical interface runs the whole workflow in three steps, without memorising command-line "
            "options:"),
          *bullets(["<b>1. Pathway Analysis</b> reads the topology and trajectory and writes every water bridge, "
                    "frame by frame, to a results file (<font face='Courier'>.jsonl</font>).",
                    "<b>2. Clustering</b> groups the bridges that recur in space into channels and writes a "
                    "cluster file (<font face='Courier'>.json</font>).",
                    "<b>3. Visualization</b> writes a script for UCSF ChimeraX, UCSF Chimera, PyMOL or VMD that "
                    "loads the protein, the ligand and the water network and shows them together."]),
          P("Every step shows its progress live, can be cancelled, and ends with a confirmation that lists the files "
            "written and the folder they are in."),
          ]

    s += [H1("2. Installation and start-up"),
          H2("Requirements"),
          *bullets(["<b>Python 3.10 or newer; 3.11 recommended</b>, in a dedicated conda environment (below). Using a separate environment "
                    "keeps the Python version and libraries Gephyra needs from clashing with other software, such "
                    "as a PyMOL installation that ships its own Python.",
                    "Conda (Miniconda, Miniforge or Anaconda) to create that environment.",
                    "MDAnalysis, NetworkX, NumPy and SciPy, installed together with Gephyra.",
                    "At least one molecular viewer to look at the results: UCSF ChimeraX, UCSF Chimera, PyMOL or VMD."]),
          H2("Installing Gephyra"),
          P("In a terminal, download Gephyra, create its environment and install it:"),
          code('git clone https://github.com/insilichem/Gephyra.git\n'
               'cd Gephyra\n'
               'conda create -n gephyra -c conda-forge python=3.11 "tk=*=xft*"\n'
               'conda activate gephyra\n'
               'pip install .'),
          P("The same environment, with the scientific libraries also taken from conda-forge, can be created from "
            "the file shipped with Gephyra:"),
          code('git clone https://github.com/insilichem/Gephyra.git\n'
               'cd Gephyra\n'
               'conda env create -f environment.yml\n'
               'conda activate gephyra\n'
               'pip install .'),
          P("<font face='Courier'>\"tk=*=xft*\"</font> selects the build of the Tk graphics library with smooth, "
            "anti-aliased fonts; with the default build the interface text looks blocky. Check the installation "
            "with <font face='Courier'>gephyra --version</font>."),
          tip("Run <font face='Courier'>conda activate gephyra</font> in every new terminal before using Gephyra. "
              "If the <font face='Courier'>gephyra</font> command is not found, the environment is not active."),
          H2("Starting the interface"),
          code("conda activate gephyra\ngephyra gui\n# or\ngephyra-gui"),
          P("The window opens on step 1. The folder you start it from is the default output folder (you can "
            "change it at any time, see section 3)."),
          tip("If the text looks blocky or pixelated, Gephyra is running in a Python whose Tk library has no "
              "anti-aliased fonts, and the interface says so in its log. This does not happen in the environment "
              "above; in another conda environment, install the right build with "
              "<font face='Courier'>conda install -c conda-forge \"tk=*=xft*\"</font> and restart the interface."),
          ]

    s += [PageBreak(), H1("3. The window at a glance"),
          screenshot("01_overview.png"),
          table([["Area", "What it does"],
                 ["Sidebar: workflow", "The three steps. Click one to open its form; the current step is highlighted."],
                 ["Sidebar: output folder", "Where all results are saved. <b>Change...</b> picks another folder, "
                  "<b>Open</b> shows it in your file manager."],
                 ["Sidebar: theme", "Switches between the dark and the light theme. The choice is remembered."],
                 ["Form", "The options of the current step, grouped in sections. Hover over a field to see a short "
                  "explanation. <b>Browse</b> opens a file picker."],
                 ["Buttons", "<b>Reset defaults</b> restores the standard values, <b>Copy CLI command</b> copies the "
                  "equivalent command line, and the highlighted button runs the step (also Ctrl+Enter)."],
                 ["Progress bar and status", "Show what is running and how far it has got. <b>Cancel</b> stops a "
                  "running step after the current frame."],
                 ["Log", "Messages from Gephyra as they happen: warnings in amber, errors in red, saved files in green. "
                  "<b>Hide log</b> and <b>Clear</b> manage it."]],
                [4.2 * cm, 12.3 * cm]),
          H2("Where results are saved"),
          P("Output fields such as <i>Results file</i> or <i>Script name</i> hold file <b>names</b>, which are saved "
            "inside the output folder shown in the sidebar. Type an absolute path in a field to save that file "
            "somewhere else. If the output folder does not exist, the interface offers to create it."),
          screenshot("01b_output_folder.png", width=12 * cm),
          P("After a step finishes, its confirmation window lists every saved file with its folder, and offers "
            "<b>Open folder</b> and <b>Save a copy to...</b>, so results can also be stored in another location "
            "after the calculation."),
          screenshot("10_light_theme.png", width=12 * cm),
          ]

    s += [PageBreak(), H1("4. Step 1 - Pathway Analysis"),
          P("Finds the water bridges in every analysed frame of the trajectory."),
          screenshot("02_analysis.png"),
          table([["Field", "Meaning", "Default"],
                 ["Topology", "Structure/topology file of the system (.prmtop, .psf, .tpr, .pdb, ...). Bond "
                  "information enables the full hydrogen-bond geometry criterion.", "required"],
                 ["Trajectory", "The frames to analyse (.dcd, .xtc, .nc, ...). For a single structure, choose a "
                  "coordinate file such as a .pdb.", "required"],
                 ["Root selection", "MDAnalysis selection of the atom(s) the bridges start from, e.g. "
                  "<font face='Courier'>resname LIG and name O4</font>.", "required"],
                 ["Water selection", "MDAnalysis selection of the bridging solvent.", "SOL, WAT or HOH"],
                 ["Stride", "Analyse every N-th frame. Larger values are faster.", "1"],
                 ["Max depth / Min depth", "Longest and shortest chains (number of waters) that are kept.", "10 / 1"],
                 ["Cooperativity", "Per-step discount along a chain, between 0 and 1 (1 switches it off).", "0.92"],
                 ["Coarse cutoff", "Neighbour search radius in angstrom.", "4.5"],
                 ["Superpose on", "Atoms used to superpose every frame on the first one, so that the same water channel "
                  "has the same coordinates in every frame even if the protein drifts or rotates in the box. "
                  "<i>auto</i> uses the protein C-alpha atoms; <i>none</i> keeps the trajectory coordinates "
                  "(only for trajectories that are already fitted).", "auto"],
                 ["Results file", "Name of the frame-by-frame results file in the output folder.", "results.jsonl"],
                 ["CSV summary", "Optional table of all paths, for spreadsheets.", "-"]],
                [3.4 * cm, 10.1 * cm, 3.0 * cm]),
          tip("For a first look at a long trajectory, start with a stride of 10-50. The analysis time grows with the "
              "number of frames and with Max depth."),
          screenshot("02b_analysis_running.png", width=14 * cm),
          P("Click <b>Run analysis</b>. The progress bar counts the frames; when it finishes, a confirmation reports "
            "how many water-bridge paths were found and where the files were saved. Click <b>Next: Clustering</b> "
            "to continue: the results file is already filled in on the next step."),
          screenshot("03_analysis_done.png", width=12 * cm),
          ]

    s += [PageBreak(), H1("5. Step 2 - Clustering"),
          P("Groups the paths found in step 1 into recurring spatial channels. Different water molecules occupying "
            "the same channel at different times end up in the same cluster."),
          screenshot("04_clustering.png"),
          table([["Field", "Meaning", "Default"],
                 ["Pathway data", "The results file of step 1 (filled in automatically).", "-"],
                 ["Frechet threshold", "How similar two paths must be (in angstrom) to belong to the same cluster. "
                  "Lower values give more, finer clusters; 3-8 is a typical range.", "6.0"],
                 ["Coarse threshold / trigger", "Settings of a fast pre-screening used for very large data sets; "
                  "the defaults suit most cases.", "auto / 1000"],
                 ["Max paths", "Memory cap: beyond this, only the most frequent paths are clustered.", "60000"],
                 ["Cluster file", "Name of the cluster file in the output folder.", "clustered_pathways.json"]],
                [3.4 * cm, 10.1 * cm, 3.0 * cm]),
          screenshot("04b_clustering_running.png", width=14 * cm),
          P("Click <b>Run clustering</b>. When it finishes, a confirmation window reports how many channels were "
            "identified and lists the saved file. <b>Next: Visualization</b> opens step 3 with the cluster file "
            "already selected."),
          screenshot("05_clustering_done.png", width=12 * cm),
          ]

    s += [PageBreak(), H1("6. Step 3 - Visualization"),
          P("Writes a script that opens a complete scene in your molecular viewer: the protein as a cartoon, the "
            "ligand and the bridging waters as sticks, and the water network drawn on top."),
          screenshot("06_visualization.png"),
          H2("Choosing what to draw"),
          table([["Mode", "Input file", "What the scene shows"],
                 ["cluster", "cluster file (step 2)", "The protein and ligand with the medoid (most representative) path "
                  "of each cluster as a colour-coded tube, and a legend with each cluster's occupancy. <b>Top "
                  "clusters</b> limits the picture to the most occupied ones. Enter a <b>Cluster ID</b> to rebuild "
                  "that one cluster as real water molecules at the frame it comes from."],
                 ["frame", "results file (step 1)", "One frame: the real water molecules (oxygen and hydrogens) and every "
                  "water bridge as a dashed bond. Choose the frame number in <b>Frame</b>."],
                 ["density", "results file (step 1)", "Every path of every frame as thin transparent tubes, showing "
                  "where water bridges form most often."]],
                [2.0 * cm, 3.4 * cm, 11.1 * cm]),
          H2("Molecular scene"),
          P("With <b>Include protein, ligand and waters</b> ticked (the default), Gephyra re-opens the analysed "
            "trajectory and writes a small structure file (<font face='Courier'>&lt;name&gt;_structure.pdb</font>) next to "
            "the script, which the script loads by itself. The topology and trajectory are read from the data file; "
            "the line under the section says which ones. Fill in <b>Topology</b> and <b>Trajectory</b> only for "
            "older result files that do not record them, or if the files have moved."),
          P("Untick it to write only the network geometry, then open the analysed structure in the viewer yourself "
            "before running the script."),
          H2("Writing and opening the script"),
          P("Pick the <b>Viewer</b>, click <b>Write script</b>, and the confirmation shows the exact command that opens "
            "it. Do not open the structure first: the script does it."),
          screenshot("07_script_ready.png", width=12 * cm),
          table([["Viewer", "Script", "How to open it"],
                 ["UCSF ChimeraX", ".cxc", "<font face='Courier'>chimerax scene.cxc</font>, drag the file onto the "
                  "window, or type <font face='Courier'>open scene.cxc</font>."],
                 ["UCSF Chimera", ".py", "<font face='Courier'>chimera scene.py</font>, or File > Open..."],
                 ["PyMOL", ".py", "<font face='Courier'>pymol scene.py</font>, or <font face='Courier'>run scene.py</font> "
                  "inside PyMOL."],
                 ["VMD", ".tcl", "<font face='Courier'>vmd -e scene.tcl</font>, or <font face='Courier'>source "
                  "scene.tcl</font> in the Tk Console. The <font face='Courier'>-e</font> matters: without it VMD tries "
                  "to read the script as a molecule."]],
                [3.0 * cm, 1.6 * cm, 11.9 * cm]),
          P("<b>VMD scripts must be run with <font face='Courier'>vmd -e scene.tcl</font>.</b> Without "
            "<font face='Courier'>-e</font>, VMD treats the script as a molecule file, reports <i>Unable to determine "
            "file type</i>, and draws nothing. Inside a running VMD, use <font face='Courier'>source scene.tcl</font> "
            "in Extensions > Tk Console instead.", "tip"),
          P("Running a script again replaces the previous Gephyra scene."),
          screenshot("08_scene_frame.png"),
          screenshot("09_scene_clusters.png"),
          ]

    s += [PageBreak(), H1("7. Output files"),
          table([["File", "Written by", "Contents"],
                 ["results.jsonl", "Step 1", "One line per analysed frame with every water bridge (atoms, coordinates, "
                  "H-quality score), plus the analysed topology/trajectory and settings."],
                 ["summary.csv", "Step 1 (optional)", "The same paths as a table."],
                 ["clustered_pathways.json", "Step 2", "The channels: size, occupancy, persistence, H-quality and the "
                  "medoid path of each, plus the analysed system."],
                 ["name.cxc / .py / .tcl", "Step 3", "The viewer script."],
                 ["name_structure.pdb", "Step 3", "The protein, ligand and bridging waters loaded by the script."],
                 ["name.bild, name.pb", "Step 3", "Network geometry and pseudobonds used by the Chimera and ChimeraX "
                  "scripts."]],
                [4.2 * cm, 3.0 * cm, 9.3 * cm]),
          P("H-quality (Z) sums the geometric quality of every route through a channel. It is not a probability and "
            "can exceed 1 for channels with many parallel routes; <i>hquality_norm</i> = Z/(1+Z) is a bounded "
            "version for comparisons. See the README for details."),
          ]

    s += [H1("8. Troubleshooting"),
          table([["Message or symptom", "What to do"],
                 ["Text looks blocky or pixelated", "Install the anti-aliased Tk build: <font face='Courier'>conda "
                  "install -c conda-forge \"tk=*=xft*\"</font> (section 2)."],
                 ["Root selection returned no atoms", "Check the residue and atom names in your topology; the selection "
                  "uses MDAnalysis syntax."],
                 ["A trajectory file is required", "Choose a trajectory. For a single structure, choose a .pdb as the "
                  "trajectory."],
                 ["Mode 'cluster' needs the cluster file", "In step 3, cluster mode reads the file from step 2; frame and "
                  "density modes read the results file from step 1."],
                 ["The data file does not record its structure", "The file comes from an older version of Gephyra. Fill in "
                  "Topology and Trajectory in the Molecular scene section."],
                 ["VMD: Unable to determine file type", "Open the script with <font face='Courier'>vmd -e scene.tcl</font>."],
                 ["Clustering finds no channels", "Check that the trajectory frames are superposed: keep <b>Superpose on</b> "
                  "at <i>auto</i> (or a selection of rigid atoms) and repeat step 1. Also check how many frames contain "
                  "bridges at all: a channel must appear in at least two frames to be kept, so for rare bridges use a "
                  "smaller stride."],
                 ["Too many clusters to read the picture", "Lower <b>Top clusters</b>, or draw a single cluster with "
                  "<b>Cluster ID</b>."],
                 ["Bridges or protein cut at the box edge", "Image/centre the trajectory before the analysis (for example "
                  "cpptraj autoimage or gmx trjconv -pbc mol -center)."],
                 ["Something else went wrong", "Start the interface with <font face='Courier'>gephyra -v gui</font> to see "
                  "full error details in the terminal, and include them when reporting the problem."]],
                [5.4 * cm, 11.1 * cm]),
          ]

    s += [H1("9. Command-line equivalents"),
          P("Every run writes its command-line equivalent to the log, and <b>Copy CLI command</b> copies it, so a "
            "setup made in the interface can be repeated in scripts or on a cluster. For example:"),
          code('gephyra calculate --topo system.prmtop --traj run.dcd \\\n'
               '    --root "resname LIG and name O4" --stride 20 --output results.jsonl\n'
               'gephyra cluster --data results.jsonl --output clustered_pathways.json\n'
               'gephyra visualize --data clustered_pathways.json --format chimerax \\\n'
               '    --mode cluster --top_clusters 10 --output cluster_scene'),
          ]

    s += [H1("10. Video walkthrough")]
    if VIDEO_URL:
        s += [P(f"A short video shows the interface in use: <link href='{VIDEO_URL}' color='#0891b2'>{VIDEO_URL}</link>")]
    else:
        s += [P("A short video walkthrough of the interface will be linked here.")]
    return s


def main():
    doc = Doc(OUTPUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2.2 * cm,
              title="Gephyra - Graphical interface manual", author="Peralta-Morales, Sciortino, Marechal",
              subject="User manual of the Gephyra graphical interface")
    doc.multiBuild(story(), onFirstPage=cover, onLaterPages=on_page)
    missing = [n for n in SCREENSHOTS if not os.path.exists(os.path.join(IMAGES, n))]
    print(f"Wrote {OUTPUT}")
    if missing:
        print(f"{len(missing)} screenshot placeholder(s): " + ", ".join(missing))


if __name__ == "__main__":
    sys.exit(main())
