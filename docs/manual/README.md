# Gephyra interface manual

`build_manual.py` builds [`../Gephyra-GUI-Manual.pdf`](../Gephyra-GUI-Manual.pdf), the user manual of
`gephyra gui`. It needs `reportlab`:

```bash
pip install reportlab
python3 docs/manual/build_manual.py
```

## Screenshots

Put screenshots in `images/` under these names and rebuild. Any that are missing appear as captioned
placeholder frames, and the build lists them.

| File | What it should show |
|---|---|
| `01_overview.png` | The main window: sidebar with the three steps and the output folder, form, progress bar and log |
| `02_analysis.png` | Step 1 filled in with a topology, a trajectory and a root selection |
| `03_analysis_done.png` | The confirmation after the analysis, listing the saved files |
| `04_clustering.png` | Step 2 with the pathway data filled in |
| `05_clustering_done.png` | The confirmation after clustering |
| `06_visualization.png` | Step 3: viewer, mode and the molecular-scene section |
| `07_script_ready.png` | The "Script ready" confirmation |
| `08_scene_frame.png` | A frame scene in a viewer: protein, ligand, water molecules and dashed bridges |
| `09_scene_clusters.png` | A cluster scene: colour-coded medoids around the protein, with the legend |
| `10_light_theme.png` | The window in the light theme |

PNG at the window's native size works best; images are scaled to the page width.

## Video

Set `VIDEO_URL` at the top of `build_manual.py` to link the walkthrough video in section 10.
