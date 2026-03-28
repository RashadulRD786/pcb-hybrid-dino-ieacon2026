# Reviewer Script Bundle

This folder contains the scripts that were actually used to support the **methodology**, **quantitative results**, and **paper figures** of the manuscript:

`Training-Free Hybrid Fusion for Reference-Guided PCB Defect Localization`

## Included scripts

- `run_full_experiment.py`
  - Main experimental pipeline.
  - Computes the classical pixel-difference branch, the frozen DINOv2 branch, the board-held-out hybrid fusion, and the image-wise patch-level metrics.
  - Saves the quantitative outputs used in the paper under `results/`.

- `prepare_paper_assets.py`
  - Post-processing script for the paper.
  - Computes paired statistical tests and bootstrap confidence intervals from `results/per_image_metrics.csv`.
  - Regenerates the publication figures used in the manuscript, including the boxplot and qualitative comparison grid.

## What is intentionally not included

- Graphical-abstract utilities
- Alternative design/mockup scripts
- Reviewer-preview figure scripts not currently used in the manuscript

Those files were not used to generate the core methodology or the reported quantitative results.

## Expected project structure

These reviewer copies assume the current project layout:

- `archive/PCB_DATASET`
- `archive/ieacon_pcb_hybrid_dino`

They use project-relative paths from this folder and therefore do **not** depend on the original absolute path of the author machine.

## Minimal execution order

From this folder or from the project root:

1. `python run_full_experiment.py`
2. `python prepare_paper_assets.py`

## Main outputs produced by the scripts

Quantitative outputs:

- `results/summary_overall.json`
- `results/per_image_metrics.csv`
- `results/class_summary.csv`
- `results/board_summary.csv`
- `results/fold_summary.csv`
- `results/paired_stats.json`
- `results/bootstrap_ci_ap.json`

Figure outputs used in the paper:

- `paper/boxplot_ap.png`
- `paper/qualitative_grid.png`

## Dependencies

The scripts rely on the Python stack already used in the project, notably:

- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `scipy`
- `scikit-learn`
- `torch`

## Note

The evaluation reported in the paper is **patch-level localization**, not bounding-box detector mAP.
