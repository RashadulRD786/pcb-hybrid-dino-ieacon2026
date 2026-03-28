# Gradient-Free Hybrid Fusion for Reference-Guided PCB Defect Localization

[![IEEE IEACon 2026](https://img.shields.io/badge/IEEE-IEACon%202026-blue)](https://ieeeieacon.org)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

This repository contains the complete experiment code, results, and
publication figures for the paper:

> **Gradient-Free Hybrid Fusion for Reference-Guided PCB Defect Localization**
> Rashadul Nafis Riyad, Gianmarco Goycochea Casas, Zool Hilmi Ismail
> *7th IEEE Industrial Electronics and Applications Conference (IEACon 2026)*
> Kuala Lumpur, Malaysia, September 7-8, 2026

### Key Results

| Method | Mean AP | Mean AUC | Top-1 | Top-5 |
|--------|---------|----------|-------|-------|
| Pixel diff | 0.5313 | 0.7865 | 0.9365 | 0.9942 |
| DINOv2 diff | 0.4032 | 0.9198 | 0.6926 | 0.9163 |
| Hybrid, fixed α=0.75 | 0.6104 | 0.8908 | 0.9336 | 0.9942 |
| **Hybrid, held-out α** | **0.6092** | **0.8890** | **0.9365** | **0.9942** |

- Hybrid improves mean AP by **+0.0779** (14.7% relative) over pixel differencing
- Consistent improvement across **all 6 defect classes** and **all 10 boards**
- Processes each image pair in **98 ms** on NVIDIA Quadro RTX 5000
- **Zero training** required — gradient-free and immediately deployable

---

## Method

The proposed pipeline fuses two complementary branches:

1. **Classical Pixel-Difference Branch** — Computes patch-level
   absolute RGB differences between the test image and golden
   reference on a 16×16 pixel-patch grid.

2. **Frozen DINOv2 Branch** — Extracts ℓ2-normalised patch tokens
   from DINOv2 ViT-B/14 and computes cosine-gap scores,
   upsampled to the pixel-branch grid.

Both maps are min-max normalised and fused as:
```
S_hybrid = α × S_pixel + (1 - α) × S_DINOv2
```

The fusion weight α is selected via a **board-held-out protocol**
using only 21 candidate values {0.00, 0.05, ..., 1.00}.
The optimal α converges to 0.75 across 9 of 10 boards.

---

## Repository Structure
```
pcb_research/
├── src/
│   ├── run_full_experiment.py       # Main experiment pipeline
│   ├── prepare_paper_assets.py      # Publication figures and stats
│   ├── generate_labels_only.py      # Fast label regeneration utility
│   └── run_fixed_alpha_ablation.py  # Fixed-α ablation experiment
│
├── ieacon_pcb_hybrid_dino/
│   ├── results/
│   │   ├── summary_overall.json     # Overall method comparison
│   │   ├── per_image_metrics.csv    # Per-image AP and AUC (693 rows)
│   │   ├── class_summary.csv        # Per-class results
│   │   ├── board_summary.csv        # Per-board results
│   │   ├── fold_summary.csv         # Board-held-out fold results
│   │   ├── paired_stats.json        # t-test and Wilcoxon results
│   │   ├── bootstrap_ci_ap.json     # 95% bootstrap CI for mean AP
│   │   └── fixed_alpha_ablation.json # Fixed α=0.75 ablation results
│   │
│   ├── paper/
│   │   ├── boxplot_ap.png           # Figure 2 in paper
│   │   ├── qualitative_grid.png     # Figure 3 in paper
│   │   └── method_pipeline.png      # Figure 1 in paper
│   │
│   └── figures/                     # Additional generated figures
│
├── data/                            # Dataset (not included, see below)
│   └── PCB_DATASET/
│       ├── images/
│       ├── Annotations/
│       └── PCB_USED/
│
├── notebooks/                       # Exploration notebooks
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/RashadulRD786/pcb-hybrid-dino-ieacon2026.git
cd pcb-hybrid-dino-ieacon2026
```

### 2. Create Environment
```bash
conda create -n pcb_experiment python=3.10 -y
conda activate pcb_experiment
```

### 3. Install PyTorch (CUDA 12.4)
```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download Dataset

The dataset is the paired-reference PCB benchmark from Kim et al. (2021).
Download from Kaggle:
```
https://www.kaggle.com/datasets/akhatova/pcb-defects/data
```

Extract and place at:
```
pcb_research/data/PCB_DATASET/
├── images/
│   ├── Missing_hole/
│   ├── Mouse_bite/
│   ├── Open_circuit/
│   ├── Short/
│   ├── Spur/
│   └── Spurious_copper/
├── Annotations/
└── PCB_USED/
```

---

## Reproducing the Results

### Step 1 — Run the Full Experiment
```bash
cd src/
python3 run_full_experiment.py
```

**Expected runtime:** ~20-40 minutes on NVIDIA GPU

**Expected output:**
```
[timestamp] Device: cuda
[timestamp] Collected 693 PCB samples
[timestamp] Loading DINOv2 ViT-B/14 from torch hub
[timestamp] Board 01: selected alpha=0.80, test mean AP=0.5063
[timestamp] Board 04: selected alpha=0.75, test mean AP=0.6286
...
[timestamp] Overall hybrid mean AP: 0.6092
[timestamp] Experiment finished successfully
```

### Step 2 — Generate Publication Figures
```bash
python3 prepare_paper_assets.py
```

### Step 3 — Run Fixed-Alpha Ablation
```bash
python3 generate_labels_only.py   # Generate label files first
python3 run_fixed_alpha_ablation.py
```

**Expected output:**
```
Fixed alpha: 0.75
Images evaluated: 693
Mean AP: 0.6104
Mean AUC: 0.8908
```

---

## Dataset

| Property | Value |
|----------|-------|
| Total defective images | 693 |
| Defect classes | 6 |
| Golden reference boards | 10 |
| Annotated defects | 2,953 |
| Annotation format | Pascal VOC bounding boxes |
| Image resolution | Variable (resized to 448px shorter side) |

**Six defect classes:**
Missing hole, Mouse bite, Open circuit, Short, Spur, Spurious copper

---

## Evaluation

This paper evaluates **patch-level localization** — NOT bounding-box
detection mAP. Each image is divided into a 28×28 patch grid and
scored at the patch level.

**Metrics reported:**
- Average Precision (AP) — patch-level precision-recall curve area
- AUC-ROC — global separation of defect vs normal patches
- Top-k Hit Rate (k=1,5,10) — defect in top-k ranked patches

**Statistical validation:**
- Paired t-test: p = 2.87 × 10⁻⁷¹
- Wilcoxon signed-rank: p = 4.81 × 10⁻⁶³
- 95% bootstrap CI: [0.5972, 0.6211]

---

## Key Findings

1. **Hybrid beats both branches** — Mean AP 0.6092 vs 0.5313
   (pixel) and 0.4032 (DINOv2 alone)

2. **α is stable** — Fixed α=0.75 achieves 0.6104 mean AP,
   only 0.0012 below board-held-out selection

3. **Consistent gains** — Hybrid improves all 6 defect classes
   and all 10 held-out boards

4. **Fast inference** — 98ms per image pair, 10.2 pairs/second
   on NVIDIA Quadro RTX 5000

5. **Failure analysis** — Degradation concentrates in spurious
   copper (40 images) and boards 04/07 (29 and 28 images).
   Mean gain when helped (0.111) exceeds mean loss when hurt
   (0.048), confirming favourable asymmetry

---

## Hardware Used

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA Quadro RTX 5000 (16GB VRAM) |
| CUDA | 12.4 |
| Driver | 550.163.01 |
| OS | Ubuntu 24.04 |
| Python | 3.10.12 |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| torch | DINOv2 inference |
| timm | Vision model utilities |
| numpy | Array operations |
| pandas | Results management |
| matplotlib | Figure generation |
| Pillow | Image loading |
| scikit-learn | AP and AUC computation |
| scipy | Statistical tests |

---

## Citation

If you use this code or results, please cite:
```bibtex
@inproceedings{riyad2026gradient,
  title     = {Gradient-Free Hybrid Fusion for Reference-Guided
               PCB Defect Localization},
  author    = {Riyad, Rashadul Nafis and
               Goycochea Casas, Gianmarco and
               Ismail, Zool Hilmi},
  booktitle = {Proceedings of the 7th IEEE Industrial Electronics
               and Applications Conference (IEACon)},
  year      = {2026},
  address   = {Kuala Lumpur, Malaysia}
}
```

---

## Contact

**Corresponding Author:**
Zool Hilmi Ismail
Email: zool@utm.my
ORCID: https://orcid.org/0000-0002-5918-636X

Center for Artificial Intelligence and Robotics
Universiti Teknologi Malaysia

---

## License

This code is released under the MIT License.
The PCB dataset is subject to its original license — see Kaggle page.

---

## Acknowledgements

- DINOv2 by Meta AI Research (Oquab et al., 2023)
- PCB dataset by Kim et al. (2021), accessed via Kaggle
- IEACon 2026 organizing committee
