# Gradient-Free Hybrid Fusion for Reference-Guided PCB Defect Localization

<div align="center">

[![Paper](https://img.shields.io/badge/IEEE-IEACon%202026-blue?style=flat-square&logo=ieee)](https://ieeeieacon.org)
[![Python](https://img.shields.io/badge/Python-3.10-green?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![DINOv2](https://img.shields.io/badge/Backbone-DINOv2%20ViT--B/14-purple?style=flat-square)](https://github.com/facebookresearch/dinov2)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Rashadul Nafis Riyad · Gianmarco Goycochea Casas · Zool Hilmi Ismail**

*7th IEEE Industrial Electronics and Applications Conference (IEACon 2026)*
*JW Marriott Hotel, Kuala Lumpur, Malaysia · September 7–8, 2026*

</div>

---

## Abstract

Automated optical inspection of printed circuit boards remains difficult when defects are extremely small, annotations are limited, and only a few board layouts are available. This work introduces a **gradient-free hybrid pipeline** that fuses classical reference-guided pixel differencing with frozen DINOv2 patch features under a board-held-out weight selection protocol. No model weights are updated at any stage. On a public six-class PCB benchmark comprising 693 defective images and ten golden reference boards, the hybrid achieves a mean average precision of **0.6092** — a **+0.0779 absolute gain** (14.7% relative) over pixel differencing alone — with statistically significant improvement across all six defect categories and all ten held-out boards. Each image pair is processed in **98 ms** on a commodity GPU, making the method immediately deployable in industrial AOI workflows without any training overhead.


---

## Highlights

- **Zero training** — no gradient updates, no fine-tuning, no labeled defect examples required
- **Reference-guided** — exploits the golden reference image standard in industrial AOI
- **Statistically rigorous** — paired t-test (p = 2.87×10⁻⁷¹) and Wilcoxon (p = 4.81×10⁻⁶³)
- **Consistent** — hybrid improves all 6 defect classes and all 10 held-out board layouts
- **Fast** — 98 ms per image pair (10.2 pairs/second) on NVIDIA Quadro RTX 5000
- **Reproducible** — full code, results, and figures publicly available

---

## Results

### Overall Patch-Level Localization Performance

| Method | Mean AP | 95% CI | Mean AUC | Top-1 | Top-5 |
|--------|:-------:|:------:|:--------:|:-----:|:-----:|
| Pixel differencing | 0.5313 | [0.520, 0.544] | 0.7865 | 0.9365 | 0.9942 |
| DINOv2 (frozen) | 0.4032 | [0.388, 0.419] | 0.9198 | 0.6926 | 0.9163 |
| **Hybrid, fixed α=0.75** | **0.6104** | — | **0.8908** | 0.9336 | 0.9942 |
| **Hybrid, held-out α** | **0.6092** | [0.597, 0.621] | **0.8890** | **0.9365** | **0.9942** |

> **Note:** These are patch-level localization metrics, not bounding-box detector mAP.
> Results are not directly comparable to supervised detector benchmarks.

### Class-Wise Mean AP

| Class | Pixel | DINOv2 | Hybrid | Gain |
|-------|:-----:|:------:|:------:|:----:|
| Missing hole | 0.5127 | 0.3652 | 0.5992 | +0.0864 |
| Mouse bite | 0.5271 | 0.4912 | 0.6258 | **+0.0986** |
| Open circuit | 0.6364 | 0.4563 | 0.7042 | +0.0678 |
| Short | 0.4241 | 0.3753 | 0.5050 | +0.0809 |
| Spur | 0.5112 | 0.4101 | 0.6000 | +0.0889 |
| Spurious copper | 0.5761 | 0.3216 | 0.6209 | +0.0448 |

### Board-Wise Mean AP

| Board | Pixel | Hybrid | Δ |
|-------|:-----:|:------:|:-:|
| 01 | 0.4210 | 0.5063 | +0.0853 |
| 04 | 0.5539 | 0.6286 | +0.0747 |
| 05 | 0.5613 | 0.6085 | +0.0473 |
| 06 | 0.5725 | 0.6464 | +0.0739 |
| 07 | 0.5145 | 0.5326 | +0.0182 |
| 08 | 0.5563 | 0.6415 | +0.0852 |
| 09 | 0.5852 | 0.6525 | +0.0674 |
| 10 | 0.5257 | 0.6182 | +0.0925 |
| 11 | 0.5559 | 0.6907 | **+0.1348** |
| 12 | 0.5515 | 0.6537 | +0.1022 |

> Board IDs follow original dataset numbering. Boards 02 and 03 are absent from the dataset.

---

## Method

### Pipeline Overview

```
┌──────────────────────────────────────────────────────────┐
│               Image Pair Input                           │
│       (Golden Reference + Defective Test Image)          │
└─────────────────────┬────────────────────────────────────┘
                      │  Resize: shorter side = 448 px
          ┌───────────┴────────────┐
          │                        │
   ┌──────▼──────┐          ┌──────▼──────┐
   │   PIXEL     │          │   DINOv2    │
   │   BRANCH    │          │   BRANCH    │
   │  p = 16 px  │          │  p = 14 px  │
   │             │          │             │
   │  |I - R|    │          │  ℓ₂-norm    │
   │  avg per    │          │  patch      │
   │  patch      │          │  cosine gap │
   └──────┬──────┘          └──────┬──────┘
          │                        │ bilinear upsample
          │    min-max normalise    │  to 28×28 grid
          └────────────┬───────────┘
                       │
             ┌─────────▼─────────┐
             │   HYBRID FUSION   │
             │                   │
             │  S = α·Ŝ_pix +   │
             │  (1−α)·Ŝ_dino    │
             │                   │
             │  α via board-     │
             │  held-out search  │
             └─────────┬─────────┘
                       │
             ┌─────────▼─────────┐
             │   ANOMALY MAP     │
             │   28 × 28 grid    │
             │  patch-level AP   │
             └───────────────────┘
```

### Board-Held-Out Weight Selection

To avoid tuning α on the same board being evaluated, we use a leave-one-board-out protocol:

1. For each target board `b`, hold it out as the test set
2. Evaluate all α ∈ {0.00, 0.05, ..., 1.00} on the remaining 9 boards
3. Select `α* = argmax mean AP` on the 9 training boards
4. Apply `α*` to board `b` for final evaluation

**Key finding:** `α* = 0.75` for 9 of 10 boards independently.
A globally fixed `α = 0.75` achieves mean AP of **0.6104** — only **0.0012 below** held-out selection — confirming stability and immediate deployability without per-board calibration.

---

## Repository Structure

```
pcb_research/
│
├── src/
│   ├── run_full_experiment.py         # Main pipeline — pixel + DINOv2 + hybrid
│   ├── prepare_paper_assets.py        # Publication figures and statistical tests
│   ├── generate_labels_only.py        # Fast patch-label regeneration utility
│   └── run_fixed_alpha_ablation.py    # Fixed α=0.75 ablation experiment
│
├── ieacon_pcb_hybrid_dino/
│   ├── results/
│   │   ├── summary_overall.json       # Overall method comparison
│   │   ├── per_image_metrics.csv      # Per-image AP and AUC (693 rows)
│   │   ├── class_summary.csv          # Per-class results
│   │   ├── board_summary.csv          # Per-board results
│   │   ├── fold_summary.csv           # Board-held-out fold results
│   │   ├── paired_stats.json          # t-test and Wilcoxon p-values
│   │   ├── bootstrap_ci_ap.json       # 95% bootstrap CI (5,000 resamples)
│   │   └── fixed_alpha_ablation.json  # Fixed α=0.75 ablation results
│   │
│   ├── paper/
│   │   ├── method_pipeline.png        # Figure 1 — Pipeline overview
│   │   ├── boxplot_ap.png             # Figure 2 — AP distributions
│   │   └── qualitative_grid.png       # Figure 3 — Qualitative comparison
│   │
│   └── figures/                       # Additional analysis figures
│       ├── alpha_by_board.png         # Selected α per held-out board
│       ├── class_mean_ap.png          # Class-wise AP bar chart
│       └── board_mean_ap.png          # Board-wise AP bar chart
│
├── data/                              # Dataset (not tracked — see below)
│   └── PCB_DATASET/
│       ├── images/                    # Defective test images by class
│       ├── Annotations/               # Pascal VOC bounding boxes
│       └── PCB_USED/                  # 10 golden reference boards (.JPG)
│
├── notebooks/                         # Exploration notebooks
├── requirements.txt
└── README.md
```

---

## Setup and Installation

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.x (recommended)
- ~16 GB GPU VRAM (tested on Quadro RTX 5000)
- ~5 GB disk space for dataset

### Step 1 — Clone

```bash
git clone https://github.com/RashadulRD786/pcb-hybrid-dino-ieacon2026.git
cd pcb-hybrid-dino-ieacon2026
```

### Step 2 — Environment

```bash
conda create -n pcb_experiment python=3.10 -y
conda activate pcb_experiment
```

### Step 3 — PyTorch (CUDA 12.4)

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

### Step 4 — Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Dataset

Download the PCB Defects dataset:

```
https://www.kaggle.com/datasets/akhatova/pcb-defects/data
```

Place the extracted folder at `data/PCB_DATASET/` with this structure:

```
data/PCB_DATASET/
├── images/
│   ├── Missing_hole/      (115 images)
│   ├── Mouse_bite/        (115 images)
│   ├── Open_circuit/      (116 images)
│   ├── Short/             (116 images)
│   ├── Spur/              (115 images)
│   └── Spurious_copper/   (116 images)
├── Annotations/           (Pascal VOC .xml per image)
└── PCB_USED/              (10 golden reference .JPG files)
```

---

## Reproducing Results

### Full Experiment Pipeline

```bash
cd src/

# Main experiment — pixel diff + DINOv2 + hybrid + board-held-out selection
# Runtime: ~20–40 minutes on GPU
python3 run_full_experiment.py
```

Expected terminal output (final lines):
```
[timestamp] Board 01: selected alpha=0.80, test mean AP=0.5063
[timestamp] Board 04: selected alpha=0.75, test mean AP=0.6286
...
[timestamp] Board 11: selected alpha=0.75, test mean AP=0.6907
[timestamp] Overall hybrid mean AP: 0.6092
[timestamp] Experiment finished successfully
```

```bash
# Generate publication-ready figures
# Runtime: ~2 minutes
python3 prepare_paper_assets.py
```

### Fixed-Alpha Ablation

```bash
# Generate patch label files (required once, < 1 min)
python3 generate_labels_only.py

# Run the fixed-α ablation
python3 run_fixed_alpha_ablation.py
```

Expected output:
```
Fixed alpha:       0.75
Images evaluated:  693
Mean AP:           0.6104
Mean AUC:          0.8908
Top-1 hit rate:    0.9336
Top-5 hit rate:    0.9942
```

### Verifying Reported Numbers

Every number in the paper can be verified directly from saved outputs:

```bash
# Table I — overall results
python3 -c "
import json
d = json.load(open('ieacon_pcb_hybrid_dino/results/summary_overall.json'))
for m, v in d['overall_summary'].items():
    print(f'{m}: AP={v[\"mean_ap\"]:.4f}, AUC={v[\"mean_auc\"]:.4f}')
"

# Statistical significance
python3 -c "
import json
d = json.load(open('ieacon_pcb_hybrid_dino/results/paired_stats.json'))
for r in d['ap']:
    if r['comparison'] == 'hybrid_vs_pixel':
        print(f't-test p={r[\"t_pvalue\"]:.2e}')
        print(f'Wilcoxon p={r[\"wilcoxon_pvalue\"]:.2e}')
"

# Fixed-alpha ablation
python3 -c "
import json
d = json.load(open('ieacon_pcb_hybrid_dino/results/fixed_alpha_ablation.json'))
print(f'Fixed alpha=0.75 Mean AP: {d[\"mean_ap\"]:.4f}')
"
```

---

## Inference Speed

Benchmarked on NVIDIA Quadro RTX 5000 (16 GB VRAM), 50-run average:

| Component | Latency |
|-----------|:-------:|
| DINOv2 ViT-B/14 forward pass × 2 | 95 ms |
| Pixel differencing + normalisation + fusion | 3 ms |
| **Total per image pair** | **98 ms** |
| **Throughput** | **10.2 pairs/second** |

No training loop. No weight updates at any stage.

---

## Dataset Statistics

| Property | Value |
|----------|:-----:|
| Total defective images | 693 |
| Defect classes | 6 |
| Total annotated defects | 2,953 |
| Golden reference boards | 10 |
| Annotation format | Pascal VOC bounding boxes |
| Patch grid resolution | 28 × 28 |
| Input resolution | 448 px (shorter side) |

---

## Important Notes

### On the Evaluation Protocol

This paper reports **patch-level localization AP**, not bounding-box IoU-based mAP. The evaluation divides each image into a 28×28 patch grid and scores each patch independently. A patch is labelled positive if it intersects any ground-truth bounding box. This formulation is **not directly comparable** to YOLO-style detector benchmarks on the same dataset.

### On the Training-Free Claim

The method is **gradient-free** — no backpropagation, no fine-tuning, and no weight updates occur. The only data-dependent step is selecting a scalar α from a 21-value grid using patch-level labels from held-out boards. DINOv2 features are loaded directly from the public Meta AI pretrained checkpoint with no modification.

### On Comparing to PatchCore and PaDiM

Methods such as PatchCore and PaDiM learn normality from a corpus of defect-free images and do not use a paired golden reference. Our method requires only one paired reference per board — the standard asset in any industrial AOI workflow. These represent different problem formulations and direct numerical comparison is not meaningful.

---

## Citation

```bibtex
@inproceedings{riyad2026gradient,
  title     = {Gradient-Free Hybrid Fusion for Reference-Guided
               {PCB} Defect Localization},
  author    = {Riyad, Rashadul Nafis and
               Goycochea Casas, Gianmarco and
               Ismail, Zool Hilmi},
  booktitle = {Proceedings of the 7th IEEE Industrial Electronics
               and Applications Conference (IEACon 2026)},
  year      = {2026},
  month     = {September},
  address   = {Kuala Lumpur, Malaysia},
  publisher = {IEEE}
}
```

---

## References

- **Kim et al. (2021)** — *Printed circuit board defect detection using deep learning via a skip-connected convolutional autoencoder.* Sensors.
- **Oquab et al. (2023)** — *DINOv2: Learning robust visual features without supervision.* arXiv:2304.07193.
- **Roth et al. (2022)** — *Towards total recall in industrial anomaly detection* (PatchCore). CVPR.
- **Defard et al. (2021)** — *PaDiM: A patch distribution modeling framework for anomaly detection.* ICPR.
- **Tang et al. (2019)** — *Online PCB defect detector on a new PCB defect dataset* (DeepPCB). arXiv:1902.06197.

---

## Contact

**Corresponding Author**

**Prof. Zool Hilmi Ismail**
Center for Artificial Intelligence and Robotics (CAIRO)
Universiti Teknologi Malaysia
Email: zool@utm.my | ORCID: [0000-0002-5918-636X](https://orcid.org/0000-0002-5918-636X)

---

## License

The experiment code in this repository is released under the [MIT License](LICENSE).

The PCB Defects dataset is subject to its original terms — see the [Kaggle dataset page](https://www.kaggle.com/datasets/akhatova/pcb-defects) for details.

DINOv2 is released under the [Apache 2.0 License](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) by Meta AI Research.