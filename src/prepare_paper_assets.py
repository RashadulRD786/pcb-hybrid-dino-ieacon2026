#prepare_paper_assests.py

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
PCB_ROOT = ROOT / "data" / "PCB_DATASET"
PROJECT_DIR = ROOT / "ieacon_pcb_hybrid_dino"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
PAPER_DIR = PROJECT_DIR / "paper"
SEED = 42
BOOTSTRAP_RESAMPLES = 5000
PROCESS_RES = 448
PIXEL_PATCH = 16
DISPLAY_CLASSES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short"]
METHODS = ["pixel_diff", "dinov2_diff", "hybrid_pixel_dino"]
CLASS_LABELS = {
    "Missing_hole": "Missing hole",
    "Mouse_bite": "Mouse bite",
    "Open_circuit": "Open circuit",
    "Short": "Short",
    "Spur": "Spur",
    "Spurious_copper": "Spurious copper",
}
METHOD_LABELS = {
    "pixel_diff": "Classical\nPixel-Difference",
    "dinov2_diff": "Frozen DINOv2\nCosine-Gap",
    "hybrid_pixel_dino": "Reference-Guided\nHybrid Fusion",
}
METHOD_COLORS = {
    "pixel_diff": "#4C78A8",
    "dinov2_diff": "#54A24B",
    "hybrid_pixel_dino": "#F58518",
}


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def compute_paired_stats(df: pd.DataFrame):
    payload = {"ap": [], "auc": []}
    comparisons = [
        ("hybrid_vs_pixel", "hybrid_pixel_dino", "pixel_diff"),
        ("hybrid_vs_dino", "hybrid_pixel_dino", "dinov2_diff"),
        ("pixel_vs_dino", "pixel_diff", "dinov2_diff"),
    ]
    for metric in ["ap", "auc"]:
        for label, lhs, rhs in comparisons:
            lhs_vals = df[f"{lhs}_{metric}"].to_numpy(dtype=float)
            rhs_vals = df[f"{rhs}_{metric}"].to_numpy(dtype=float)
            diff = lhs_vals - rhs_vals
            t_stat = ttest_rel(lhs_vals, rhs_vals, nan_policy="omit")
            try:
                wilc = wilcoxon(diff)
                wilcoxon_pvalue = float(wilc.pvalue)
            except ValueError:
                wilcoxon_pvalue = float("nan")
            payload[metric].append(
                {
                    "comparison": label,
                    "mean_diff": float(np.nanmean(diff)),
                    "median_diff": float(np.nanmedian(diff)),
                    "improved_fraction": float(np.nanmean(diff > 0)),
                    "t_pvalue": float(t_stat.pvalue),
                    "wilcoxon_pvalue": wilcoxon_pvalue,
                }
            )
    return payload


def compute_bootstrap_ci(df: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    payload = {}
    n = len(df)
    for method in METHODS:
        values = df[f"{method}_ap"].to_numpy(dtype=float)
        boot_means = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            sample = rng.choice(values, size=n, replace=True)
            boot_means.append(np.nanmean(sample))
        boot_means = np.asarray(boot_means, dtype=float)
        payload[method] = {
            "mean_ap": float(np.nanmean(values)),
            "ci95_low": float(np.quantile(boot_means, 0.025)),
            "ci95_high": float(np.quantile(boot_means, 0.975)),
        }
    return payload


def minmax_map(x: np.ndarray):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def resize_shorter_side(img: Image.Image, shorter_side: int, patch_size: int):
    w, h = img.size
    if h < w:
        new_h = shorter_side
        new_w = int(w * shorter_side / h)
    else:
        new_w = shorter_side
        new_h = int(h * shorter_side / w)
    new_h = max(patch_size, (new_h // patch_size) * patch_size)
    new_w = max(patch_size, (new_w // patch_size) * patch_size)
    return img.resize((new_w, new_h), Image.BICUBIC), new_w, new_h


def parse_boxes(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    orig_w = int(size.findtext("width"))
    orig_h = int(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        boxes.append(
            (
                int(box.findtext("xmin")),
                int(box.findtext("ymin")),
                int(box.findtext("xmax")),
                int(box.findtext("ymax")),
            )
        )
    return boxes, orig_w, orig_h


def fit_to_canvas(img: Image.Image, target_w: int, target_h: int, bg=(255, 255, 255)):
    scale = min(target_w / img.width, target_h / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def draw_boxes_on_image(img: Image.Image, boxes, orig_w: int, orig_h: int, color=(50, 255, 80), width=3):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    sx = out.width / orig_w
    sy = out.height / orig_h
    for xmin, ymin, xmax, ymax in boxes:
        draw.rectangle(
            [xmin * sx, ymin * sy, xmax * sx, ymax * sy],
            outline=color,
            width=width,
        )
    return out


def make_overlay(test_img: Image.Image, score_map: np.ndarray, boxes, orig_w: int, orig_h: int):
    heat = (colormaps["viridis"](minmax_map(score_map))[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(test_img.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(test_img, heat_img, alpha=0.43)
    return draw_boxes_on_image(overlay, boxes, orig_w, orig_h)


def draw_box(ax, x, y, w, h, text, fc="#eef2f7"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.6,
        edgecolor="black",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12)


def draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=18, linewidth=1.5, color="black")
    ax.add_patch(arrow)


def make_method_pipeline():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "ref": (0.05, 0.64, 0.13, 0.14, "Golden\nreference"),
        "test": (0.05, 0.24, 0.13, 0.14, "Test PCB"),
        "pix": (0.26, 0.56, 0.15, 0.18, "Patchwise\npixel difference\nmap"),
        "dino": (0.26, 0.22, 0.15, 0.18, "Frozen DINOv2\npatch tokens\nand cosine-gap map"),
        "norm_pix": (0.52, 0.58, 0.13, 0.14, "Image-wise\nmin-max\nnormalization"),
        "norm_dino": (0.52, 0.24, 0.13, 0.14, "Image-wise\nmin-max\nnormalization"),
        "alpha": (0.72, 0.24, 0.14, 0.14, "Board-held-out\nalpha selection"),
        "fuse": (0.72, 0.56, 0.14, 0.16, "Weighted fusion\n$\\alpha\\tilde{S}^{pix}$ +\n$(1-\\alpha)\\tilde{S}^{dino}$"),
        "out": (0.90, 0.47, 0.08, 0.16, "Hybrid\nscore map"),
    }

    for _, (x, y, w, h, text) in nodes.items():
        draw_box(ax, x, y, w, h, text)

    draw_arrow(ax, (0.18, 0.71), (0.26, 0.65))
    draw_arrow(ax, (0.18, 0.31), (0.26, 0.31))
    draw_arrow(ax, (0.18, 0.69), (0.26, 0.33))
    draw_arrow(ax, (0.18, 0.29), (0.26, 0.59))
    draw_arrow(ax, (0.41, 0.65), (0.52, 0.65))
    draw_arrow(ax, (0.41, 0.31), (0.52, 0.31))
    draw_arrow(ax, (0.65, 0.65), (0.72, 0.64))
    draw_arrow(ax, (0.65, 0.31), (0.72, 0.31))
    draw_arrow(ax, (0.79, 0.38), (0.79, 0.56))
    draw_arrow(ax, (0.86, 0.64), (0.90, 0.55))

    ax.text(0.5, 0.92, "Reproducible reference-guided hybrid localization pipeline", ha="center", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "method_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_publication_boxplot(df: pd.DataFrame):
    data = [df[f"{method}_ap"].to_numpy(dtype=float) for method in METHODS]
    labels = [METHOD_LABELS[method] for method in METHODS]
    colors = [METHOD_COLORS[method] for method in METHODS]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#1F1F1F", "linewidth": 2.2},
        boxprops={"linewidth": 1.6, "edgecolor": "#222222"},
        whiskerprops={"linewidth": 1.5, "color": "#333333"},
        capprops={"linewidth": 1.5, "color": "#333333"},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    for idx, (vals, color) in enumerate(zip(data, colors), start=1):
        mean_val = float(np.nanmean(vals))
        ax.scatter(idx, mean_val, marker="D", s=58, color=color, edgecolor="black", linewidth=0.7, zorder=4)
        ax.text(idx, mean_val + 0.03, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title(
        "Training-Free Hybrid Fusion for Reference-Guided PCB Defect Localization",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Patch-level Localization Method", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel("Image-wise Average Precision (AP)", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12.5, fontweight="bold", linespacing=1.1)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.02, 1.05)
    legend_handle = Line2D([0], [0], marker="D", color="w", markerfacecolor="#666666", markeredgecolor="black", markersize=7, label="Method mean")
    ax.legend(handles=[legend_handle], loc="upper left", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(PAPER_DIR / "boxplot_ap.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def choose_qualitative_samples(df: pd.DataFrame):
    ranked = df.sort_values("hybrid_pixel_dino_ap", ascending=False, kind="mergesort")
    chosen = []
    used = set()
    for _, row in ranked.iterrows():
        cls = row["class"]
        if cls in DISPLAY_CLASSES and cls not in used:
            chosen.append(
                {
                    "class": cls,
                    "sample_id": row["sample_id"],
                    "hybrid_ap": float(row["hybrid_pixel_dino_ap"]),
                }
            )
            used.add(cls)
        if len(chosen) == len(DISPLAY_CLASSES):
            break
    return chosen


def make_qualitative_grid(chosen):
    target_w, target_h = 420, 280
    fig, axes = plt.subplots(len(chosen), 5, figsize=(18.8, 10.6), dpi=220)
    if len(chosen) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Golden\nReference",
        "Defective Test Image\nwith Ground-Truth Boxes",
        "Classical Pixel-Difference\nAnomaly Map",
        "Frozen DINOv2 Cosine-Gap\nAnomaly Map",
        "Reference-Guided Hybrid\nAnomaly Map",
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13.5, fontweight="bold", pad=8)

    for row_idx, item in enumerate(chosen):
        class_code = item["class"]
        class_label = CLASS_LABELS.get(class_code, class_code.replace("_", " "))
        sample_id = item["sample_id"]
        board_id = sample_id.split("_")[0]
        ref_path = PCB_ROOT / "PCB_USED" / f"{board_id}.JPG"
        img_path = PCB_ROOT / "images" / class_code / f"{sample_id}.jpg"
        xml_path = PCB_ROOT / "Annotations" / class_code / f"{sample_id}.xml"

        ref_img = Image.open(ref_path).convert("RGB")
        test_img = Image.open(img_path).convert("RGB")
        boxes, orig_w, orig_h = parse_boxes(xml_path)
        ref_resized, _, _ = resize_shorter_side(ref_img, PROCESS_RES, PIXEL_PATCH)
        test_resized, _, _ = resize_shorter_side(test_img, PROCESS_RES, PIXEL_PATCH)

        pixel_map = np.load(RESULTS_DIR / "pixel_maps" / f"{sample_id}_pixel.npy")
        dino_map = np.load(RESULTS_DIR / "pixel_maps" / f"{sample_id}_dino_resized.npy")
        hybrid_map = np.load(RESULTS_DIR / "pixel_maps" / f"{sample_id}_hybrid.npy")

        test_with_boxes = draw_boxes_on_image(test_resized, boxes, orig_w, orig_h)
        pixel_overlay = make_overlay(test_resized, pixel_map, boxes, orig_w, orig_h)
        dino_overlay = make_overlay(test_resized, dino_map, boxes, orig_w, orig_h)
        hybrid_overlay = make_overlay(test_resized, hybrid_map, boxes, orig_w, orig_h)

        row_images = [
            fit_to_canvas(ref_resized, target_w, target_h),
            fit_to_canvas(test_with_boxes, target_w, target_h),
            fit_to_canvas(pixel_overlay, target_w, target_h),
            fit_to_canvas(dino_overlay, target_w, target_h),
            fit_to_canvas(hybrid_overlay, target_w, target_h),
        ]

        for col_idx, img in enumerate(row_images):
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            for spine in axes[row_idx, col_idx].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.1)
                spine.set_edgecolor("#404040")

        sample_label = sample_id.replace("_", " ")
        row_header = (
            f"{class_label}\n"
            f"Representative sample: {sample_label}\n"
            f"Hybrid AP = {item['hybrid_ap']:.3f}"
        )
        axes[row_idx, 0].text(
            -0.18,
            0.5,
            row_header,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=12.5,
            fontweight="bold",
        )

    fig.suptitle(
        "Qualitative Comparison of Patch-level Localization Responses Across Representative PCB Defect Classes",
        fontsize=19,
        fontweight="bold",
        y=0.995,
    )
    plt.subplots_adjust(left=0.12, right=0.995, top=0.90, bottom=0.03, wspace=0.06, hspace=0.14)
    out_path = PAPER_DIR / "qualitative_grid.png"
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path


def copy_paper_figures():
    for name in ["class_mean_ap.png", "alpha_by_board.png"]:
        src = FIGURES_DIR / name
        dst = PAPER_DIR / name
        Image.open(src).save(dst)


def main():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "per_image_metrics.csv")

    paired_stats = compute_paired_stats(df)
    save_json(RESULTS_DIR / "paired_stats.json", paired_stats)

    bootstrap_ci = compute_bootstrap_ci(df)
    save_json(RESULTS_DIR / "bootstrap_ci_ap.json", bootstrap_ci)

    make_method_pipeline()
    make_publication_boxplot(df)
    chosen = choose_qualitative_samples(df)
    make_qualitative_grid(chosen)
    copy_paper_figures()

    manifest = {
        "display_classes": DISPLAY_CLASSES,
        "qualitative_selection_rule": "Stable descending sort by hybrid_pixel_dino_ap; first sample per displayed class.",
        "selected_samples": chosen,
        "paired_stats_file": str(RESULTS_DIR / "paired_stats.json"),
        "bootstrap_ci_file": str(RESULTS_DIR / "bootstrap_ci_ap.json"),
        "paper_figures": [
            str(PAPER_DIR / "method_pipeline.png"),
            str(PAPER_DIR / "qualitative_grid.png"),
            str(PAPER_DIR / "boxplot_ap.png"),
            str(PAPER_DIR / "class_mean_ap.png"),
            str(PAPER_DIR / "alpha_by_board.png"),
        ],
    }
    save_json(PAPER_DIR / "paper_assets_manifest.json", manifest)


if __name__ == "__main__":
    main()
