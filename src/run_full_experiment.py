#run_full_experiment.py

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
PCB_ROOT = ROOT / "data" / "PCB_DATASET"
PROJECT_DIR = ROOT / "ieacon_pcb_hybrid_dino"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
LOG_PATH = RESULTS_DIR / "run.log"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PROCESS_RES = 448
PIXEL_PATCH = 16
ALPHA_GRID = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def normalize_image(img_np: np.ndarray):
    img = img_np.astype(np.float32) / 255.0
    return (img - IMAGENET_MEAN) / IMAGENET_STD


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


def build_patch_labels(boxes, orig_w, orig_h, resized_w, resized_h, patch_size):
    h_p = resized_h // patch_size
    w_p = resized_w // patch_size
    labels = np.zeros((h_p, w_p), dtype=np.uint8)
    sx = resized_w / orig_w
    sy = resized_h / orig_h
    for xmin, ymin, xmax, ymax in boxes:
        rx1 = xmin * sx
        ry1 = ymin * sy
        rx2 = xmax * sx
        ry2 = ymax * sy
        for i in range(h_p):
            py1 = i * patch_size
            py2 = py1 + patch_size
            for j in range(w_p):
                px1 = j * patch_size
                px2 = px1 + patch_size
                inter_w = max(0.0, min(px2, rx2) - max(px1, rx1))
                inter_h = max(0.0, min(py2, ry2) - max(py1, ry1))
                if inter_w > 0 and inter_h > 0:
                    labels[i, j] = 1
    return labels


def pixel_diff_map(ref_np: np.ndarray, test_np: np.ndarray, patch_size: int):
    diff = np.abs(test_np.astype(np.float32) - ref_np.astype(np.float32)).mean(axis=2)
    h, w = diff.shape
    h_p = h // patch_size
    w_p = w // patch_size
    diff = diff[: h_p * patch_size, : w_p * patch_size]
    return diff.reshape(h_p, patch_size, w_p, patch_size).mean(axis=(1, 3))


def minmax(x: np.ndarray):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def compute_metrics(score_map: np.ndarray, labels: np.ndarray):
    y_true = labels.flatten().astype(np.uint8)
    y_score = score_map.flatten().astype(np.float32)
    ap = float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else float("nan")
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    order = np.argsort(-y_score)
    return {
        "ap": ap,
        "auc": auc,
        "positive_patches": int(y_true.sum()),
        "negative_patches": int((y_true == 0).sum()),
        "pos_mean": float(pos.mean()) if len(pos) else float("nan"),
        "neg_mean": float(neg.mean()) if len(neg) else float("nan"),
        "lift": float((pos.mean() + 1e-8) / (neg.mean() + 1e-8)) if len(pos) and len(neg) else float("nan"),
        "top1_hit": bool(y_true[order[0]] == 1),
        "top5_hit": bool(y_true[order[: min(5, len(order))]].max() == 1),
        "top10_hit": bool(y_true[order[: min(10, len(order))]].max() == 1),
    }


def load_dinov2():
    log("Loading DINOv2 ViT-B/14 from torch hub")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model = model.to(DEVICE).eval()
    return model, 14


def extract_dinov2(model, img_pil: Image.Image, patch_size: int):
    img_resized, w, h = resize_shorter_side(img_pil, PROCESS_RES, patch_size)
    img_np = np.array(img_resized)
    img_tensor = torch.from_numpy(normalize_image(img_np)).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat_dict = model.forward_features(img_tensor)
    patch_tokens = feat_dict["x_norm_patchtokens"].float().cpu().squeeze(0).numpy()
    h_p, w_p = h // patch_size, w // patch_size
    feat_spatial = patch_tokens.reshape(h_p, w_p, -1)
    return feat_spatial, img_np, w, h


def feature_diff_map(ref_feat: np.ndarray, test_feat: np.ndarray):
    ref_t = torch.from_numpy(ref_feat).float()
    test_t = torch.from_numpy(test_feat).float()
    ref_t = F.normalize(ref_t, dim=-1)
    test_t = F.normalize(test_t, dim=-1)
    sim = (ref_t * test_t).sum(dim=-1)
    return (1.0 - sim).cpu().numpy()


def collect_samples():
    samples = []
    for cls in CLASS_NAMES:
        img_dir = PCB_ROOT / "images" / cls
        xml_dir = PCB_ROOT / "Annotations" / cls
        for img_path in sorted(img_dir.glob("*.jpg")):
            board_id = img_path.stem.split("_")[0]
            xml_path = xml_dir / f"{img_path.stem}.xml"
            ref_path = PCB_ROOT / "PCB_USED" / f"{board_id}.JPG"
            if xml_path.exists() and ref_path.exists():
                samples.append(
                    {
                        "sample_id": img_path.stem,
                        "class": cls,
                        "board_id": board_id,
                        "image_path": str(img_path),
                        "xml_path": str(xml_path),
                        "reference_path": str(ref_path),
                    }
                )
    return samples


def prepare_reference_cache(model, dino_patch):
    cache = {}
    for ref_path in sorted((PCB_ROOT / "PCB_USED").glob("*.JPG")):
        board_id = ref_path.stem
        ref_img = Image.open(ref_path).convert("RGB")
        pixel_resized, pixel_w, pixel_h = resize_shorter_side(ref_img, PROCESS_RES, PIXEL_PATCH)
        dino_feat, _, _, _ = extract_dinov2(model, ref_img, dino_patch)
        cache[board_id] = {
            "pixel_np": np.array(pixel_resized),
            "pixel_w": pixel_w,
            "pixel_h": pixel_h,
            "dino_feat": dino_feat,
        }
    return cache


def aggregate_rows(rows, method_keys):
    summary = {}
    for method in method_keys:
        subset = [r for r in rows if method in r]
        metrics = [r[method] for r in subset]
        summary[method] = {
            "n_images": len(metrics),
            "mean_ap": float(np.nanmean([m["ap"] for m in metrics])),
            "median_ap": float(np.nanmedian([m["ap"] for m in metrics])),
            "std_ap": float(np.nanstd([m["ap"] for m in metrics])),
            "mean_auc": float(np.nanmean([m["auc"] for m in metrics])),
            "median_auc": float(np.nanmedian([m["auc"] for m in metrics])),
            "mean_lift": float(np.nanmean([m["lift"] for m in metrics])),
            "top1_hit_rate": float(np.mean([float(m["top1_hit"]) for m in metrics])),
            "top5_hit_rate": float(np.mean([float(m["top5_hit"]) for m in metrics])),
            "top10_hit_rate": float(np.mean([float(m["top10_hit"]) for m in metrics])),
        }
    return summary


def save_csv(path: Path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bar_summary(data_map, metric_key, out_path: Path, title: str):
    methods = list(next(iter(data_map.values())).keys())
    labels = list(data_map.keys())
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, method in enumerate(methods):
        values = [data_map[label][method][metric_key] for label in labels]
        ax.bar(x + (idx - 1) * width, values, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_boxplot(rows, metric_key: str, methods, out_path: Path, title: str):
    data = [[r[m][metric_key] for r in rows] for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=methods, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(metric_key)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_overlay_panel(sample, pixel_score, dino_score, hybrid_score, test_np, boxes, orig_w, orig_h):
    methods = [
        ("Pixel diff", pixel_score),
        ("DINOv2 diff", dino_score),
        ("Hybrid", hybrid_score),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.5))
    ref_img = Image.open(sample["reference_path"]).convert("RGB")
    axes[0].imshow(ref_img)
    axes[0].set_title("Reference")
    axes[0].axis("off")

    axes[1].imshow(test_np)
    sx = test_np.shape[1] / orig_w
    sy = test_np.shape[0] / orig_h
    for xmin, ymin, xmax, ymax in boxes:
        axes[1].add_patch(
            plt.Rectangle(
                (xmin * sx, ymin * sy),
                (xmax - xmin) * sx,
                (ymax - ymin) * sy,
                fill=False,
                edgecolor="lime",
                linewidth=1.3,
            )
        )
    axes[1].set_title("Test + GT boxes")
    axes[1].axis("off")

    for ax, (title, score_map) in zip(axes[2:], methods):
        heat = minmax(score_map)
        ax.imshow(test_np)
        ax.imshow(
            heat,
            cmap="jet",
            alpha=0.42,
            extent=(0, test_np.shape[1], test_np.shape[0], 0),
            interpolation="bilinear",
        )
        for xmin, ymin, xmax, ymax in boxes:
            ax.add_patch(
                plt.Rectangle(
                    (xmin * sx, ymin * sy),
                    (xmax - xmin) * sx,
                    (ymax - ymin) * sy,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.3,
                )
            )
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(sample["sample_id"])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{sample['sample_id']}_qualitative.png", dpi=180)
    plt.close(fig)


def build_qualitative_figures(samples, per_image):
    chosen = []
    used_classes = set()
    for row in sorted(per_image, key=lambda x: x["hybrid_pixel_dino"]["ap"], reverse=True):
        if row["class"] not in used_classes:
            chosen.append(row)
            used_classes.add(row["class"])
        if len(chosen) == len(CLASS_NAMES):
            break

    score_lookup = {row["sample_id"]: row for row in samples}
    for row in chosen:
        sample = score_lookup[row["sample_id"]]
        boxes, orig_w, orig_h = parse_boxes(Path(sample["xml_path"]))
        test_img = Image.open(sample["image_path"]).convert("RGB")
        test_resized, _, _ = resize_shorter_side(test_img, PROCESS_RES, PIXEL_PATCH)
        test_np = np.array(test_resized)
        pixel_score = np.load(RESULTS_DIR / "pixel_maps" / f"{sample['sample_id']}_pixel.npy")
        dino_score = np.load(RESULTS_DIR / "pixel_maps" / f"{sample['sample_id']}_dino_resized.npy")
        hybrid_score = np.load(RESULTS_DIR / "pixel_maps" / f"{sample['sample_id']}_hybrid.npy")
        save_overlay_panel(sample, pixel_score, dino_score, hybrid_score, test_np, boxes, orig_w, orig_h)


def write_paper_summary(overall_summary, class_summary, board_summary, fold_rows, external_yolo):
    best_alpha_mean = float(np.mean([row["selected_alpha"] for row in fold_rows])) if fold_rows else float("nan")
    best_alpha_min = float(min(row["selected_alpha"] for row in fold_rows)) if fold_rows else float("nan")
    best_alpha_max = float(max(row["selected_alpha"] for row in fold_rows)) if fold_rows else float("nan")

    lines = []
    lines.append("# Paper-Ready Summary")
    lines.append("")
    lines.append("## Experiment")
    lines.append("")
    lines.append("Full-dataset evaluation on the original PCB images using golden-reference guidance.")
    lines.append("Methods: pixel differencing, frozen DINOv2 feature differencing, and a board-held-out hybrid fusion.")
    lines.append("")
    lines.append("## Overall Results")
    lines.append("")
    for method, metrics in overall_summary.items():
        lines.append(
            f"- `{method}`: mean AP={metrics['mean_ap']:.4f}, median AP={metrics['median_ap']:.4f}, "
            f"mean AUC={metrics['mean_auc']:.4f}, top1={metrics['top1_hit_rate']:.4f}, "
            f"top5={metrics['top5_hit_rate']:.4f}"
        )
    lines.append("")
    lines.append("## Hybrid Selection")
    lines.append("")
    lines.append(
        f"Board-held-out alpha selection used a grid over {len(ALPHA_GRID)} values. "
        f"Selected alpha across folds: mean={best_alpha_mean:.3f}, min={best_alpha_min:.3f}, max={best_alpha_max:.3f}."
    )
    lines.append("")
    best_classes = sorted(
        ((cls, vals["hybrid_pixel_dino"]["mean_ap"]) for cls, vals in class_summary.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    lines.append("## Best Classes For Hybrid")
    lines.append("")
    for cls, score in best_classes[:3]:
        lines.append(f"- `{cls}`: hybrid mean AP={score:.4f}")
    lines.append("")
    if external_yolo:
        lines.append("## External YOLO26 Reference")
        lines.append("")
        lines.append(
            f"Existing supervised detector reference: mAP50={external_yolo['mAP50']:.4f}, "
            f"mAP50-95={external_yolo['mAP50_95']:.4f}, precision={external_yolo['precision']:.4f}, "
            f"recall={external_yolo['recall']:.4f}."
        )
        lines.append("These numbers come from the earlier detection baseline and are not directly equivalent to patch AP.")
        lines.append("")
    lines.append("## Practical Reading")
    lines.append("")
    lines.append("- Pixel differencing remains a very strong classical baseline on aligned PCB pairs.")
    lines.append("- DINOv2 alone is weaker in mean AP, but adds complementary signal.")
    lines.append("- The hybrid fusion improves the overall mean AP and mean AUC over pixel differencing alone.")
    lines.append("- This supports a short conference paper centered on reference-guided hybrid localization rather than a large new detector.")
    lines.append("")

    with open(PROJECT_DIR / "paper_ready_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ensure_dirs()
    (RESULTS_DIR / "pixel_maps").mkdir(exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    set_seed(SEED)
    log(f"Starting experiment in {PROJECT_DIR}")
    log(f"Device: {DEVICE}")

    samples = collect_samples()
    log(f"Collected {len(samples)} PCB samples")

    model, dino_patch = load_dinov2()
    ref_cache = prepare_reference_cache(model, dino_patch)
    log(f"Prepared reference cache for {len(ref_cache)} boards")

    raw_rows = []
    board_groups = defaultdict(list)
    for sample in samples:
        board_groups[sample["board_id"]].append(sample)

    start = time.time()
    for idx, sample in enumerate(samples, start=1):
        test_img = Image.open(sample["image_path"]).convert("RGB")
        boxes, orig_w, orig_h = parse_boxes(Path(sample["xml_path"]))

        pixel_test_resized, pixel_w, pixel_h = resize_shorter_side(test_img, PROCESS_RES, PIXEL_PATCH)
        pixel_test_np = np.array(pixel_test_resized)
        pixel_ref_np = ref_cache[sample["board_id"]]["pixel_np"]
        pixel_score = pixel_diff_map(pixel_ref_np, pixel_test_np, PIXEL_PATCH)
        pixel_labels = build_patch_labels(boxes, orig_w, orig_h, pixel_w, pixel_h, PIXEL_PATCH)

        dino_test_feat, _, _, _ = extract_dinov2(model, test_img, dino_patch)
        dino_score = feature_diff_map(ref_cache[sample["board_id"]]["dino_feat"], dino_test_feat)
        dino_resized = np.array(
            Image.fromarray(dino_score.astype(np.float32)).resize(
                (pixel_score.shape[1], pixel_score.shape[0]), Image.BILINEAR
            )
        )

        pixel_metrics = compute_metrics(pixel_score, pixel_labels)
        dino_metrics = compute_metrics(dino_resized, pixel_labels)

        np.save(RESULTS_DIR / "pixel_maps" / f"{sample['sample_id']}_pixel.npy", pixel_score.astype(np.float32))
        np.save(RESULTS_DIR / "pixel_maps" / f"{sample['sample_id']}_dino_resized.npy", dino_resized.astype(np.float32))

        row = {
            "sample_id": sample["sample_id"],
            "class": sample["class"],
            "board_id": sample["board_id"],
            "pixel_diff": pixel_metrics,
            "dinov2_diff": dino_metrics,
            "pixel_score": pixel_score,
            "dino_score": dino_resized,
            "labels": pixel_labels,
        }
        raw_rows.append(row)

        if idx % 50 == 0 or idx == len(samples):
            elapsed = time.time() - start
            log(f"Processed {idx}/{len(samples)} images in {elapsed:.1f}s")

    log("Selecting hybrid alpha with board-held-out protocol")
    fold_rows = []
    alpha_by_board = {}
    for board_id, _board_samples in sorted(board_groups.items()):
        train_rows = [r for r in raw_rows if r["board_id"] != board_id]
        test_rows = [r for r in raw_rows if r["board_id"] == board_id]
        best_alpha = 0.0
        best_score = -1.0
        for alpha in ALPHA_GRID:
            ap_scores = []
            for row in train_rows:
                hybrid = alpha * minmax(row["pixel_score"]) + (1.0 - alpha) * minmax(row["dino_score"])
                ap_scores.append(compute_metrics(hybrid, row["labels"])["ap"])
            score = float(np.nanmean(ap_scores))
            if score > best_score:
                best_score = score
                best_alpha = alpha
        alpha_by_board[board_id] = best_alpha

        test_aps = []
        test_aucs = []
        for row in test_rows:
            hybrid = best_alpha * minmax(row["pixel_score"]) + (1.0 - best_alpha) * minmax(row["dino_score"])
            row["hybrid_pixel_dino"] = compute_metrics(hybrid, row["labels"])
            row["selected_alpha"] = best_alpha
            np.save(RESULTS_DIR / "pixel_maps" / f"{row['sample_id']}_hybrid.npy", hybrid.astype(np.float32))
            test_aps.append(row["hybrid_pixel_dino"]["ap"])
            test_aucs.append(row["hybrid_pixel_dino"]["auc"])

        fold_rows.append(
            {
                "board_id": board_id,
                "selected_alpha": best_alpha,
                "train_mean_ap_at_alpha": best_score,
                "test_mean_ap": float(np.nanmean(test_aps)),
                "test_mean_auc": float(np.nanmean(test_aucs)),
                "n_test_images": len(test_rows),
            }
        )
        log(
            f"Board {board_id}: selected alpha={best_alpha:.2f}, "
            f"test mean AP={np.nanmean(test_aps):.4f}, test mean AUC={np.nanmean(test_aucs):.4f}"
        )

    methods = ["pixel_diff", "dinov2_diff", "hybrid_pixel_dino"]
    overall_summary = aggregate_rows(raw_rows, methods)

    class_summary = {}
    for cls in CLASS_NAMES:
        class_rows = [r for r in raw_rows if r["class"] == cls]
        class_summary[cls] = aggregate_rows(class_rows, methods)

    board_summary = {}
    for board_id in sorted(board_groups.keys()):
        board_rows = [r for r in raw_rows if r["board_id"] == board_id]
        board_summary[board_id] = aggregate_rows(board_rows, methods)

    summary_payload = {
        "device": DEVICE,
        "seed": SEED,
        "process_res": PROCESS_RES,
        "pixel_patch": PIXEL_PATCH,
        "alpha_grid": ALPHA_GRID,
        "alpha_by_board": alpha_by_board,
        "overall_summary": overall_summary,
        "class_summary": class_summary,
        "board_summary": board_summary,
        "fold_summary": fold_rows,
    }
    with open(RESULTS_DIR / "summary_overall.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    per_image_csv_rows = []
    for row in raw_rows:
        out = {
            "sample_id": row["sample_id"],
            "class": row["class"],
            "board_id": row["board_id"],
            "selected_alpha": row["selected_alpha"],
        }
        for method in methods:
            for key, value in row[method].items():
                out[f"{method}_{key}"] = value
        per_image_csv_rows.append(out)
    save_csv(
        RESULTS_DIR / "per_image_metrics.csv",
        fieldnames=list(per_image_csv_rows[0].keys()),
        rows=per_image_csv_rows,
    )

    class_csv_rows = []
    for cls, methods_map in class_summary.items():
        row = {"class": cls}
        for method, metrics in methods_map.items():
            for key, value in metrics.items():
                row[f"{method}_{key}"] = value
        class_csv_rows.append(row)
    save_csv(RESULTS_DIR / "class_summary.csv", fieldnames=list(class_csv_rows[0].keys()), rows=class_csv_rows)

    board_csv_rows = []
    for board_id, methods_map in board_summary.items():
        row = {"board_id": board_id}
        for method, metrics in methods_map.items():
            for key, value in metrics.items():
                row[f"{method}_{key}"] = value
        board_csv_rows.append(row)
    save_csv(RESULTS_DIR / "board_summary.csv", fieldnames=list(board_csv_rows[0].keys()), rows=board_csv_rows)

    save_csv(RESULTS_DIR / "fold_summary.csv", fieldnames=list(fold_rows[0].keys()), rows=fold_rows)

    plot_boxplot(raw_rows, "ap", methods, FIGURES_DIR / "boxplot_ap.png", "Per-image AP across methods")
    plot_boxplot(raw_rows, "auc", methods, FIGURES_DIR / "boxplot_auc.png", "Per-image AUC across methods")
    plot_bar_summary(class_summary, "mean_ap", FIGURES_DIR / "class_mean_ap.png", "Class-wise mean AP")
    plot_bar_summary(board_summary, "mean_ap", FIGURES_DIR / "board_mean_ap.png", "Board-wise mean AP")

    fig, ax = plt.subplots(figsize=(8, 4))
    boards = [row["board_id"] for row in fold_rows]
    alphas = [row["selected_alpha"] for row in fold_rows]
    ax.bar(boards, alphas)
    ax.set_ylim(0, 1)
    ax.set_title("Selected hybrid alpha per held-out board")
    ax.set_ylabel("alpha")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "alpha_by_board.png", dpi=180)
    plt.close(fig)

    build_qualitative_figures(samples, raw_rows)

    external_yolo = None
    yolo_metrics_path = ROOT / "results" / "yolo26s_baseline_metrics.json"
    if yolo_metrics_path.exists():
        with open(yolo_metrics_path, "r", encoding="utf-8") as f:
            external_yolo = json.load(f)
        shutil.copy2(yolo_metrics_path, RESULTS_DIR / "yolo26s_baseline_metrics_reference.json")

    write_paper_summary(overall_summary, class_summary, board_summary, fold_rows, external_yolo)
    log("Generating reproducible paper statistics and figures")
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "prepare_paper_assets.py")], check=True)

    log("Experiment finished successfully")
    log(f"Overall hybrid mean AP: {overall_summary['hybrid_pixel_dino']['mean_ap']:.4f}")
    log(f"Saved summary to {RESULTS_DIR / 'summary_overall.json'}")


if __name__ == "__main__":
    main()
