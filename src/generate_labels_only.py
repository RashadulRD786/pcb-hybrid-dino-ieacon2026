"""
Generates and saves patch-level label .npy files
from the dataset annotations only.
No model inference needed — very fast (< 1 minute).
"""

import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from PIL import Image

ROOT        = Path(__file__).resolve().parents[1]
PCB_ROOT    = ROOT / "data" / "PCB_DATASET"
RESULTS_DIR = ROOT / "ieacon_pcb_hybrid_dino" / "results"
PIXEL_MAPS  = RESULTS_DIR / "pixel_maps"
PROCESS_RES = 448
PIXEL_PATCH = 16
CLASS_NAMES = [
    "Missing_hole", "Mouse_bite", "Open_circuit",
    "Short", "Spur", "Spurious_copper",
]

def resize_shorter_side(img, shorter_side, patch_size):
    w, h = img.size
    if h < w:
        new_h = shorter_side
        new_w = int(w * shorter_side / h)
    else:
        new_w = shorter_side
        new_h = int(h * shorter_side / w)
    new_h = max(patch_size, (new_h // patch_size) * patch_size)
    new_w = max(patch_size, (new_w // patch_size) * patch_size)
    return new_w, new_h

def parse_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    orig_w = int(size.findtext("width"))
    orig_h = int(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        boxes.append((
            int(box.findtext("xmin")),
            int(box.findtext("ymin")),
            int(box.findtext("xmax")),
            int(box.findtext("ymax")),
        ))
    return boxes, orig_w, orig_h

def build_patch_labels(boxes, orig_w, orig_h,
                       resized_w, resized_h, patch_size):
    h_p = resized_h // patch_size
    w_p = resized_w  // patch_size
    labels = np.zeros((h_p, w_p), dtype=np.uint8)
    sx = resized_w / orig_w
    sy = resized_h / orig_h
    for xmin, ymin, xmax, ymax in boxes:
        rx1, ry1 = xmin * sx, ymin * sy
        rx2, ry2 = xmax * sx, ymax * sy
        for i in range(h_p):
            py1, py2 = i * patch_size, (i + 1) * patch_size
            for j in range(w_p):
                px1, px2 = j * patch_size, (j + 1) * patch_size
                iw = max(0.0, min(px2, rx2) - max(px1, rx1))
                ih = max(0.0, min(py2, ry2) - max(py1, ry1))
                if iw > 0 and ih > 0:
                    labels[i, j] = 1
    return labels

def main():
    saved = 0
    skipped = 0

    for cls in CLASS_NAMES:
        img_dir = PCB_ROOT / "images" / cls
        xml_dir = PCB_ROOT / "Annotations" / cls

        for img_path in sorted(img_dir.glob("*.jpg")):
            sid      = img_path.stem
            xml_path = xml_dir / f"{sid}.xml"
            board_id = sid.split("_")[0]
            ref_path = PCB_ROOT / "PCB_USED" / f"{board_id}.JPG"

            if not xml_path.exists() or not ref_path.exists():
                skipped += 1
                continue

            label_path = PIXEL_MAPS / f"{sid}_labels.npy"
            if label_path.exists():
                saved += 1
                continue  # already exists

            img = Image.open(img_path).convert("RGB")
            new_w, new_h = resize_shorter_side(
                img, PROCESS_RES, PIXEL_PATCH)

            boxes, orig_w, orig_h = parse_boxes(xml_path)
            labels = build_patch_labels(
                boxes, orig_w, orig_h,
                new_w, new_h, PIXEL_PATCH)

            np.save(label_path, labels)
            saved += 1

    print(f"Done. Saved {saved} label files. Skipped {skipped}.")
    print(f"Location: {PIXEL_MAPS}")

if __name__ == "__main__":
    main()
