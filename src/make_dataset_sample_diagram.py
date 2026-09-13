"""Generate golden-reference / defective-test-image sample figures.

For each configured sample, shows the clean golden reference on the left and
the defective test image on the right, with every annotated defect boxed and
called out with a small leader-line label (so the label text never covers the
(often tiny) defect region itself).
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PCB_ROOT = ROOT / "data" / "PCB_DATASET"
OUT_DIR = ROOT / "ieacon_pcb_hybrid_dino" / "figures"

BOX_COLOR = "#FFE100"  # bright yellow box outline, contrasts against the green PCB substrate
LABEL_BG_COLOR = "#E4572E"  # red-orange label background, distinct from the box outline
LABEL_TEXT_COLOR = "white"
LABEL_FONTSIZE = 9.5
LABEL_OFFSET = 46  # points

SAMPLES = [
    {
        "board_id": "04",
        "class_dir": "Missing_hole",
        "sample_id": "04_missing_hole_10",
        "out_name": "dataset_sample_pair_missing_hole.png",
    },
    {
        "board_id": "06",
        "class_dir": "Short",
        "sample_id": "06_short_08",
        "out_name": "dataset_sample_pair_short.png",
    },
]


def parse_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes


def pretty_class_name(name):
    return name.replace("_", " ").capitalize()


def make_figure(board_id, class_dir, sample_id, out_name):
    ref_path = PCB_ROOT / "PCB_USED" / f"{board_id}.JPG"
    test_path = PCB_ROOT / "images" / class_dir / f"{sample_id}.jpg"
    xml_path = PCB_ROOT / "Annotations" / class_dir / f"{sample_id}.xml"

    ref_img = Image.open(ref_path).convert("RGB")
    test_img = Image.open(test_path).convert("RGB")
    boxes = parse_boxes(xml_path)

    img_w, img_h = test_img.size
    cx0, cy0 = img_w / 2, img_h / 2
    lw = max(1.6, img_w / 1100)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    axes[0].imshow(ref_img)
    axes[0].set_title(f"Golden Reference — Board {board_id}", fontsize=15, pad=10)
    axes[0].axis("off")

    axes[1].imshow(test_img)
    for i, (name, xmin, ymin, xmax, ymax) in enumerate(boxes, start=1):
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor=BOX_COLOR, linewidth=lw,
        )
        axes[1].add_patch(rect)

        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        # Push the callout outward from image center so labels fan away
        # from the board interior instead of stacking on top of each other.
        dx = LABEL_OFFSET if cx <= cx0 else -LABEL_OFFSET
        dy = LABEL_OFFSET if cy <= cy0 else -LABEL_OFFSET
        ha = "left" if dx > 0 else "right"
        va = "bottom" if dy > 0 else "top"

        label = f"{pretty_class_name(name)} #{i}"
        axes[1].annotate(
            label,
            xy=(cx, cy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=LABEL_FONTSIZE,
            color=LABEL_TEXT_COLOR,
            ha=ha,
            va=va,
            bbox=dict(boxstyle="square,pad=0.22", facecolor=LABEL_BG_COLOR, edgecolor="none"),
            arrowprops=dict(arrowstyle="-", color=LABEL_BG_COLOR, linewidth=1.3, shrinkA=0, shrinkB=3),
        )

    axes[1].set_title(
        f"Defective Test Image — {len(boxes)} Annotated Defects", fontsize=15, pad=10
    )
    axes[1].axis("off")

    fig.suptitle(
        "Example Golden Reference and Annotated Defective PCB Image Pair",
        fontsize=17, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        make_figure(**sample)


if __name__ == "__main__":
    main()
