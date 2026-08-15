"""
Render the finished labels for human review.

Machine geometry plus machine class names is a draft; machine geometry
plus *reviewed* class names is ground truth. This draws the second
thing, colour-coded, so the assignments can be checked before a model
is trained on them - an error caught here costs a minute, the same
error caught after training costs a training run and is easy to
mistake for a modelling problem.

Output stays on disk deliberately. These pages carry names, USNs,
signatures and marks; they must not be published anywhere.

Run:
    python review_labels.py
    python review_labels.py --per-sheet 4
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
LABELS_PATH = BASE_DIR / "labels" / "instances_default.json"
OUT_DIR = BASE_DIR / "review_labels"

# BGR. Chosen to stay distinct against blue/black ink on white paper.
COLOURS = {
    "paragraph": (0, 160, 0),
    "math": (220, 120, 0),
    "figure": (0, 0, 230),
    "table": (200, 0, 200),
    "code": (0, 175, 175),
    "crossed_out": (30, 30, 30),
}

CELL_WIDTH = 620


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--per-sheet", type=int, default=5)

    args = parser.parse_args()

    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} not found - run label_helper.py apply")

    with open(LABELS_PATH) as handle:
        coco = json.load(handle)

    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i["file_name"] for i in coco["images"]}

    grouped = {}

    for ann in coco["annotations"]:
        grouped.setdefault(images[ann["image_id"]], []).append(ann)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for stale in OUT_DIR.glob("*.png"):
        stale.unlink()

    cells = []

    tally = Counter()

    for name in sorted(grouped):

        source = IMAGE_DIR / name

        if not source.exists():
            continue

        page = cv2.imread(str(source))

        for ann in grouped[name]:

            label = names[ann["category_id"]]

            tally[label] += 1

            x, y, w, h = (int(v) for v in ann["bbox"])

            colour = COLOURS.get(label, (0, 0, 0))

            cv2.rectangle(page, (x, y), (x + w, y + h), colour, 5)

            cv2.rectangle(page, (x, max(0, y - 44)), (x + 24 * len(label), y),
                          colour, -1)
            cv2.putText(page, label, (x + 6, max(24, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        rgb = cv2.cvtColor(page, cv2.COLOR_BGR2RGB)

        thumb = Image.fromarray(rgb)

        thumb = thumb.resize(
            (CELL_WIDTH, int(thumb.height * CELL_WIDTH / thumb.width))
        )

        cells.append((name, thumb))

    sheets = 0

    for start in range(0, len(cells), args.per_sheet):

        batch = cells[start:start + args.per_sheet]

        height = max(c.height for _, c in batch)
        width = CELL_WIDTH * len(batch) + 8 * (len(batch) - 1)

        sheet = Image.new("RGB", (width, height), (110, 110, 110))

        for index, (_, cell) in enumerate(batch):
            sheet.paste(cell, (index * (CELL_WIDTH + 8), 0))

        sheets += 1

        sheet.save(OUT_DIR / f"review_{sheets:02d}.png")

    print(f"Pages : {len(cells)}")
    print(f"Boxes : {sum(tally.values())}")

    print("\nClass distribution:")
    total = sum(tally.values()) or 1
    for label in COLOURS:
        print(f"  {label:<12} {tally.get(label, 0):>4}  "
              f"{100 * tally.get(label, 0) / total:5.1f}%")

    print(f"\n{sheets} sheet(s) -> {OUT_DIR}")
    print("Local only - these pages contain student PII.")


if __name__ == "__main__":
    main()
