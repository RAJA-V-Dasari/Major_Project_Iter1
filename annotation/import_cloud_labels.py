"""
Bring cloud-annotated labels back to full resolution.

export_for_cloud.py downscales pages to ~1400px so they upload in
minutes instead of hours. Boxes therefore come back in that smaller
coordinate space, and pasting them straight into training would make
every region ~20% too small. Nothing errors, the labels just look
plausible and are quietly wrong - which is far worse than a crash.

This maps every box back using the per-page factor recorded in
scales.json, renames pages from .jpg to the .png the corpus uses, and
checks the result actually lands inside the full-resolution page.

Run:
    python import_cloud_labels.py annotations.json
    python import_cloud_labels.py annotations.json --out labels/cloud.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SCALES_PATH = BASE_DIR / "cloud_upload" / "scales.json"

DEFAULT_OUT = BASE_DIR / "labels" / "cloud_labels.json"

CLASSES = ["paragraph", "math", "figure", "table", "code", "crossed_out"]


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("labels", help="COCO json exported from the tool")
    parser.add_argument("--out", default=str(DEFAULT_OUT))

    args = parser.parse_args()

    label_path = Path(args.labels)

    if not label_path.exists():
        sys.exit(f"{label_path} not found")

    if not SCALES_PATH.exists():
        sys.exit(
            f"{SCALES_PATH} not found - it is written by "
            f"export_for_cloud.py and is required to undo the downscale"
        )

    with open(label_path) as handle:
        coco = json.load(handle)

    with open(SCALES_PATH) as handle:
        scales = json.load(handle)

    categories = {c["id"]: c["name"] for c in coco["categories"]}

    unknown = set(categories.values()) - set(CLASSES)

    if unknown:
        sys.exit(f"labels outside the schema: {sorted(unknown)}")

    images = {}

    missing = []

    for image in coco["images"]:

        name = Path(image["file_name"]).name

        record = scales.get(name)

        if record is None:
            missing.append(name)
            continue

        images[image["id"]] = (name, record)

        # rewrite to the corpus filename and true size
        image["file_name"] = name.replace(".jpg", ".png")
        image["width"] = record["full_width"]
        image["height"] = record["full_height"]

    if missing:
        sys.exit(
            f"{len(missing)} annotated page(s) are not in scales.json, so "
            f"their scale is unknown: {missing[:5]}"
        )

    counts = Counter()

    clipped = 0

    for ann in coco["annotations"]:

        name, record = images[ann["image_id"]]

        factor = record["to_full"]

        x, y, w, h = ann["bbox"]

        x1, y1 = x * factor, y * factor
        x2, y2 = (x + w) * factor, (y + h) * factor

        # rounding at the edges can push a box a pixel past the page
        full_w = record["full_width"]
        full_h = record["full_height"]

        before = (x1, y1, x2, y2)

        x1 = max(0.0, min(x1, full_w))
        y1 = max(0.0, min(y1, full_h))
        x2 = max(0.0, min(x2, full_w))
        y2 = max(0.0, min(y2, full_h))

        if (x1, y1, x2, y2) != before:
            clipped += 1

        ann["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        ann["area"] = (x2 - x1) * (y2 - y1)

        counts[categories[ann["category_id"]]] += 1

    coco["info"] = {
        "description": (
            "Cloud-annotated layout labels, rescaled to full "
            "resolution via cloud_upload/scales.json."
        ),
    }

    out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as handle:
        json.dump(coco, handle, indent=1)

    print(f"Pages  : {len(coco['images'])}")
    print(f"Boxes  : {len(coco['annotations'])}")

    print("\nClass distribution:")
    total = sum(counts.values()) or 1
    for name in CLASSES:
        print(f"  {name:<12} {counts.get(name, 0):>5}  "
              f"{100 * counts.get(name, 0) / total:5.1f}%")

    if clipped:
        print(f"\n{clipped} box(es) clipped to the page edge (rounding).")

    print(f"\nWritten: {out_path}")
    print("Now run:  python validate_labels.py " + str(out_path))


if __name__ == "__main__":
    main()
