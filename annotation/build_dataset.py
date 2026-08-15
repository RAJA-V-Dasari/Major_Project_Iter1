"""
Turn corrected annotations into a YOLO training tree.

The split is taken from manifest.csv and never recomputed here. That
matters: the manifest splits by STUDENT, and re-splitting at page level
(the obvious thing to do at this stage) would put pages from one
booklet on both sides of the train/test line and inflate the score by
letting the model recognise handwriting it had already seen.

Emits detection boxes, not segmentation masks, matching how the pages
were annotated - the downstream consumer routes regions to handlers and
does not need pixel-accurate outlines.

Run:
    python build_dataset.py labels/instances_default.json
    python build_dataset.py labels/... --out ../dataset
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
MANIFEST_PATH = BASE_DIR / "manifest.csv"

DEFAULT_OUT = BASE_DIR / "dataset"

# Index here becomes the YOLO class id. Keep it identical to
# preannotate.CLASSES and stable forever - retraining with a different
# order silently relabels every existing weight file's outputs.
CLASSES = ["paragraph", "math", "figure", "table", "code", "crossed_out"]

SPLITS = ("train", "val", "test")


def load_manifest():

    with open(MANIFEST_PATH) as handle:
        return {row["file_name"]: row for row in csv.DictReader(handle)}


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("labels", help="COCO json exported from the tool")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--symlink", action="store_true",
                        help="link images instead of copying (saves disk, "
                             "breaks if the source moves)")
    parser.add_argument(
        "--images", default=str(IMAGE_DIR),
        help="directory of page images; point at images_cleaned/ when "
             "training on the cleaned corpus (labels must have been "
             "remapped to match)",
    )

    args = parser.parse_args()

    image_dir = Path(args.images)

    label_path = Path(args.labels)

    if not label_path.exists():
        sys.exit(f"{label_path} not found")

    with open(label_path) as handle:
        coco = json.load(handle)

    manifest = load_manifest()

    categories = {c["id"]: c["name"] for c in coco["categories"]}

    unknown = set(categories.values()) - set(CLASSES)

    if unknown:
        sys.exit(
            f"export contains labels outside the schema: {sorted(unknown)}\n"
            f"Run validate_labels.py first."
        )

    images = {img["id"]: img for img in coco["images"]}

    by_image = defaultdict(list)

    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    out_dir = Path(args.out)

    for split in SPLITS:
        for kind in ("images", "labels"):
            target = out_dir / kind / split
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)

    counts = Counter()
    per_split = Counter()
    boxes_per_split = Counter()

    skipped = []

    for image_id, image in images.items():

        name = image["file_name"]

        row = manifest.get(name)

        if row is None:
            skipped.append(name)
            continue

        split = row["split"]

        source = image_dir / name

        if not source.exists():
            skipped.append(name)
            continue

        target = out_dir / "images" / split / name

        if args.symlink:
            target.symlink_to(source.resolve())
        else:
            # resolve(): images/ holds symlinks into preprocessing/output
            shutil.copy(source.resolve(), target)

        width = image["width"]
        height = image["height"]

        lines = []

        for ann in by_image.get(image_id, []):

            label = categories[ann["category_id"]]

            x, y, w, h = ann["bbox"]

            # clip to the page before normalising - a box drawn slightly
            # past the edge would otherwise produce out-of-range
            # coordinates that YOLO silently accepts and trains on
            x1 = max(0.0, min(x, width))
            y1 = max(0.0, min(y, height))
            x2 = max(0.0, min(x + w, width))
            y2 = max(0.0, min(y + h, height))

            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height

            lines.append(
                f"{CLASSES.index(label)} "
                f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

            counts[label] += 1
            boxes_per_split[split] += 1

        # a page with no boxes still belongs in the tree: blank pages
        # are real, and YOLO reads an empty .txt as a valid negative
        (out_dir / "labels" / split / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

        per_split[split] += 1

    data_yaml = out_dir / "data.yaml"

    data_yaml.write_text(
        "# Generated by build_dataset.py - do not hand-edit.\n"
        "# Split comes from annotation/manifest.csv and is by student,\n"
        "# so no handwriting appears in more than one split.\n"
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )

    print(f"Output : {out_dir}")

    print("\nPages / boxes per split:")
    for split in SPLITS:
        print(f"  {split:<6} {per_split.get(split, 0):>4} pages  "
              f"{boxes_per_split.get(split, 0):>5} boxes")

    print("\nClass distribution:")
    total = sum(counts.values())
    for name in CLASSES:
        count = counts.get(name, 0)
        share = 100 * count / total if total else 0
        print(f"  {name:<12} {count:>5}  {share:5.1f}%")

    rare = [n for n in CLASSES if 0 < counts.get(n, 0) < 30]
    absent = [n for n in CLASSES if counts.get(n, 0) == 0]

    if absent:
        print(f"\nNo examples at all: {absent} - these cannot be learned. "
              f"Drop them from CLASSES or sample pages that contain them.")

    if rare:
        print(f"\nUnder 30 examples: {rare} - expect poor recall. "
              f"Targeted extra sampling beats more random pages.")

    if skipped:
        print(f"\n{len(skipped)} image(s) skipped (not in manifest or "
              f"missing from images/): {skipped[:5]}")

    print(f"\ndata.yaml : {data_yaml}")


if __name__ == "__main__":
    main()
