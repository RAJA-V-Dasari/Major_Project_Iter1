"""
Check an exported annotation file against the rules in
LABELING_GUIDE.md, before it is used to train anything.

Annotation errors are cheap to fix now and expensive later: a model
trained on inconsistent labels learns the inconsistency, and the
resulting score is not a measurement of anything. Each check below
corresponds to a rule in the guide, so a failure names the rule it
broke.

The two structural checks are the ones worth having:

  * `crossed_out` is an OVERLAPPING layer. A crossed_out box that
    overlaps nothing means the underlying region was deleted rather
    than kept, which loses the surrounding good text.

  * Text inside a figure STAYS inside the figure. A paragraph box
    nested inside a figure box is the single most common way these
    annotations go wrong, so it is detected explicitly.

Exit status is non-zero if any error (not warning) is found, so this
can gate a training run.

Run:
    python validate_labels.py labels/instances_default.json
    python validate_labels.py labels/... --strict   # warnings fail too
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = BASE_DIR / "manifest.csv"

SCHEMA = ["paragraph", "math", "figure", "table", "code", "crossed_out"]

OVERLAY_CLASS = "crossed_out"

# A box smaller than this is almost certainly a slip of the mouse
# rather than a region. Guide says a region is about a text line tall;
# crossed_out is exempt because a struck word is legitimately small.
MIN_SIDE_PX = 12
MIN_AREA_PX = 400

# Share of the smaller box that must sit inside the larger one before
# the two count as "nested". Deliberately high - regions legitimately
# touch and slightly overlap; near-total containment is the error.
NESTING_IOA = 0.80

# A crossed_out box must overlap some region by at least this share of
# itself, or nothing was labelled underneath it.
OVERLAY_MIN_IOA = 0.30


def intersection_over_area(a, b):
    """Share of box `a` that lies inside box `b`."""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    area = (ax2 - ax1) * (ay2 - ay1)

    if area <= 0:
        return 0.0

    return ((ix2 - ix1) * (iy2 - iy1)) / area


def load_manifest():

    if not MANIFEST_PATH.exists():
        return {}

    with open(MANIFEST_PATH) as handle:
        return {row["file_name"]: row for row in csv.DictReader(handle)}


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("labels", help="COCO json exported from the tool")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as errors")

    args = parser.parse_args()

    path = Path(args.labels)

    if not path.exists():
        sys.exit(f"{path} not found")

    with open(path) as handle:
        coco = json.load(handle)

    manifest = load_manifest()

    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}

    images = {img["id"]: img for img in coco.get("images", [])}

    errors = []
    warnings = []

    # --- schema ----------------------------------------------------
    named = set(categories.values())

    unknown = named - set(SCHEMA)
    unused = set(SCHEMA) - named

    if unknown:
        errors.append(
            f"labels not in the schema: {sorted(unknown)} "
            f"(schema is {SCHEMA})"
        )

    if unused:
        warnings.append(f"schema classes never defined in export: {sorted(unused)}")

    # --- group boxes by image --------------------------------------
    by_image = defaultdict(list)

    for ann in coco.get("annotations", []):

        x, y, w, h = ann["bbox"]

        by_image[ann["image_id"]].append(
            {
                "id": ann["id"],
                "label": categories.get(ann["category_id"], "?"),
                "box": (x, y, x + w, y + h),
                "w": w,
                "h": h,
            }
        )

    counts = Counter()

    empty_pages = []

    for image_id, image in images.items():

        name = image["file_name"]

        boxes = by_image.get(image_id, [])

        if not boxes:
            empty_pages.append(name)
            continue

        regions = [b for b in boxes if b["label"] != OVERLAY_CLASS]
        overlays = [b for b in boxes if b["label"] == OVERLAY_CLASS]

        for box in boxes:

            counts[box["label"]] += 1

            x1, y1, x2, y2 = box["box"]

            # --- geometry ---
            if x2 <= x1 or y2 <= y1:
                errors.append(f"{name}: box #{box['id']} has non-positive area")
                continue

            if (
                x1 < -1 or y1 < -1
                or x2 > image["width"] + 1
                or y2 > image["height"] + 1
            ):
                errors.append(
                    f"{name}: box #{box['id']} ({box['label']}) "
                    f"extends outside the image"
                )

            if box["label"] != OVERLAY_CLASS:

                if min(box["w"], box["h"]) < MIN_SIDE_PX or \
                        box["w"] * box["h"] < MIN_AREA_PX:
                    warnings.append(
                        f"{name}: box #{box['id']} ({box['label']}) is tiny "
                        f"({int(box['w'])}x{int(box['h'])}px) - a slip?"
                    )

        # --- crossed_out must overlay something --------------------
        for overlay in overlays:

            best = max(
                (intersection_over_area(overlay["box"], r["box"])
                 for r in regions),
                default=0.0,
            )

            if best < OVERLAY_MIN_IOA:
                errors.append(
                    f"{name}: crossed_out box #{overlay['id']} overlaps no "
                    f"region (best {best:.2f}). The guide keeps the "
                    f"underlying region and adds crossed_out on top; this "
                    f"looks like the region was deleted instead."
                )

        # --- figure/table must not have their contents carved out --
        for inner in regions:

            for outer in regions:

                if inner is outer:
                    continue

                if outer["label"] not in ("figure", "table"):
                    continue

                if inner["label"] in ("figure", "table"):
                    continue

                if intersection_over_area(inner["box"], outer["box"]) >= NESTING_IOA:
                    errors.append(
                        f"{name}: {inner['label']} box #{inner['id']} sits "
                        f"inside {outer['label']} box #{outer['id']}. Text "
                        f"inside a {outer['label']} stays part of it - do "
                        f"not carve it out."
                    )

    # --- coverage against the manifest -----------------------------
    annotated = {images[i]["file_name"] for i in images}

    if manifest:

        missing = sorted(set(manifest) - annotated)

        if missing:
            warnings.append(
                f"{len(missing)} sampled page(s) not in this export "
                f"(annotation incomplete): {missing[:5]}"
            )

        stray = sorted(annotated - set(manifest))

        if stray:
            warnings.append(
                f"{len(stray)} annotated page(s) are not in the manifest: "
                f"{stray[:5]}"
            )

    # --- report ----------------------------------------------------
    print(f"File      : {path}")
    print(f"Pages     : {len(images)}")
    print(f"Boxes     : {sum(counts.values())}")

    if counts:
        print("\nClass distribution:")
        total = sum(counts.values())
        for name in SCHEMA:
            count = counts.get(name, 0)
            share = 100 * count / total if total else 0
            print(f"  {name:<12} {count:>5}  {share:5.1f}%")

    if manifest:
        per_split = Counter(
            manifest[n]["split"] for n in annotated if n in manifest
        )
        print("\nPages annotated per split:")
        for split in ("train", "val", "test"):
            print(f"  {split:<6} {per_split.get(split, 0):>4}")

    if empty_pages:
        print(f"\n{len(empty_pages)} page(s) with no boxes "
              f"(genuinely blank, or not yet done):")
        for name in empty_pages[:10]:
            print(f"  {name}")

    if warnings:
        print(f"\n--- {len(warnings)} WARNING(S) ---")
        for item in warnings:
            print(" ?", item)

    if errors:
        print(f"\n--- {len(errors)} ERROR(S) ---")
        for item in errors[:40]:
            print(" *", item)
        if len(errors) > 40:
            print(f"   ... and {len(errors) - 40} more")

    if errors or (args.strict and warnings):
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
