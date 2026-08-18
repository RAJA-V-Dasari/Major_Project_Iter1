"""
Move the hand-drawn layout boxes onto the pages the pipeline actually runs on.

    annotation/labels/instances_default.json   (boxes, raw 1700x2338 space)
    01_prepare/01_deskew/input/                (the raw pages themselves)
    02_segment/input/                          (the prepared pages, 1598x2177)
        -> 06_evaluation/output/annotations_prepared.json   COCO, prepared space
        -> 06_evaluation/output/registration.csv            per-page quality

WHY THIS STEP EXISTS AT ALL
---------------------------
`annotation/` is the only ground truth in the repo and no pipeline stage
can read it, because it is in the wrong coordinate space. The boxes were
drawn on the raw pages (1700x2338). `01_prepare` then rotated every page
by its own measured angle and cropped it to 1598x2177 anchored on the
detected paper edge. Both are per-page, so there is no constant offset to
subtract - every box is stale by a couple of degrees and a few dozen
pixels, and by a different amount on every page.

`annotation/remap_annotations.py` did this job for the *previous*
generation of the cleaning code, reading a `transforms.json` that stage
wrote. The current `01_prepare` does not write one, and its output is a
different size, so `labels/instances_cleaned.json` is stale too - it is
still 1700x2338. This recovers the transform instead of requiring it to
have been recorded.

HOW THE TRANSFORM IS RECOVERED
------------------------------
Not by generic image registration. That was tried first and does not
work: ECC over the ink masks converged to nonsense (aligned ink IoU
0.03-0.07) because handwriting is sparse, high-frequency and has no
gradient basin for a gradient method to descend.

Instead the pipeline's own two operations are reproduced in order:

1. **Rotation** is re-derived with `01_prepare/01_deskew`'s own
   `rule_angle()`, imported rather than reimplemented, so this cannot
   drift away from what the stage actually did.

2. **Translation** is then a pure 2-D shift - the crop - and is found by
   FFT cross-correlation of the two ink masks. A shift is the one thing
   correlation is reliable at, and having removed the rotation first
   there is nothing else left to solve.

Measured over the 38 annotated pages that have both a raw and a prepared
copy on disk: aligned ink IoU median 0.917, minimum 0.666, none below
0.6. The ceiling is well under 1.0 by construction - `03_tone` changes
stroke weight, so identical geometry still leaves ink disagreeing at the
edges - which is why the acceptance floor is a floor and not a target.

A page that registers worse than MIN_INK_IOU is DROPPED rather than
carried with a warning. Ground truth that is quietly 40px out is worse
than no ground truth, because every score computed from it looks
plausible.

ROTATED BOXES BECOME BIGGER BOXES
---------------------------------
A rotated rectangle is no longer axis-aligned, so all four corners are
mapped and the axis-aligned bounding box of the result is taken. On a
1-degree page that grows a full-width box by ~20px vertically. Layout
regions are approximate by design and the alternative - carrying
rotation through every consumer - costs far more than it buys.

Run:
    python register.py                # map every page that has both copies
    python register.py --check 6      # also draw N pages to look at
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES_DIR = STAGE_DIR.parent
REPO_DIR = MODULES_DIR.parent

# Imported, not copied - the angle here must be the angle 01_deskew used.
sys.path.insert(0, str(MODULES_DIR / "01_prepare" / "01_deskew" / "src"))
import deskew  # noqa: E402

ANNOTATION_DIR = REPO_DIR / "annotation"
LABELS_PATH = ANNOTATION_DIR / "labels" / "instances_default.json"
MANIFEST_PATH = ANNOTATION_DIR / "manifest.csv"

RAW_DIR = MODULES_DIR / "01_prepare" / "01_deskew" / "input"
PREPARED_DIR = MODULES_DIR / "02_segment" / "input"

OUT_DIR = STAGE_DIR / "output"
OUT_LABELS = OUT_DIR / "annotations_prepared.json"
OUT_REPORT = OUT_DIR / "registration.csv"
CHECK_DIR = STAGE_DIR / "check"

# Same cut 02_segment uses, so "ink" means the same thing on both sides.
INK_THRESHOLD = 180

# Below this the transform is not believed and the page is dropped. See
# the measurements above: the real distribution sits at 0.67-0.99, so
# this rejects a failure rather than trimming a tail.
MIN_INK_IOU = 0.55

CLASS_COLOURS = {
    "paragraph": (32, 140, 27),
    "math": (200, 120, 0),
    "figure": (0, 0, 220),
    "table": (160, 0, 160),
    "code": (0, 150, 150),
    "crossed_out": (0, 80, 255),
}


def ink(gray):
    return (gray < INK_THRESHOLD).astype(np.float32)


def best_shift(target, moving):
    """
    The (dx, dy) that puts `moving` onto `target`, by FFT cross-correlation.

    Both are float ink fields. Mean-subtracting first turns the product
    into a covariance, so a large blank margin cannot win by agreeing
    with another large blank margin.
    """

    height = max(target.shape[0], moving.shape[0])
    width = max(target.shape[1], moving.shape[1])

    a = np.zeros((height, width), np.float32)
    b = np.zeros((height, width), np.float32)

    a[:target.shape[0], :target.shape[1]] = target - target.mean()
    b[:moving.shape[0], :moving.shape[1]] = moving - moving.mean()

    correlation = np.fft.irfft2(
        np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=(height, width)
    )

    peak = np.unravel_index(np.argmax(correlation), correlation.shape)

    # the correlation is circular, so the far half of each axis is a
    # negative shift, not a large positive one
    dy = peak[0] - (height if peak[0] > height // 2 else 0)
    dx = peak[1] - (width if peak[1] > width // 2 else 0)

    return float(dx), float(dy)


def page_transform(raw_path, prepared_path):
    """
    The 2x3 affine taking raw page coordinates to prepared ones.

    Returns (matrix, angle, ink_iou) or None if either page is unreadable.
    """

    raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    prepared = cv2.imread(str(prepared_path), cv2.IMREAD_GRAYSCALE)

    if raw is None or prepared is None:
        return None

    angle, _ = deskew.rule_angle(raw)

    height, width = raw.shape

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

    rotated = cv2.warpAffine(
        raw, matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )

    dx, dy = best_shift(ink(prepared), ink(rotated))

    matrix = matrix.copy()
    matrix[0, 2] += dx
    matrix[1, 2] += dy

    warped = cv2.warpAffine(
        raw, matrix, (prepared.shape[1], prepared.shape[0]),
        flags=cv2.INTER_LINEAR, borderValue=255,
    )

    a = warped < INK_THRESHOLD
    b = prepared < INK_THRESHOLD

    union = (a | b).sum()

    return matrix, angle, float((a & b).sum() / union) if union else 0.0


def map_box(bbox, matrix, size):
    """A COCO [x, y, w, h] through the transform, as an axis-aligned box."""

    x, y, w, h = bbox

    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                       dtype=np.float64)

    moved = corners @ matrix[:, :2].T + matrix[:, 2]

    x1 = max(0.0, float(moved[:, 0].min()))
    y1 = max(0.0, float(moved[:, 1].min()))
    x2 = min(float(size[0]), float(moved[:, 0].max()))
    y2 = min(float(size[1]), float(moved[:, 1].max()))

    if x2 <= x1 or y2 <= y1:
        return None

    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def load_sources():
    """file_name -> corpus-relative path, from the annotation manifest."""

    with open(MANIFEST_PATH) as handle:
        return {row["file_name"]: row for row in csv.DictReader(handle)}


def draw_check(prepared_path, boxes, categories, target):
    """The prepared page with the mapped boxes on it, to judge by eye."""

    gray = cv2.imread(str(prepared_path), cv2.IMREAD_GRAYSCALE)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for box in boxes:
        name = categories[box["category_id"]]
        x, y, w, h = (int(round(v)) for v in box["bbox"])
        colour = CLASS_COLOURS.get(name, (0, 0, 0))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 3)
        cv2.putText(canvas, name, (x + 4, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), canvas)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=int, default=0, metavar="N",
                        help="also render N mapped pages to look at")
    args = parser.parse_args()

    if not LABELS_PATH.exists():
        raise SystemExit(f"labels not found: {LABELS_PATH}")

    for directory, what in ((RAW_DIR, "raw pages"),
                            (PREPARED_DIR, "prepared pages")):
        if not directory.exists():
            raise SystemExit(f"{what} not found: {directory}")

    with open(LABELS_PATH) as handle:
        coco = json.load(handle)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}

    by_image = {}
    for annotation in coco["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    sources = load_sources()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kept_images = []
    kept_annotations = []
    report = []

    print(f"{len(images)} annotated page(s) in {LABELS_PATH.name}\n")

    for image_id, image in sorted(images.items(), key=lambda kv: kv[1]["file_name"]):

        name = image["file_name"]
        row = sources.get(name)

        if row is None:
            report.append((name, "", "", "", "not in manifest"))
            continue

        raw_path = RAW_DIR / row["source"]
        prepared_path = PREPARED_DIR / row["source"]

        if not raw_path.exists():
            report.append((name, "", "", "", "raw page not on disk"))
            continue

        if not prepared_path.exists():
            report.append((name, "", "", "", "prepared page not on disk"))
            continue

        result = page_transform(raw_path, prepared_path)

        if result is None:
            report.append((name, "", "", "", "unreadable"))
            continue

        matrix, angle, iou = result

        if iou < MIN_INK_IOU:
            report.append((name, f"{angle:.3f}", "", f"{iou:.3f}",
                           "rejected: registration not trustworthy"))
            continue

        prepared = cv2.imread(str(prepared_path), cv2.IMREAD_GRAYSCALE)
        size = (prepared.shape[1], prepared.shape[0])

        mapped = []

        for annotation in by_image.get(image_id, []):

            bbox = map_box(annotation["bbox"], matrix, size)

            if bbox is None:
                continue

            mapped.append({
                "id": len(kept_annotations) + len(mapped) + 1,
                "image_id": image_id,
                "category_id": annotation["category_id"],
                "bbox": bbox,
                "area": round(bbox[2] * bbox[3], 2),
                "iscrowd": 0,
                "segmentation": [],
            })

        kept_annotations.extend(mapped)

        kept_images.append({
            "id": image_id,
            "file_name": name,
            "source": row["source"],
            "width": size[0],
            "height": size[1],
            "split": row["split"],
            "page_kind": row["page_kind"],
            "registration_angle": angle,
            "registration_ink_iou": round(iou, 4),
        })

        report.append((name, f"{angle:.3f}", str(len(mapped)),
                       f"{iou:.3f}", "ok"))

        if args.check and len(kept_images) <= args.check:
            draw_check(prepared_path, mapped, categories,
                       CHECK_DIR / name)

    out = {
        "info": {
            "description": "annotation/ layout boxes mapped into the "
                           "prepared (02_segment/input) coordinate space",
            "source_labels": str(LABELS_PATH.relative_to(REPO_DIR)),
            "min_ink_iou": MIN_INK_IOU,
        },
        "licenses": coco.get("licenses", []),
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": coco["categories"],
    }

    with open(OUT_LABELS, "w") as handle:
        json.dump(out, handle, indent=1)

    with open(OUT_REPORT, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "angle", "boxes", "ink_iou", "status"])
        writer.writerows(report)

    ok = [r for r in report if r[4] == "ok"]
    ious = [float(r[3]) for r in ok]

    counts = {}
    for annotation in kept_annotations:
        name = categories[annotation["category_id"]]
        counts[name] = counts.get(name, 0) + 1

    print(f"Mapped    : {len(ok)} page(s), {len(kept_annotations)} box(es)")

    if ious:
        print(f"Ink IoU   : median {np.median(ious):.3f}, "
              f"min {min(ious):.3f}")

    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14}{count}")

    dropped = [r for r in report if r[4] != "ok"]

    if dropped:
        reasons = {}
        for row in dropped:
            reasons[row[4]] = reasons.get(row[4], 0) + 1
        print(f"\nDropped   : {len(dropped)} page(s)")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<44}{count}")

    print(f"\nLabels    : {OUT_LABELS}")
    print(f"Report    : {OUT_REPORT}")

    if args.check:
        print(f"Check     : {CHECK_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
