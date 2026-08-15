"""
Move hand-made boxes onto the cleaned pages.

clean_pages.py rotates and crops, so every existing coordinate is stale
by up to a few degrees and a few dozen pixels. Nothing errors if this
step is skipped - the boxes simply sit slightly wrong, which is the
worst kind of bug in training data because it looks fine and quietly
degrades the model.

A rotated box is no longer axis-aligned, so each of its four corners is
mapped and the axis-aligned bounding box of the result is taken. That
grows the box slightly on tilted pages; for layout regions, which are
approximate by design, that is the right trade against carrying
rotation through the whole pipeline.

Run:
    python remap_annotations.py
    python remap_annotations.py --check      # also draw 4 pages to verify
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

SOURCE_LABELS = BASE_DIR / "labels" / "instances_default.json"
OUT_LABELS = BASE_DIR / "labels" / "instances_cleaned.json"

CLEAN_DIR = BASE_DIR.parent / "preprocessing" / "cleaned"
TRANSFORMS_PATH = CLEAN_DIR / "transforms.json"

CHECK_DIR = BASE_DIR / "remap_check"

CLASSES = ["paragraph", "math", "figure", "table", "code", "crossed_out"]

# Must match clean.UNDERSIZED_FRACTION.
UNDERSIZED_FRACTION = 0.90


def corpus_key(file_name):
    """s01_c1_p06.png -> student_01/cie_1/page_06.png"""

    match = re.fullmatch(r"s(\d+)_c(\d+)_p(\d+)\.png", file_name)

    if not match:
        return None

    student, cie, page = (int(g) for g in match.groups())

    return f"student_{student:02d}/cie_{cie}/page_{page:02d}.png"


def rotation_matrix(angle, width, height):
    """Same matrix OpenCV used to rotate the page."""

    centre_x, centre_y = width / 2.0, height / 2.0

    radians = np.deg2rad(angle)

    cos, sin = np.cos(radians), np.sin(radians)

    # cv2.getRotationMatrix2D(centre, angle, 1.0)
    return np.array([
        [cos, sin, (1 - cos) * centre_x - sin * centre_y],
        [-sin, cos, sin * centre_x + (1 - cos) * centre_y],
    ], dtype=np.float64)


def page_scale(transform):
    """
    Scale factor clean.py applied before padding.

    Recomputed here rather than read from the transform: clean.py only
    resizes a page that arrives far smaller than canonical (one page in
    the corpus does), and the rule it uses is deterministic, so
    deriving it keeps the two files from drifting apart.
    """

    target_w, target_h = transform["final_size"]

    width, height = transform["trimmed_size"]

    if (width < UNDERSIZED_FRACTION * target_w
            or height < UNDERSIZED_FRACTION * target_h):
        return min(target_w / width, target_h / height)

    return 1.0


def remap_box(bbox, transform):
    """
    COCO [x,y,w,h] on the original page -> [x,y,w,h] on the cleaned one.

    Applies the same chain clean.py did, in the same order: rotate
    about the source centre, subtract the crop origin, scale, then add
    the padding offset. Getting the order wrong shifts boxes by tens of
    pixels without erroring.
    """

    x, y, w, h = bbox

    source_w, source_h = transform["source_size"]
    final_w, final_h = transform["final_size"]

    crop_x, crop_y = transform["crop"][0], transform["crop"][1]
    pad_x, pad_y = transform["pad_offset"]

    corners = np.array([
        [x, y], [x + w, y], [x + w, y + h], [x, y + h],
    ], dtype=np.float64)

    angle = transform["angle"]

    if abs(angle) > 0:

        matrix = rotation_matrix(angle, source_w, source_h)

        corners = np.hstack([corners, np.ones((4, 1))]) @ matrix.T

    corners[:, 0] -= crop_x
    corners[:, 1] -= crop_y

    corners *= page_scale(transform)

    corners[:, 0] += pad_x
    corners[:, 1] += pad_y

    x1 = float(np.clip(corners[:, 0].min(), 0, final_w))
    y1 = float(np.clip(corners[:, 1].min(), 0, final_h))
    x2 = float(np.clip(corners[:, 0].max(), 0, final_w))
    y2 = float(np.clip(corners[:, 1].max(), 0, final_h))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2 - x1, y2 - y1]


def draw_check(coco, count=4):
    """Render a few remapped pages so the alignment can be eyeballed."""

    import cv2

    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    images = {i["id"]: i for i in coco["images"]}

    grouped = {}

    for ann in coco["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)

    drawn = 0

    for image_id, image in images.items():

        if drawn >= count:
            break

        key = corpus_key(image["file_name"])

        path = CLEAN_DIR / key

        if not path.exists():
            continue

        page = cv2.imread(str(path))

        if page is None:
            continue

        for ann in grouped.get(image_id, []):

            x, y, w, h = (int(v) for v in ann["bbox"])

            cv2.rectangle(page, (x, y), (x + w, y + h), (0, 0, 220), 5)

        cv2.imwrite(str(CHECK_DIR / image["file_name"]), page)

        drawn += 1

    print(f"\n{drawn} check image(s) -> {CHECK_DIR}")


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--check", action="store_true")

    args = parser.parse_args()

    if not SOURCE_LABELS.exists():
        sys.exit(f"{SOURCE_LABELS} not found")

    if not TRANSFORMS_PATH.exists():
        sys.exit(f"{TRANSFORMS_PATH} not found - run clean_pages.py first")

    with open(SOURCE_LABELS) as handle:
        coco = json.load(handle)

    with open(TRANSFORMS_PATH) as handle:
        transforms = json.load(handle)

    missing = []

    kept_images = []

    id_to_transform = {}

    for image in coco["images"]:

        key = corpus_key(image["file_name"])

        transform = transforms.get(key) if key else None

        if transform is None:
            missing.append(image["file_name"])
            continue

        image["width"], image["height"] = transform["final_size"]

        id_to_transform[image["id"]] = transform

        kept_images.append(image)

    coco["images"] = kept_images

    remapped = []

    dropped = 0

    max_shift = 0.0

    for ann in coco["annotations"]:

        transform = id_to_transform.get(ann["image_id"])

        if transform is None:
            continue

        before = ann["bbox"]

        bbox = remap_box(before, transform)

        if bbox is None:
            dropped += 1
            continue

        shift = max(abs(bbox[0] - before[0]), abs(bbox[1] - before[1]))

        max_shift = max(max_shift, shift)

        ann["bbox"] = bbox
        ann["area"] = bbox[2] * bbox[3]

        remapped.append(ann)

    coco["annotations"] = remapped

    coco["info"] = {
        "description": (
            "Hand-made layout labels, remapped onto the cleaned "
            "(deskewed + cropped) pages by remap_annotations.py."
        ),
    }

    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_LABELS, "w") as handle:
        json.dump(coco, handle, indent=1)

    print(f"Pages   : {len(coco['images'])}")
    print(f"Boxes   : {len(remapped)}")
    print(f"Dropped : {dropped} (fell outside the cropped page)")
    print(f"Largest corner shift: {max_shift:.1f} px")

    if missing:
        print(f"\n{len(missing)} page(s) had no transform: {missing[:5]}")

    print(f"\nWritten: {OUT_LABELS}")

    if args.check:
        draw_check(coco)


if __name__ == "__main__":
    main()
