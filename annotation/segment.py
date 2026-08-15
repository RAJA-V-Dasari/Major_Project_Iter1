"""
Segment every cleaned page: COCO JSON plus a colour-coded image.

Cover pages (`page_01`) are excluded - they hold the identity block,
signatures and marks, not answer content.

COVERAGE FIRST
--------------
Each region goes on to OCR and then to document reconstruction, so a
region that is never boxed is content that is silently lost. That is
far worse than a region with the wrong class, which a later stage can
still re-read. The trained model on its own does not clear that bar: on
some pages it boxes almost nothing.

So geometry comes from the CLASSICAL segmenter (projection profiling,
which finds ink wherever it sits) and the class from the model,
matched to each block by overlap. Model boxes covering something the
classical pass missed - usually diagrams, which have no ruled-line
structure to profile - are added as their own regions. The union is
biased hard toward recall.

`ink_coverage` reports the share of ink that ended up inside some box.
That is the number that answers "will OCR see everything", so it is
printed rather than left to be assumed.

Hand-labelled pages use their own boxes; every box records whether it
came from a human or the model.

Run:
    python segment.py                    # every content page
    python segment.py --count 50         # a sample
    python segment.py --no-images        # JSON only
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

CORPUS_DIR = BASE_DIR.parent / "preprocessing" / "cleaned"

HUMAN_LABELS = BASE_DIR / "labels" / "instances_cleaned.json"

OUT_DIR = BASE_DIR / "segmented"
IMAGE_OUT = OUT_DIR / "images"

DEFAULT_WEIGHTS = BASE_DIR / "runs" / "seg" / "weights" / "best.pt"

CLASSES = ["paragraph", "math", "figure", "table", "code", "crossed_out"]

DEFAULT_LABEL = "paragraph"

COVER_PAGE = 1

# Dark, mutually distinguishable, and none close to the black ink they
# are drawn over. BGR for OpenCV.
COLOURS = {
    "paragraph":   (32, 94, 27),      # #1B5E20 dark green
    "math":        (9, 83, 180),      # #B45309 dark amber
    "figure":      (28, 28, 183),     # #B71C1C dark red
    "table":       (140, 20, 74),     # #4A148C dark purple
    "code":        (92, 105, 0),      # #00695C dark teal
    "crossed_out": (79, 14, 136),     # #880E4F dark crimson
}

HEADER_HEIGHT = 62
BOX_THICKNESS = 5
JPEG_QUALITY = 92

# Cleaned pages put paper at 255, so anything well below is ink.
INK_THRESHOLD = 160

# A model box must cover this much of a classical block before its
# class is believed.
CLASS_MIN_OVERLAP = 0.35

# A model box this uncovered by classical blocks is a region the
# classical pass missed, and is kept as its own box.
NEW_REGION_MAX_OVERLAP = 0.55


def content_pages():

    by_student = defaultdict(list)

    covers = 0

    for path in sorted(CORPUS_DIR.glob("student_*/cie_*/page_*.png")):

        student = int(re.search(r"student_(\d+)", path.parts[-3]).group(1))
        cie = int(re.search(r"cie_(\d+)", path.parts[-2]).group(1))
        number = int(re.search(r"(\d+)", path.stem).group(1))

        if number == COVER_PAGE:
            covers += 1
            continue

        by_student[student].append(
            (f"s{student:02d}_c{cie}_p{number:02d}.png", path)
        )

    return by_student, covers


def round_robin(by_student, count):
    """One page per student per pass, so a partial run spans all writers."""

    students = sorted(by_student)

    picked = []
    depth = 0

    while len(picked) < count:

        added = False

        for student in students:

            if len(picked) >= count:
                break

            if depth < len(by_student[student]):
                picked.append(by_student[student][depth])
                added = True

        if not added:
            break

        depth += 1

    return picked


def load_human():

    if not HUMAN_LABELS.exists():
        return {}

    with open(HUMAN_LABELS) as handle:
        coco = json.load(handle)

    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i["file_name"] for i in coco["images"]}

    grouped = defaultdict(list)

    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        grouped[images[ann["image_id"]]].append(
            (names[ann["category_id"]], [x, y, x + w, y + h])
        )

    return dict(grouped)


def overlap_ratio(inner, outer):
    """Share of `inner` inside `outer`."""

    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    area = (ax2 - ax1) * (ay2 - ay1)

    return 0.0 if area <= 0 else ((ix2 - ix1) * (iy2 - iy1)) / area


def ink_coverage(gray, boxes):
    """Share of ink pixels inside some box - uncovered ink never reaches OCR."""

    ink = gray < INK_THRESHOLD

    total = int(ink.sum())

    if total == 0:
        return 1.0

    covered = np.zeros(gray.shape, bool)

    for _, (x1, y1, x2, y2) in boxes:
        covered[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = True

    return float((ink & covered).sum()) / total


def hybrid_boxes(path, model, imgsz, conf):
    """Classical geometry for coverage, model for the class name."""

    from preannotate import page_boxes

    try:
        blocks, _, _ = page_boxes(path, use_classifier=False, want_image=False)
    except Exception:
        blocks = []

    result = model.predict(str(path), imgsz=imgsz, conf=conf, verbose=False)[0]

    predictions = []

    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        predictions.append(
            (CLASSES[int(box.cls)], [x1, y1, x2, y2], float(box.conf))
        )

    boxes, scores = [], []

    for _, bx1, by1, bx2, by2 in blocks:

        block = [float(bx1), float(by1), float(bx2), float(by2)]

        best_label, best_share, best_score = DEFAULT_LABEL, 0.0, None

        for label, prediction, score in predictions:

            share = overlap_ratio(block, prediction)

            if share > best_share:
                best_share, best_label, best_score = share, label, score

        # a weak overlap is not evidence of class; fall back to the
        # majority class rather than inventing one
        if best_share < CLASS_MIN_OVERLAP:
            best_label, best_score = DEFAULT_LABEL, None

        boxes.append((best_label, block))
        scores.append(round(best_score, 4) if best_score else None)

    for label, prediction, score in predictions:

        if max((overlap_ratio(prediction, b) for _, b in boxes),
               default=0.0) < NEW_REGION_MAX_OVERLAP:

            boxes.append((label, prediction))
            scores.append(round(score, 4))

    return boxes, scores


def draw_legend(canvas, present):

    if not present:
        return

    pad, row, width = 12, 34, 250

    height = pad * 2 + row * len(present)

    x2 = canvas.shape[1] - 20
    x1 = x2 - width
    y1 = HEADER_HEIGHT + 20
    y2 = y1 + height

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), 2)

    for index, label in enumerate(present):

        cy = y1 + pad + row * index + 8

        cv2.rectangle(canvas, (x1 + pad, cy), (x1 + pad + 26, cy + 20),
                      COLOURS[label], -1)
        cv2.putText(canvas, label, (x1 + pad + 36, cy + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)


def render(page, boxes, source, coverage, out_path):

    height, width = page.shape[:2]

    canvas = np.full((height + HEADER_HEIGHT, width, 3), 255, np.uint8)

    canvas[HEADER_HEIGHT:] = (
        page if page.ndim == 3 else cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    )

    for label, (x1, y1, x2, y2) in boxes:

        colour = COLOURS.get(label, (0, 0, 0))

        top = int(y1) + HEADER_HEIGHT
        bottom = int(y2) + HEADER_HEIGHT

        cv2.rectangle(canvas, (int(x1), top), (int(x2), bottom),
                      colour, BOX_THICKNESS)

        tag_w = 20 + 19 * len(label)
        tag_top = max(HEADER_HEIGHT, top - 40)

        cv2.rectangle(canvas, (int(x1), tag_top),
                      (int(x1) + tag_w, tag_top + 36), colour, -1)
        cv2.putText(canvas, label, (int(x1) + 10, tag_top + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)

    draw_legend(canvas, [c for c in CLASSES if any(b[0] == c for b in boxes)])

    cv2.rectangle(canvas, (0, 0), (width, HEADER_HEIGHT), (35, 35, 35), -1)

    cv2.putText(
        canvas,
        f"{out_path.stem}   |   {len(boxes)} regions   |   "
        f"{source.upper()}   |   ink covered {100 * coverage:.0f}%",
        (18, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
    )

    cv2.imwrite(str(out_path), canvas,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--count", type=int)
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-images", action="store_true")

    args = parser.parse_args()

    weights = Path(args.weights)

    if not weights.exists():
        sys.exit(f"{weights} not found - train first")

    if not CORPUS_DIR.exists():
        sys.exit(f"{CORPUS_DIR} not found - run preprocessing/clean.py first")

    by_student, covers = content_pages()

    available = sum(len(v) for v in by_student.values())

    pages = round_robin(by_student, args.count or available)

    human = load_human()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_images:
        IMAGE_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Excluded  : {covers} cover page(s)")
    print(f"Segmenting: {len(pages)} of {available} content pages")
    print(f"Weights   : {weights}\n")

    from ultralytics import YOLO

    model = YOLO(str(weights))

    images, annotations = [], []

    counts = Counter()
    by_source = Counter()

    coverages = []
    empty = 0

    for index, (name, path) in enumerate(pages, start=1):

        if index % 100 == 0:
            print(f"  {index}/{len(pages)}", flush=True)

        if name in human:
            boxes = human[name]
            scores = [None] * len(boxes)
            source = "human"
        else:
            boxes, scores = hybrid_boxes(path, model, args.imgsz, args.conf)
            source = "model"

        by_source[source] += 1

        page = cv2.imread(str(path))

        if page is None:
            continue

        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

        coverage = ink_coverage(gray, boxes)

        coverages.append(coverage)

        if not boxes:
            empty += 1

        height, width = gray.shape

        image_id = len(images) + 1

        images.append({
            "id": image_id,
            "file_name": name,
            "width": width,
            "height": height,
            "source": source,
            "ink_coverage": round(coverage, 4),
        })

        for (label, (x1, y1, x2, y2)), score in zip(boxes, scores):

            record = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": CLASSES.index(label) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "segmentation": [],
                "source": source,
            }

            if score is not None:
                record["score"] = score

            annotations.append(record)

            counts[label] += 1

        if not args.no_images:
            render(page, boxes, source, coverage,
                   IMAGE_OUT / (Path(name).stem + ".jpg"))

    coco = {
        "info": {
            "description": (
                "Layout segmentation of the cleaned handwritten answer "
                "scripts. Cover pages excluded. source=human boxes are "
                "hand made; source=model are predicted and are not "
                "ground truth."
            ),
            "date_created": str(date.today()),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": n, "supercategory": ""}
            for i, n in enumerate(CLASSES)
        ],
    }

    json_path = OUT_DIR / "segmentation.json"

    with open(json_path, "w") as handle:
        json.dump(coco, handle)

    total = sum(counts.values()) or 1

    mean_cov = sum(coverages) / max(1, len(coverages))

    lines = [
        f"pages segmented : {len(images)}",
        f"  human labels  : {by_source['human']}",
        f"  model output  : {by_source['model']}",
        f"regions         : {sum(counts.values())} "
        f"({sum(counts.values()) / max(1, len(images)):.1f} per page)",
        f"pages with none : {empty}",
        "",
        f"ink coverage    : mean {100 * mean_cov:.1f}%  "
        f"worst {100 * min(coverages or [0]):.1f}%",
        f"pages under 90% : {sum(1 for c in coverages if c < 0.90)}",
        "",
        "class distribution:",
    ]

    for label in CLASSES:
        lines.append(
            f"  {label:<12} {counts.get(label, 0):>6}  "
            f"{100 * counts.get(label, 0) / total:5.1f}%"
        )

    summary = "\n".join(lines)

    (OUT_DIR / "summary.txt").write_text(summary + "\n")

    print("\n" + summary)
    print(f"\nJSON   : {json_path}")

    if not args.no_images:
        print(f"Images : {IMAGE_OUT}")


if __name__ == "__main__":
    main()
