"""
Label the whole corpus with a trained model, so all 1385 pages become
training data instead of only the 120 that were annotated by hand.

WHY THIS EXISTS RATHER THAN JUST AUTO-LABELLING EVERYTHING
----------------------------------------------------------
The obvious shortcut is to run the classical scan_doc_v2 classifier
over every page and train on its output. Measured on the sample, that
classifier over-calls `table` about 3x and finds almost no `figure`. A
model trained on those labels learns to reproduce them - it cannot come
out better than the labels it was given, so the result is a slower copy
of a classifier we already know is wrong.

Bootstrapping works instead because the model is trained on human
labels first. It generalises from correct examples, and its mistakes on
new pages are ordinary generalisation error rather than a systematic
bias baked into every page.

The loop:

    1. hand-annotate the 120-page sample        (CVAT)
    2. train on it                              (train_layout.py)
    3. pseudo-label the other ~1265 pages       (this script)
    4. review the low-confidence pages          (validate + CVAT)
    5. retrain on human + accepted pseudo labels

CONFIDENCE IS THE WHOLE POINT. Only detections at or above --conf are
kept, and pages where the model is unsure are listed for review rather
than silently trusted. Keeping everything would put the model's worst
guesses into its own training set, which is how bootstrapping turns
into drift.

Pages that already have human labels are never overwritten.

Run:
    python pseudo_label.py --weights runs/layout/weights/best.pt
    python pseudo_label.py --weights ... --conf 0.6 --review-out review.txt
"""

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

CORPUS_DIR = BASE_DIR.parent / "preprocessing" / "output"
MANIFEST_PATH = BASE_DIR / "manifest.csv"
OUTPUT_DIR = BASE_DIR / "preannotations"

CLASSES = ["paragraph", "math", "figure", "table", "code", "crossed_out"]

# Detections below this are dropped rather than written out as labels.
# 0.5 is a starting point, not a tuned value - check the review list
# and the class counts before trusting it.
DEFAULT_CONF = 0.5

# A page whose best detection is below this is flagged: the model had
# nothing it was confident about, which usually means the page is
# unusual rather than empty.
REVIEW_CONF = 0.35

# Pages with fewer boxes than this are also worth a look - a full page
# of writing that yields one box is more likely a miss than a page with
# one paragraph on it.
REVIEW_MIN_BOXES = 2


def corpus_pages():

    import re

    pages = []

    for path in sorted(CORPUS_DIR.glob("student_*/cie_*/page_*.png")):

        student = int(re.search(r"student_(\d+)", path.parts[-3]).group(1))
        cie = int(re.search(r"cie_(\d+)", path.parts[-2]).group(1))
        number = int(re.search(r"(\d+)", path.stem).group(1))

        pages.append((f"s{student:02d}_c{cie}_p{number:02d}.png", path))

    return pages


def human_labelled(labels_path):
    """File names that already carry hand-made labels."""

    if not labels_path or not Path(labels_path).exists():
        return set()

    with open(labels_path) as handle:
        coco = json.load(handle)

    return {image["file_name"] for image in coco.get("images", [])}


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--human",
        help="COCO of hand labels; those pages are skipped, not overwritten",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", default="pseudo_labels_coco.json")
    parser.add_argument("--review-out", default="pseudo_review.txt")

    args = parser.parse_args()

    weights = Path(args.weights)

    if not weights.exists():
        sys.exit(
            f"{weights} not found.\n"
            f"Bootstrapping needs a model trained on human labels first - "
            f"see the docstring."
        )

    pages = corpus_pages()

    skip = human_labelled(args.human)

    if skip:
        pages = [(name, path) for name, path in pages if name not in skip]
        print(f"Skipping {len(skip)} page(s) that already have human labels")

    if args.limit:
        pages = pages[:args.limit]

    if not pages:
        sys.exit("no pages left to label")

    from ultralytics import YOLO

    model = YOLO(str(weights))

    images = []
    annotations = []

    counts = Counter()
    review = []

    print(f"Pages: {len(pages)}   conf>={args.conf}   imgsz={args.imgsz}")

    for image_id, (name, path) in enumerate(pages, start=1):

        if image_id % 100 == 0:
            print(f"  {image_id}/{len(pages)}", flush=True)

        result = model.predict(
            str(path), imgsz=args.imgsz, conf=args.conf, verbose=False
        )[0]

        height, width = result.orig_shape

        images.append(
            {
                "id": image_id,
                "file_name": name,
                "width": int(width),
                "height": int(height),
            }
        )

        best = 0.0
        kept = 0

        for box in result.boxes:

            score = float(box.conf)
            best = max(best, score)

            label = CLASSES[int(box.cls)]

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])

            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": CLASSES.index(label) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "segmentation": [],
                    # kept so a later pass can raise the bar without
                    # re-running inference over the whole corpus
                    "score": round(score, 4),
                }
            )

            counts[label] += 1
            kept += 1

        if best < REVIEW_CONF or kept < REVIEW_MIN_BOXES:
            review.append((name, round(best, 3), kept))

    coco = {
        "info": {
            "description": (
                "PSEUDO-LABELS from a trained model, not human ground "
                f"truth. conf>={args.conf}, weights={weights.name}. "
                "Review before treating as truth."
            ),
            "date_created": str(date.today()),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": name, "supercategory": ""}
            for index, name in enumerate(CLASSES)
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUTPUT_DIR / args.out

    with open(out_path, "w") as handle:
        json.dump(coco, handle, indent=1)

    review_path = OUTPUT_DIR / args.review_out

    with open(review_path, "w") as handle:
        handle.write(
            "# Pages the model was unsure about. Annotate these by hand\n"
            "# before retraining - they are where pseudo-labelling drifts.\n"
            "# file_name  best_conf  boxes_kept\n"
        )
        for name, best, kept in review:
            handle.write(f"{name}\t{best}\t{kept}\n")

    print(f"\nPages    : {len(images)}")
    print(f"Boxes    : {len(annotations)}  "
          f"({len(annotations) / max(1, len(images)):.1f} per page)")

    print("\nClass distribution:")
    total = sum(counts.values())
    for name in CLASSES:
        count = counts.get(name, 0)
        share = 100 * count / total if total else 0
        print(f"  {name:<12} {count:>6}  {share:5.1f}%")

    print(f"\nFlagged for review: {len(review)} "
          f"({100 * len(review) / max(1, len(images)):.0f}% of pages)")

    print(f"\nLabels : {out_path}")
    print(f"Review : {review_path}")

    print(
        "\nDo not retrain on this blindly. Work through the review list "
        "first - those pages are where the model is weakest, and they are "
        "exactly the ones that would teach it its own mistakes."
    )


if __name__ == "__main__":
    main()
