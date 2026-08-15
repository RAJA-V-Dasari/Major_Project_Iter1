"""
Assign classes to machine-generated boxes without a GUI.

The expensive half of annotation is finding regions, and preannotate.py
already does that acceptably. The half it cannot do is naming them -
measured: `table` over-called ~3x, `figure` barely found, `crossed_out`
0/18 precision. That half is a judgement call on a rendered page, which
does not need CVAT.

So: render each page with its boxes NUMBERED, look at it, and record
`page -> {box number: class}` in a small JSON file. This turns the
class assignment into data that can be reviewed and corrected, instead
of a mouse-driven session that leaves no trace.

Boxes left unassigned keep the default (`paragraph`). Numbers are the
box's index within its page in the source COCO, and are stable as long
as that file is not regenerated.

assignments.json format:

    {
      "s17_c2_p08.png": {"4": "figure", "5": "code", "6": "code"},

      // "2+3+4" merges those boxes into one region. Segmentation
      // splits a table across the text lines it is made of, and the
      // guide wants a table marked whole - without this, the labels
      // would contradict the guide they are supposed to follow.
      "s32_c2_p05.png": {"1": "paragraph", "2+3+4": "table"},

      // "drop" deletes a box: segmentation sometimes emits a sliver
      // over blank paper or bleed-through. Labelling that `paragraph`
      // would teach the model that empty paper is text.
      "s03_c1_p03.png": {"2": "drop"},

      // geometry too wrong to label - excluded rather than guessed at
      "s16_c1_p07.png": "skip"
    }

Run:
    python label_helper.py render --pages s01_c1_p06.png,s02_c1_p08.png
    python label_helper.py render --split train --start 0 --count 6
    python label_helper.py apply            # assignments.json -> COCO
    python label_helper.py status
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
MANIFEST_PATH = BASE_DIR / "manifest.csv"

PREANN_PATH = BASE_DIR / "preannotations" / "preannotations_coco.json"

SHEET_DIR = BASE_DIR / "preannotations" / "label_sheets"

ASSIGN_PATH = BASE_DIR / "assignments.json"
OUT_PATH = BASE_DIR / "labels" / "instances_default.json"

CLASSES = ["paragraph", "math", "figure", "table", "code", "crossed_out"]

DEFAULT_LABEL = "paragraph"

# Not a class: marks a box for deletion rather than labelling.
DROP = "drop"

SHEET_WIDTH = 900


def load_coco():

    with open(PREANN_PATH) as handle:
        return json.load(handle)


def boxes_by_page(coco):
    """file_name -> [(index, [x1,y1,x2,y2])], index 1-based within page."""

    images = {img["id"]: img["file_name"] for img in coco["images"]}

    grouped = {}

    for ann in coco["annotations"]:

        name = images[ann["image_id"]]

        x, y, w, h = ann["bbox"]

        grouped.setdefault(name, []).append([x, y, x + w, y + h])

    return {
        name: list(enumerate(boxes, start=1))
        for name, boxes in grouped.items()
    }


def load_assignments():

    if ASSIGN_PATH.exists():
        with open(ASSIGN_PATH) as handle:
            return json.load(handle)

    return {}


def render(args, coco):

    grouped = boxes_by_page(coco)

    manifest = {}

    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as handle:
            manifest = {r["file_name"]: r for r in csv.DictReader(handle)}

    if args.pages:
        names = [n.strip() for n in args.pages.split(",") if n.strip()]
    else:
        names = sorted(grouped)

        if args.split:
            names = [
                n for n in names
                if manifest.get(n, {}).get("split") == args.split
            ]

        names = names[args.start:args.start + args.count]

    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    for name in names:

        source = IMAGE_DIR / name

        if not source.exists():
            print(f"  ! {name} not in images/")
            continue

        image = cv2.imread(str(source))

        for index, (x1, y1, x2, y2) in grouped.get(name, []):

            x1, y1, x2, y2 = (int(v) for v in (x1, y1, x2, y2))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)

            # number in a filled tag, so it stays readable over ink
            cv2.rectangle(image, (x1, max(0, y1 - 46)), (x1 + 74, y1),
                          (0, 0, 255), -1)
            cv2.putText(image, str(index), (x1 + 8, max(26, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

        target = SHEET_DIR / name

        cv2.imwrite(str(target), image)

        print(f"  {target}   ({len(grouped.get(name, []))} boxes)")


def seed(args, coco):
    """
    Pre-fill assignments.json with a trained model's guesses.

    This is the accelerator. Assigning a class to every box from
    scratch is slow; CORRECTING a mostly-right guess is much faster,
    and gets faster still as the model improves. The reviewer stays in
    the loop on every box - nothing is accepted unseen - but the work
    changes from "decide" to "spot the wrong one".

    Each proposed box is matched to the model's best-overlapping
    prediction. Where the model predicts nothing, the box keeps the
    default rather than inventing a class.

    Pages already in assignments.json are never overwritten.
    """

    from ultralytics import YOLO

    grouped = boxes_by_page(coco)

    assignments = load_assignments()

    manifest = {}

    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as handle:
            manifest = {r["file_name"]: r for r in csv.DictReader(handle)}

    todo = [n for n in sorted(grouped) if n not in assignments]

    if args.split:
        todo = [
            n for n in todo
            if manifest.get(n, {}).get("split") == args.split
        ]

    todo = todo[:args.count]

    if not todo:
        print("nothing left to seed")
        return

    model = YOLO(args.weights)

    added = Counter()

    for name in todo:

        source = IMAGE_DIR / name

        if not source.exists():
            continue

        result = model.predict(
            str(source), imgsz=args.imgsz, conf=args.conf, verbose=False
        )[0]

        predicted = []

        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            predicted.append((CLASSES[int(box.cls)], (x1, y1, x2, y2)))

        page = {}

        for index, (bx1, by1, bx2, by2) in grouped[name]:

            best_label = None
            best_overlap = 0.0

            area = max(1.0, (bx2 - bx1) * (by2 - by1))

            for label, (px1, py1, px2, py2) in predicted:

                ix1, iy1 = max(bx1, px1), max(by1, py1)
                ix2, iy2 = min(bx2, px2), min(by2, py2)

                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                overlap = (ix2 - ix1) * (iy2 - iy1) / area

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = label

            # a weak overlap is not evidence; leave the default so the
            # reviewer sees an honest "unknown" rather than a guess
            label = best_label if best_overlap >= 0.35 else DEFAULT_LABEL

            page[str(index)] = label

            added[label] += 1

        assignments[name] = page

    with open(ASSIGN_PATH, "w") as handle:
        json.dump(assignments, handle, indent=2)

    print(f"Seeded {len(todo)} page(s): {todo}")
    print(f"Boxes  : {sum(added.values())}")

    for label, count in added.most_common():
        print(f"  {label:<12} {count:>4}")

    print(
        f"\nThese are GUESSES written into {ASSIGN_PATH.name}. Render the "
        f"pages, correct what is wrong, then `apply`."
    )


def apply_assignments(coco):

    assignments = load_assignments()

    if not assignments:
        raise SystemExit(f"{ASSIGN_PATH} is empty - nothing to apply")

    grouped = boxes_by_page(coco)

    image_ids = {img["file_name"]: img["id"] for img in coco["images"]}

    annotations = []

    skipped = []

    for name, spec in assignments.items():

        if spec == "skip":
            skipped.append(name)
            continue

        if name not in grouped:
            raise SystemExit(f"{name} has no boxes in {PREANN_PATH.name}")

        boxes = dict(grouped[name])

        claimed = set()

        # explicit assignments first, so merges consume their members
        # before the leftovers are defaulted
        for key, label in spec.items():

            # A list means the SAME box carries more than one label.
            # That is how the guide's overlapping `crossed_out` layer
            # is expressed: ["paragraph", "crossed_out"] keeps the
            # prose region and marks it cancelled on top, instead of
            # replacing the prose and losing it.
            labels = label if isinstance(label, list) else [label]

            for item in labels:
                if item not in CLASSES and item != DROP:
                    raise SystemExit(
                        f"{name}: {item!r} is not in the schema {CLASSES} "
                        f"(or {DROP!r})"
                    )

            indices = [int(part) for part in str(key).split("+")]

            missing = [i for i in indices if i not in boxes]

            if missing:
                raise SystemExit(
                    f"{name}: box {missing} does not exist "
                    f"(page has 1..{len(boxes)})"
                )

            repeated = [i for i in indices if i in claimed]

            if repeated:
                raise SystemExit(f"{name}: box {repeated} assigned twice")

            claimed.update(indices)

            if DROP in labels:
                continue

            members = [boxes[i] for i in indices]

            x1 = min(b[0] for b in members)
            y1 = min(b[1] for b in members)
            x2 = max(b[2] for b in members)
            y2 = max(b[3] for b in members)

            for item in labels:
                annotations.append((name, item, [x1, y1, x2, y2]))

        for index, box in boxes.items():

            if index not in claimed:
                annotations.append((name, DEFAULT_LABEL, box))

    if skipped:
        print(f"Skipped {len(skipped)} page(s) with unusable geometry: "
              f"{skipped}")

    reviewed = {n for n in assignments if assignments[n] != "skip"}

    coco["images"] = [
        i for i in coco["images"] if i["file_name"] in reviewed
    ]

    coco["annotations"] = []

    for index, (name, label, (x1, y1, x2, y2)) in enumerate(
            annotations, start=1):

        coco["annotations"].append(
            {
                "id": index,
                "image_id": image_ids[name],
                "category_id": CLASSES.index(label) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "segmentation": [],
            }
        )

    coco["info"] = {
        "description": (
            "Layout ground truth. Boxes are machine-proposed geometry; "
            "every class was assigned by reviewing the rendered page."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w") as handle:
        json.dump(coco, handle, indent=1)

    kept = Counter()

    for ann in coco["annotations"]:
        kept[CLASSES[ann["category_id"] - 1]] += 1

    print(f"Pages reviewed : {len(coco['images'])}")
    print(f"Boxes          : {len(coco['annotations'])}")

    print("\nClass distribution:")
    total = sum(kept.values()) or 1
    for name in CLASSES:
        print(f"  {name:<12} {kept.get(name, 0):>4}  "
              f"{100 * kept.get(name, 0) / total:5.1f}%")

    print(f"\nWritten: {OUT_PATH}")


def status(coco):

    grouped = boxes_by_page(coco)

    assignments = load_assignments()

    done = len(assignments)

    print(f"Pages available : {len(grouped)}")
    print(f"Pages reviewed  : {done}")

    counts = Counter()

    for page in assignments.values():
        counts.update(page.values())

    if counts:
        print("\nAssigned so far (non-default only):")
        for name, count in counts.most_common():
            print(f"  {name:<12} {count:>4}")

    remaining = [n for n in sorted(grouped) if n not in assignments]

    if remaining:
        print(f"\nNext up: {remaining[:6]}")


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    sub = parser.add_subparsers(dest="command", required=True)

    render_cmd = sub.add_parser("render")
    render_cmd.add_argument("--pages")
    render_cmd.add_argument("--split")
    render_cmd.add_argument("--start", type=int, default=0)
    render_cmd.add_argument("--count", type=int, default=6)

    seed_cmd = sub.add_parser("seed")
    seed_cmd.add_argument("--weights", required=True)
    seed_cmd.add_argument("--count", type=int, default=12)
    seed_cmd.add_argument("--split")
    seed_cmd.add_argument("--imgsz", type=int, default=640)
    seed_cmd.add_argument("--conf", type=float, default=0.15)

    sub.add_parser("apply")
    sub.add_parser("status")

    args = parser.parse_args()

    coco = load_coco()

    if args.command == "render":
        render(args, coco)
    elif args.command == "seed":
        seed(args, coco)
    elif args.command == "apply":
        apply_assignments(coco)
    else:
        status(coco)


if __name__ == "__main__":
    main()
