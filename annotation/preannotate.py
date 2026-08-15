"""
Produce candidate boxes for the annotation sample, as COCO, so the
labeller starts by correcting boxes instead of drawing them.

WHY THE BOXES ARE NOT PRE-CLASSIFIED (by default)
-------------------------------------------------
scan_doc_v2 does two separable things, and they are not equally good:

  * finding blocks   - projection profiling, reliable geometry
  * naming them      - hand-tuned thresholds, and the README is candid
                       that math-vs-paragraph is the weak link

A wrong label is worse than no label. Annotators anchor on whatever is
already on screen, so a confident-looking mislabel gets accepted more
often than a blank one gets missed, and the error ends up in the
ground truth the model is scored against.

So every box is emitted as `paragraph` - the schema's declared default
for ties - and the annotator reclassifies. Pass `--classify` to use the
predicted label instead, which is only worth doing to *evaluate* the
classifier, never to build the ground truth it will be judged on.

Geometry is likewise a starting point: blocks are runs of ruled text
lines, so a figure or table usually arrives split across several boxes
that need merging.

Run:
    python preannotate.py
    python preannotate.py --classify        # keep predicted labels
    python preannotate.py --limit 5 --debug # overlays to eyeball first
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent

MODULE_DIR = BASE_DIR.parent / "modules" / "scan_doc_v2"

sys.path.insert(0, str(MODULE_DIR))

from normalize_page import (                                 # noqa: E402
    analyze_page,
    ink_mask,
    scaled,
    split_horizontal_strokes,
)
from segment_blocks import segment_page                      # noqa: E402
from features import page_masks, block_features              # noqa: E402
from classify_blocks import classify                         # noqa: E402


IMAGE_DIR = BASE_DIR / "images"
MANIFEST_PATH = BASE_DIR / "manifest.csv"

CORPUS_DIR = BASE_DIR.parent / "preprocessing" / "output"

OUTPUT_DIR = BASE_DIR / "preannotations"

# Order fixes the category ids in the COCO file. Keep it stable: CVAT
# resolves labels by id on import, so reordering silently relabels
# everything already annotated.
CLASSES = ["paragraph", "math", "figure", "table", "code", "crossed_out"]

DEFAULT_LABEL = "paragraph"

# Segmentation emits a few degenerate slivers: a fragment of printed
# rule that survived the split (full width, ~10px tall) or the margin
# line itself (a few px wide, most of the page tall). They are not
# regions, and every one is a box the annotator has to notice and
# delete, so they are dropped here.
#
# The height floor is a PHYSICAL length (declared at
# normalize_page.REFERENCE_WIDTH), not a multiple of the measured
# pitch. That distinction matters:
#
# pitch is unreliable in exactly the direction that breaks this. On
# pages where the student writes on alternate rules, autocorrelation
# locks onto the 2x harmonic and reports ~116 instead of the true ~58.
# A 0.35*pitch floor then computes 41px while the real text bands are
# 21-28px tall - so the filter deleted every block on the page and it
# came out empty. Measured on s05_c1_p07, s08_c1_p08, s35_c3_p02.
#
# 23px at reference (~16px at 200 DPI) sits cleanly between the
# 10-13px rule fragments being removed and the 20px+ text bands being
# kept, and does not move when pitch is misread.
MIN_BLOCK_HEIGHT = 23
MIN_BLOCK_WIDTH_FRACTION = 0.02

# Colours for the debug overlay only (BGR).
COLOURS = {
    "paragraph": (0, 170, 0),
    "math": (255, 120, 0),
    "figure": (0, 0, 220),
    "table": (200, 0, 200),
    "code": (0, 190, 190),
    "crossed_out": (40, 40, 40),
}


def load_manifest():

    with open(MANIFEST_PATH) as handle:
        return list(csv.DictReader(handle))


def corpus_rows():
    """
    Every page in the converted corpus, as manifest-shaped rows.

    `file_name` is flattened to s<NN>_c<M>_p<KK>.png so it matches the
    sample's naming - the two sets then share an identifier space and a
    page annotated in the sample can be recognised here.
    """

    import re

    rows = []

    for page in sorted(CORPUS_DIR.glob("student_*/cie_*/page_*.png")):

        student = int(re.search(r"student_(\d+)", page.parts[-3]).group(1))
        cie = int(re.search(r"cie_(\d+)", page.parts[-2]).group(1))
        number = int(re.search(r"(\d+)", page.stem).group(1))

        rows.append(
            {
                "file_name": f"s{student:02d}_c{cie}_p{number:02d}.png",
                "path": str(page),
                "student_id": f"student_{student:02d}",
                "cie": cie,
                "page": number,
                "page_kind": "cover" if number == 1 else "content",
                "split": "",
            }
        )

    return rows


def _worker(job):
    """One page, in a subprocess. Returns (row, boxes, structure) or an error."""

    row, use_classifier = job

    path = Path(row.get("path") or (IMAGE_DIR / row["file_name"]))

    try:
        boxes, structure, _ = page_boxes(path, use_classifier, want_image=False)
    except Exception as exc:
        return row, None, None, f"{type(exc).__name__}: {exc}"

    return row, boxes, structure, None


def ink_extent(image_bgr, structure):
    """
    Bounding box of the page's handwriting, ignoring printed furniture.

    Used only as a last resort, when block segmentation returned
    nothing at all. The printed ruling is removed first - otherwise
    this returns the whole page every time, since the rules span it.

    Returns [x1, y1, x2, y2] or None when the page really is blank.
    """

    import numpy as np

    clean, _, _ = split_horizontal_strokes(ink_mask(image_bgr))

    height, width = clean.shape

    # ignore the scan edge, where binding shadow reads as ink
    edge = int(0.03 * width)

    if edge:
        clean[:, :edge] = 0
        clean[:, -edge:] = 0
        clean[:edge, :] = 0
        clean[-edge:, :] = 0

    # drop the margin rule, which is vertical and would drag x1 left
    margin_x = structure.get("margin_x")

    if margin_x:
        clean[:, max(0, margin_x - 4):margin_x + 4] = 0

    rows = np.where(clean.any(axis=1))[0]
    cols = np.where(clean.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return None

    return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]


def page_boxes(image_path, use_classifier, want_image=True):
    """
    (boxes, structure, image) for one page.

    Each box is (label, x1, y1, x2, y2). `want_image=False` drops the
    decoded page from the return value, which matters when thousands of
    results are being collected across processes.
    """

    structure, image, _ = analyze_page(image_path)

    _, blocks, _ = segment_page(image_path, structure)

    pitch = structure.get("pitch") or max(40, structure["height"] // 40)

    min_height = scaled(MIN_BLOCK_HEIGHT, structure["width"])
    min_width = MIN_BLOCK_WIDTH_FRACTION * structure["width"]

    blocks = [
        block for block in blocks
        if (block["bbox"][3] - block["bbox"][1]) >= min_height
        and (block["bbox"][2] - block["bbox"][0]) >= min_width
    ]

    # Line-band segmentation finds nothing on a page that is one big
    # drawing: the diagram's long horizontals are stripped as printed
    # ruling, so no ink rows survive to profile. Those pages came out
    # with zero boxes, which is the one outcome that gives the
    # annotator nothing to adjust.
    #
    # Falling back to the extent of the ink is close to right there -
    # if the page is a single figure, its bounding box IS the region.
    if not blocks:

        found = ink_extent(image, structure)

        if found is not None:
            blocks = [{"bbox": found, "line_count": 1}]

    if not use_classifier:
        return (
            [(DEFAULT_LABEL, *block["bbox"]) for block in blocks],
            structure,
            image if want_image else None,
        )

    masks = page_masks(image, structure["margin_x"])

    # NO crossed_out HERE - MEASURED, NOT ASSUMED.
    #
    # The per-line strikethrough test from classify_blocks was wired in
    # here and then removed. Audited against 18 randomly sampled
    # detections on real corpus pages: ZERO were genuine
    # strikethroughs. Every one was a diagram arrow, a box edge, or
    # plain prose.
    #
    # The cause is structural, not a threshold that needs nudging. The
    # signal is "a short dense horizontal pen stroke", and the edges
    # and arrows of hand-drawn diagrams are exactly that. It scored
    # well on the old 300 DPI test set only because that set contained
    # almost no diagrams; this corpus is full of them.
    #
    # `code` is likewise absent - the classical classifier has no such
    # class to emit.

    boxes = []

    for block in blocks:

        features = block_features(
            masks,
            block["bbox"],
            pitch,
            structure["width"],
            structure["margin_x"],
        )

        label, _ = classify(features, structure["width"])

        boxes.append((label, *block["bbox"]))

    return boxes, structure, image if want_image else None


def debug_overlay(image, boxes, output_path):

    vis = image.copy()

    for label, x1, y1, x2, y2 in boxes:

        colour = COLOURS.get(label, (0, 0, 0))

        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(
            vis, label, (x1 + 4, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2
        )

    cv2.imwrite(str(output_path), vis)


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="every page in the corpus, not just the annotation sample",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--out", help="output json filename")

    args = parser.parse_args()

    rows = corpus_rows() if args.all else load_manifest()

    if args.limit:
        rows = rows[:args.limit]

    if not rows:
        sys.exit("no pages found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    debug_dir = OUTPUT_DIR / "debug"

    if args.debug:
        debug_dir.mkdir(exist_ok=True)

    images = []
    annotations = []

    counts = {name: 0 for name in CLASSES}

    unruled = []
    failed = []

    print(f"Pages to process: {len(rows)}  (workers: {args.workers})")

    jobs = [(row, args.classify) for row in rows]

    # Debug overlays need the decoded page, which is exactly what we
    # avoid shipping between processes - so that path stays serial.
    if args.debug or args.workers <= 1:
        results = (_worker(job) for job in jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(_worker, jobs, chunksize=8)

    image_id = 0

    for done, (row, boxes, structure, error) in enumerate(results, start=1):

        if done % 100 == 0:
            print(f"  {done}/{len(rows)}", flush=True)

        if error is not None:
            failed.append((row["file_name"], error))
            continue

        image_id += 1

        if not structure["is_ruled_page"]:
            unruled.append(row["file_name"])

        images.append(
            {
                "id": image_id,
                "file_name": row["file_name"],
                "width": structure["width"],
                "height": structure["height"],
            }
        )

        for label, x1, y1, x2, y2 in boxes:

            counts[label] = counts.get(label, 0) + 1

            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": CLASSES.index(label) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )

        if args.debug:
            page = Path(row.get("path") or (IMAGE_DIR / row["file_name"]))
            debug_overlay(
                cv2.imread(str(page)), boxes, debug_dir / row["file_name"]
            )

    coco = {
        "info": {
            "description": "Pre-annotations for handwritten answer "
                           "script layout. MACHINE-GENERATED STARTING "
                           "POINT, not ground truth.",
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

    default_name = (
        "preannotations_full_coco.json" if args.all
        else "preannotations_coco.json"
    )

    out_path = OUTPUT_DIR / (args.out or default_name)

    with open(out_path, "w") as handle:
        json.dump(coco, handle, indent=1)

    # CVAT wants the label set up front; writing it out avoids typing
    # six labels by hand and getting one of them subtly wrong
    labels = [
        {"name": name, "color": "", "attributes": []} for name in CLASSES
    ]

    with open(OUTPUT_DIR / "cvat_labels.json", "w") as handle:
        json.dump(labels, handle, indent=1)

    print(f"Pages       : {len(images)}")
    print(f"Boxes       : {len(annotations)}  "
          f"({len(annotations) / max(1, len(images)):.1f} per page)")

    print(f"Labels      : {'predicted' if args.classify else 'all ' + DEFAULT_LABEL}")

    if args.classify:
        print("\nPredicted distribution:")
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            if count:
                print(f"  {name:<12} {count:>4}")

    if unruled:
        print(f"\n{len(unruled)} page(s) with no dependable ruling "
              f"(cover sheets / near-empty) - boxes there are weaker")
        for name in unruled[:10]:
            print(f"  {name}")
        if len(unruled) > 10:
            print(f"  ... and {len(unruled) - 10} more")

    empty = len(images) - len({a['image_id'] for a in annotations})

    if empty:
        print(f"\n{empty} page(s) produced no boxes at all "
              f"(blank, or all-diagram - see README)")

    if failed:
        print(f"\n{len(failed)} page(s) FAILED:")
        for name, error in failed[:10]:
            print(f"  {name}: {error}")

    print(f"\nCOCO        : {out_path}")
    print(f"CVAT labels : {OUTPUT_DIR / 'cvat_labels.json'}")

    if args.debug:
        print(f"Overlays    : {debug_dir}")


if __name__ == "__main__":
    main()
