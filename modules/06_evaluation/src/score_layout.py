"""
Score 02_segment's regions against the hand-drawn layout boxes.

    06_evaluation/output/annotations_prepared.json   (register.py)
    02_segment/input/                                (the pages)
        -> 06_evaluation/output/layout_score.json
        -> 06_evaluation/output/layout_fragments.csv

Runs the segmenter itself over exactly the annotated pages rather than
reading a whole-corpus `segmentation.json`, so a threshold can be changed
and re-scored in under a minute instead of after a full run.

WHAT THIS CAN AND CANNOT MEASURE
--------------------------------
The ground truth is **incomplete on purpose**: 409 boxes over 112 pages,
about 3.6 a page, while a page holds ~24 lines of writing. Annotators
boxed the regions the guide asks for, not every line. Look at any page in
`register.py --check` and most of the writing has no box on it.

That rules out precision. A segment region outside every annotated box is
not a false positive - it is almost always correct content nobody boxed.
Any "spurious region" count computed here would be measuring the
annotation effort, not the segmenter, and would look like a defect that
cannot be fixed.

What the annotation does support, exactly:

**FRAGMENTS PER REGION.** A human drew one box round one paragraph. How
many pieces does the segmenter cut that same area into? This is
well-defined under incomplete annotation, because it only ever looks
inside a box that exists. It is also precisely the complaint this module
was built to make falsifiable - that segmentation is sound geometrically
but shatters content into too many crops.

**INK RECALL.** Of the ink inside an annotated box, how much lands inside
some segment region? Catches the opposite failure - content dropped
before OCR ever sees it - and is likewise confined to boxed areas.

**SPILL.** How far a region assigned to a box extends beyond it, as a
fraction of the region's own area. Catches over-merging, the failure that
appears the moment you start fixing fragmentation. Reported honestly as a
weak signal: an unannotated neighbour is indistinguishable here from a
genuine over-merge, so spill is expected to be non-zero and only a large
move in it means anything.

A region is attributed to the box it overlaps most, and only if at least
ATTRIBUTION_FRACTION of the region sits inside it. Regions matching no box
are counted and otherwise ignored, for the reason above.

Run:
    python score_layout.py
    python score_layout.py --baseline output/layout_score.json   # compare
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import cv2


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES_DIR = STAGE_DIR.parent

sys.path.insert(0, str(MODULES_DIR / "02_segment" / "src"))
import segment  # noqa: E402

PAGES_DIR = MODULES_DIR / "02_segment" / "input"

OUT_DIR = STAGE_DIR / "output"
LABELS_PATH = OUT_DIR / "annotations_prepared.json"
SCORE_PATH = OUT_DIR / "layout_score.json"
FRAGMENT_PATH = OUT_DIR / "layout_fragments.csv"

# A segment region counts as belonging to an annotated box when this much
# of the region's own area is inside it. Well over half, so a region
# cannot be attributed to two boxes, and low enough that the slack a
# rotated box picks up in registration does not orphan a good region.
ATTRIBUTION_FRACTION = 0.6

INK_THRESHOLD = 180


def load_labels():

    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} not found - run register.py first")

    with open(LABELS_PATH) as handle:
        return json.load(handle)


def as_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def intersection(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def segment_page(path):
    """Every line region the segmenter emits for one page, flat."""

    record, _, _ = segment.segment_page(path)

    if record is None:
        return None, None

    regions = [line["bbox"]
               for block in record["blocks"]
               for line in block["lines"]]

    return regions, record


def score_page(path, boxes):
    """
    Returns per-annotated-box rows plus the page's unattributed count.
    """

    regions, record = segment_page(path)

    if regions is None:
        return [], 0, 0

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    ink = gray < INK_THRESHOLD

    # ink the segmenter itself works on - rules removed - so a box whose
    # only content is a printed rule is not scored as dropped content
    mask = segment.ink_mask(gray)
    clean, _, _, _ = segment.split_rules(mask)
    ink = clean > 0

    covered = np.zeros(ink.shape, bool)
    for x1, y1, x2, y2 in regions:
        covered[y1:y2, x1:x2] = True

    assigned = {}
    unattributed = 0

    for region in regions:

        best, best_share = None, 0.0

        for index, box in enumerate(boxes):
            share = intersection(region, box["xyxy"]) / max(area(region), 1.0)
            if share > best_share:
                best, best_share = index, share

        if best is None or best_share < ATTRIBUTION_FRACTION:
            unattributed += 1
            continue

        assigned.setdefault(best, []).append(region)

    rows = []

    for index, box in enumerate(boxes):

        x1, y1, x2, y2 = (int(round(v)) for v in box["xyxy"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ink.shape[1], x2), min(ink.shape[0], y2)

        window_ink = int(ink[y1:y2, x1:x2].sum())
        window_covered = int((ink[y1:y2, x1:x2] & covered[y1:y2, x1:x2]).sum())

        members = assigned.get(index, [])

        spill = 0.0
        if members:
            inside = sum(intersection(r, box["xyxy"]) for r in members)
            total = sum(area(r) for r in members)
            spill = 1.0 - inside / max(total, 1.0)

        rows.append({
            "category": box["category"],
            "fragments": len(members),
            "ink": window_ink,
            "ink_covered": window_covered,
            "ink_recall": round(window_covered / window_ink, 4)
                          if window_ink else None,
            "spill": round(spill, 4),
            "box_height_px": round(box["xyxy"][3] - box["xyxy"][1], 1),
        })

    return rows, unattributed, len(regions)


def summarise(rows):
    """Per class and overall, the numbers this module exists to report."""

    by_class = {}

    for row in rows:
        by_class.setdefault(row["category"], []).append(row)

    summary = {}

    for name, group in sorted(by_class.items()):

        fragments = np.array([r["fragments"] for r in group], float)
        recalls = [r["ink_recall"] for r in group if r["ink_recall"] is not None]
        spills = np.array([r["spill"] for r in group
                           if r["fragments"] > 0], float)

        summary[name] = {
            "boxes": len(group),
            "fragments_median": float(np.median(fragments)),
            "fragments_mean": round(float(fragments.mean()), 2),
            "fragments_max": int(fragments.max()),
            "boxes_with_no_region": int((fragments == 0).sum()),
            "ink_recall_median": round(float(np.median(recalls)), 4)
                                 if recalls else None,
            "ink_recall_min": round(float(np.min(recalls)), 4)
                              if recalls else None,
            "spill_median": round(float(np.median(spills)), 4)
                            if len(spills) else None,
        }

    return summary


def print_table(summary, baseline=None):

    print(f"\n{'class':<14}{'boxes':>7}{'frag med':>10}{'frag mean':>11}"
          f"{'frag max':>10}{'ink recall':>12}{'spill':>8}")
    print("-" * 72)

    for name, s in summary.items():

        recall = "-" if s["ink_recall_median"] is None else \
            f"{s['ink_recall_median'] * 100:.1f}%"
        spill = "-" if s["spill_median"] is None else \
            f"{s['spill_median'] * 100:.1f}%"

        line = (f"{name:<14}{s['boxes']:>7}{s['fragments_median']:>10.1f}"
                f"{s['fragments_mean']:>11.2f}{s['fragments_max']:>10}"
                f"{recall:>12}{spill:>8}")

        if baseline and name in baseline:
            delta = s["fragments_mean"] - baseline[name]["fragments_mean"]
            line += f"   ({delta:+.2f} vs baseline)"

        print(line)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        help="a previous layout_score.json to compare against")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=["train", "val", "test"],
                        help="score only one split of the manifest")
    parser.add_argument("--covers", action="store_true",
                        help="also score the identity cover sheets, which "
                             "02_segment deliberately never processes")
    args = parser.parse_args()

    coco = load_labels()

    categories = {c["id"]: c["name"] for c in coco["categories"]}

    by_image = {}
    for annotation in coco["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    images = coco["images"]

    # `segment.content_pages()` skips page_01 of every booklet - it is the
    # printed identity block, and no pipeline stage ever sees it. Scoring
    # it measures code that does not run: it is also where almost every
    # annotated `table` is, because the marks grid on the cover is a
    # table, so leaving it in reports a table score for a page the
    # segmenter is not asked about.
    covers = [i for i in images if i.get("page_kind") == "cover"]

    if not args.covers:
        images = [i for i in images if i.get("page_kind") != "cover"]

    if args.split:
        images = [i for i in images if i.get("split") == args.split]

    if args.limit:
        images = images[:args.limit]

    print(f"Scoring {len(images)} annotated page(s)"
          + (f" ({args.split} split)" if args.split else ""))

    if covers and not args.covers:
        print(f"  {len(covers)} cover sheet(s) excluded - 02_segment never "
              f"processes them (--covers to include)")

    all_rows = []
    unattributed = 0
    regions_total = 0
    scored_pages = 0

    for index, image in enumerate(images, start=1):

        path = PAGES_DIR / image["source"]

        if not path.exists():
            continue

        boxes = [{"xyxy": as_xyxy(a["bbox"]),
                  "category": categories[a["category_id"]]}
                 for a in by_image.get(image["id"], [])]

        if not boxes:
            continue

        rows, page_unattributed, page_regions = score_page(path, boxes)

        for row in rows:
            row["page"] = image["file_name"]

        all_rows.extend(rows)
        unattributed += page_unattributed
        regions_total += page_regions
        scored_pages += 1

        if index % 10 == 0:
            print(f"  {index}/{len(images)}", flush=True)

    if not all_rows:
        raise SystemExit("nothing scored")

    summary = summarise(all_rows)

    baseline = None
    if args.baseline and args.baseline.exists():
        with open(args.baseline) as handle:
            baseline = json.load(handle)["by_class"]

    print_table(summary, baseline)

    overall = np.array([r["fragments"] for r in all_rows], float)

    print(f"\nPages scored          : {scored_pages}")
    print(f"Annotated boxes       : {len(all_rows)}")
    print(f"Regions emitted       : {regions_total} "
          f"({regions_total / scored_pages:.1f} per page)")
    print(f"Fragments per box     : median {np.median(overall):.1f}, "
          f"mean {overall.mean():.2f}, max {int(overall.max())}")
    print(f"Unattributed regions  : {unattributed} "
          f"({unattributed / regions_total * 100:.1f}% - mostly content "
          f"nobody boxed, not errors)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SCORE_PATH, "w") as handle:
        json.dump({
            "pages": scored_pages,
            "boxes": len(all_rows),
            "regions_emitted": regions_total,
            "regions_per_page": round(regions_total / scored_pages, 2),
            "fragments_per_box_mean": round(float(overall.mean()), 3),
            "fragments_per_box_median": float(np.median(overall)),
            "unattributed_regions": unattributed,
            "by_class": summary,
        }, handle, indent=1)

    with open(FRAGMENT_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nScore     : {SCORE_PATH}")
    print(f"Per box   : {FRAGMENT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
