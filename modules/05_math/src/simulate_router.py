"""
Stand in for the router, so 05_math can be built and run today.

    02_segment/crops/  ->  05_math/input/routed/<student>/<cie>/routed_regions.json

WHY A SIMULATION AND NOT THE REAL ROUTER
----------------------------------------
The pipeline this module belongs to is

    02_segment  ->  router  ->  { ocr, math_ocr, diagram_parser, ... }

and the router already exists, at modules/module2_router. What it
cannot yet be fed is this corpus. The router routes on a LABEL -
its ROUTING_TABLE sends "equation" to math_ocr and "paragraph" to
ocr - and 02_segment deliberately emits no labels at all: it reports
where content is, never what it is, because there was no training
data to justify the claim.

So the real hand-off cannot happen until something labels regions.
Rather than block on that, this script produces exactly what the
router would produce, in exactly the router's schema, and 05_math
consumes that. When the classifier lands, this file is deleted and
the router's own output is pointed at 05_math unchanged.

WHAT IS REAL AND WHAT IS INVENTED
---------------------------------
Real: the images. Every crop referenced here is a genuine line
region cut by 02_segment/crop_lines.py from a genuine page, at its
true page coordinates.

Invented: the label, and therefore the routing. Nothing in this
corpus says which lines are mathematics, so this script decides by
looking at the ink, and it is a stand-in for a trained classifier,
not a substitute for one. Its confidence values are synthesised from
how many signals fire - they are NOT a model's posterior, and the
"simulated" flag in each region's metadata says so, so that no
measurement downstream can quietly credit them as real.

HOW A LINE IS GUESSED TO BE MATHS
---------------------------------
Two signals, chosen because they survive bad handwriting:

1. AN EQUALS SIGN - two short flat strokes, stacked, overlapping in
   x. It is the single most reliable marker of a worked line, and
   nothing in prose looks like it. The printed ruling would, which
   is why the rules are suppressed before this runs.

2. A SUPERSCRIPT - a small component riding high above the band of
   its neighbours. This is what catches "2^13" and hex/binary
   workings, which is most of what passes for mathematics in this
   corpus - these are computer-networks scripts, so the maths is
   arithmetic, powers and base conversion rather than calculus.

Both are structural. Neither needs to read a single character, which
is the point: reading the characters is 05_math's job, and a router
that could already do it would not need 05_math.

Run:
    python simulate_router.py                 # 6 booklets
    python simulate_router.py --booklets 40
    python simulate_router.py --all
    python simulate_router.py --review        # contact sheet of what was routed
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import prepare


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

MODULES_DIR = STAGE_DIR.parent
SEGMENT_DIR = MODULES_DIR / "02_segment"
CROP_DIR = SEGMENT_DIR / "crops"

INPUT_DIR = STAGE_DIR / "input"
ROUTED_DIR = INPUT_DIR / "routed"
PREVIEW_DIR = STAGE_DIR / "preview"

# The two rows of modules/module2_router/config.py:ROUTING_TABLE that
# this simulation can produce. Deliberately duplicated rather than
# imported: 05_math must not break when the router is refactored, and
# the real hand-off replaces this file entirely anyway.
ROUTING_TABLE = {
    "equation": "math_ocr",
    "paragraph": "ocr",
    "diagram": "diagram_parser",
}

# A flat stroke: at least this many times wider than tall, and no
# taller than this share of an x-height. Both bars of an "=" pass;
# a stroke of handwriting does not.
BAR_ASPECT = 1.8
BAR_HEIGHT_XHEIGHT = 0.35

# An "=" is also SHORT - about as wide as a digit. Without this
# bound the pair (a student's underline, a surviving fragment of
# printed rule) reads as an equals sign, which routed underlined
# headings and every underlined line of prose to math_ocr.
BAR_MAX_WIDTH_XHEIGHT = 1.80

# The two bars of an "=" are drawn as a pair, so they are close to
# the same length. An underline paired with anything is not.
BAR_LENGTH_RATIO = 2.20

# Two bars are an "=" if they overlap this much of the shorter one in
# x and sit within this many x-heights of each other vertically.
BAR_OVERLAP = 0.45
BAR_SEPARATION_XHEIGHT = 0.90

# A superscript is small and rides high in the line's ink band,
# stands clear of any stem below it, and has a full-height glyph -
# its base - close by on its left.
SUPER_HEIGHT_XHEIGHT = 0.65
SUPER_TOP_BAND = 0.30
SUPER_BASE_XHEIGHT = 0.70
SUPER_STEM_OVERLAP = 0.35
SUPER_BASE_GAP_XHEIGHT = 0.60

# How many superscripts a line needs before that alone routes it.
# One is not enough: a single high mark standing clear of a stem is
# an ordinary accident of handwriting, and taking it as a power
# routed prose like "client - server architecture" to math_ocr - 140
# of 377 routed regions came in on a lone superscript. Two on a
# short line is a working.
SUPER_MIN = 2

# A flat stroke crossing this share of the region is a drawn line,
# not a glyph. Two of them is a diagram.
DIAGRAM_BAR_WIDTH = 0.45
DIAGRAM_BARS = 2

# Long lines are prose. An exponent found in the middle of a
# full-width line of writing is far more likely to be a tall letter
# than a power, so the superscript signal is only trusted on a line
# short enough to be a working.
WORKING_WIDTH_PAGE = 0.55

# Confidences the simulation attaches. Fabricated, and flagged as
# such in metadata - see the module docstring.
CONFIDENCE_BOTH = 0.93
CONFIDENCE_ONE = 0.71
CONFIDENCE_PROSE = 0.90


def components(mask):
    """Glyph bounding boxes as (x, y, w, h), left to right."""

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    boxes = [
        (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        for label in range(1, count)
    ]

    return sorted(boxes)


def has_equals(boxes, height):
    """Two short flat strokes, stacked and aligned - an "=" sign."""

    bars = [
        box for box in boxes
        if box[3] > 0
        and box[2] >= BAR_ASPECT * box[3]
        and box[3] <= BAR_HEIGHT_XHEIGHT * height
        and box[2] <= BAR_MAX_WIDTH_XHEIGHT * height
    ]

    for i, (x1, y1, w1, h1) in enumerate(bars):
        for x2, y2, w2, h2 in bars[i + 1:]:

            overlap = min(x1 + w1, x2 + w2) - max(x1, x2)

            if overlap < BAR_OVERLAP * min(w1, w2):
                continue

            if max(w1, w2) > BAR_LENGTH_RATIO * min(w1, w2):
                continue

            gap = abs(y2 - y1)

            if 0 < gap <= BAR_SEPARATION_XHEIGHT * height:
                return True

    return False


def count_superscripts(boxes, height):
    """
    Small components riding high in their own column - exponents.

    "In their own column" is the whole test, and it is what separates
    an exponent from the dot on an "i". Both are small and both ride
    high; the difference is what sits underneath. A dot has its stem
    directly below it, an exponent has clear paper - it stands to the
    RIGHT of the base it belongs to, not on top of it.

    Without that check this fired on nearly every line of prose in
    the corpus - i-dots, the marks over "occurence", and the tittles
    of a cursive hand all counted - and routed 63% of a booklet to
    math_ocr.
    """

    if len(boxes) < 2:
        return 0

    tops = [box[1] for box in boxes]
    bottoms = [box[1] + box[3] for box in boxes]

    band_top, band_bottom = min(tops), max(bottoms)
    band = band_bottom - band_top

    if band <= 0:
        return 0

    tall = [box for box in boxes if box[3] >= SUPER_BASE_XHEIGHT * height]

    found = 0

    for x, y, w, h in boxes:

        small = h <= SUPER_HEIGHT_XHEIGHT * height and h < band * 0.55
        high = (y - band_top) <= SUPER_TOP_BAND * band

        if not (small and high):
            continue

        # A stem underneath makes it a diacritic, not a power.
        over_stem = any(
            min(x + w, bx + bw) - max(x, bx) >= SUPER_STEM_OVERLAP * w
            for bx, _, bw, _ in tall
        )

        if over_stem:
            continue

        # A power needs a base to its left, close by.
        based = any(
            bx + bw <= x + w and x - (bx + bw) <= SUPER_BASE_GAP_XHEIGHT * height
            for bx, _, bw, _ in tall
        )

        if based:
            found += 1

    return found


def count_long_bars(boxes, height, width):
    """
    Flat strokes running across the region - the lines of a drawing.

    The rules are already gone by the time this runs, so what is left
    spanning half a region is something the student drew: the frame
    of a table, the lifeline of a sequence diagram, the axis of a
    graph. Those belong to diagram_parser, and without this guard
    they reach math_ocr instead, because a pair of them stacked is
    indistinguishable from an equals sign.
    """

    return sum(
        1 for _, _, w, h in boxes
        if h <= BAR_HEIGHT_XHEIGHT * height and w >= DIAGRAM_BAR_WIDTH * width
    )


def classify(gray, pitch, page_width):
    """
    The label a trained classifier would be asked for, guessed.

    Returns (label, confidence, features). Everything here is
    measured on the ink AFTER rule suppression, because the printed
    ruling reads as a perfect equals bar and would route most of the
    corpus to math_ocr.
    """

    cleaned, _ = prepare.suppress_rules(gray, pitch)

    mask = prepare.glyph_mask(prepare.ink_mask(cleaned), pitch)

    features = {
        "equals": False,
        "superscripts": 0,
        "long_bars": 0,
        "glyphs": 0,
        "width_page_ratio": round(gray.shape[1] / max(1, page_width), 3),
    }

    if not mask.any():
        return "paragraph", CONFIDENCE_PROSE, features

    height = prepare.x_height(mask, pitch)

    boxes = components(mask)

    short = features["width_page_ratio"] <= WORKING_WIDTH_PAGE

    features["glyphs"] = len(boxes)
    features["equals"] = has_equals(boxes, height)
    features["superscripts"] = count_superscripts(boxes, height)
    features["long_bars"] = count_long_bars(boxes, height, gray.shape[1])

    # Checked before the maths signals, because a drawing sets both
    # of them off and a wrong processor is worse than a coarse label.
    if features["long_bars"] >= DIAGRAM_BARS:
        return "diagram", CONFIDENCE_ONE, features

    powered = short and features["superscripts"] >= SUPER_MIN

    if features["equals"] and powered:
        return "equation", CONFIDENCE_BOTH, features

    if features["equals"] or powered:
        return "equation", CONFIDENCE_ONE, features

    return "paragraph", CONFIDENCE_PROSE, features


def booklets(limit):
    """Crop manifest rows grouped into (student, cie) booklets."""

    path = CROP_DIR / "manifest.csv"

    if not path.exists():
        raise SystemExit(f"{path} not found - run 02_segment/crop_lines.py first")

    grouped = defaultdict(list)

    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row["crop"]:
                grouped[(int(row["student"]), int(row["cie"]))].append(row)

    keys = sorted(grouped)

    if limit:
        keys = keys[:limit]

    return [(key, grouped[key]) for key in keys]


def route_booklet(rows, pages):
    """
    Every line region of one booklet, labelled, ordered and routed.

    A real router emits ALL regions with the processor each is bound
    for, not just the interesting ones - so this does too, and
    math_ocr.py filters. That keeps the hand-off honest: the share of
    a booklet that lands on math_ocr is visible rather than assumed.
    """

    by_page = defaultdict(list)

    for row in rows:
        by_page[int(row["page"])].append(row)

    routed = []

    region_id = 0

    for page in sorted(by_page):

        regions = []

        for row in by_page[page]:

            gray = cv2.imread(str(CROP_DIR / row["crop"]), cv2.IMREAD_GRAYSCALE)

            if gray is None:
                continue

            meta = pages[row["page_id"]]

            # segment.py writes size as [width, height].
            label, confidence, features = classify(
                gray, meta["rule_pitch"], meta["size"][0],
            )

            region_id += 1

            regions.append({
                "id": region_id,
                "page": page,
                "label": label,
                "confidence": confidence,
                "bbox": {
                    "x1": int(row["x1"]), "y1": int(row["y1"]),
                    "x2": int(row["x2"]), "y2": int(row["y2"]),
                },
                "crop_path": row["crop"],
                "processor": ROUTING_TABLE[label],
                "reading_order": None,
                "ignored": False,
                "metadata": {
                    "simulated": True,
                    "page_id": row["page_id"],
                    "rule_pitch": meta["rule_pitch"],
                    "signals": features,
                },
            })

        # The router's own ordering rule, from router/ordering.py.
        regions.sort(key=lambda r: (r["bbox"]["y1"], r["bbox"]["x1"]))

        for order, region in enumerate(regions, start=1):
            region["reading_order"] = order

        routed.append({"page": page, "regions": regions})

    return routed


def _review(routed, path, limit=40):
    """A sheet of what got routed to math_ocr, to check by eye."""

    picked = [
        region
        for page in routed for region in page["regions"]
        if region["processor"] == "math_ocr"
    ][:limit]

    if not picked:
        return

    tiles = []
    width = 1100

    for region in picked:

        crop = cv2.imread(str(CROP_DIR / region["crop_path"]), cv2.IMREAD_GRAYSCALE)

        if crop is None:
            continue

        scale = min(1.0, (width - 20) / crop.shape[1], 70 / crop.shape[0])

        crop = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
        )

        signals = region["metadata"]["signals"]

        caption = (
            f"{region['metadata']['page_id']} r{region['id']}  "
            f"conf {region['confidence']}  "
            f"equals={signals['equals']} sup={signals['superscripts']}"
        )

        tiles.append(prepare._labelled(crop, width, caption))

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(tiles))

    print(f"review sheet -> {path}")


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--booklets", type=int, default=6)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--review", action="store_true")

    args = parser.parse_args()

    pages = {
        page["page_id"]: page
        for page in json.load(open(SEGMENT_DIR / "output" / "segmentation.json"))
    }

    selected = booklets(None if args.all else args.booklets)

    ROUTED_DIR.mkdir(parents=True, exist_ok=True)

    index = []

    total = {"regions": 0, "math_ocr": 0}

    for (student, cie), rows in selected:

        routed = route_booklet(rows, pages)

        regions = sum(len(page["regions"]) for page in routed)

        math = sum(
            1 for page in routed for region in page["regions"]
            if region["processor"] == "math_ocr"
        )

        target = ROUTED_DIR / f"student_{student:02d}" / f"cie_{cie}"
        target.mkdir(parents=True, exist_ok=True)

        with open(target / "routed_regions.json", "w") as handle:
            json.dump({"pages": routed}, handle, indent=4)

        index.append({
            "student": student,
            "cie": cie,
            "routed_regions": str(
                (target / "routed_regions.json").relative_to(INPUT_DIR)
            ),
            "regions": regions,
            "math_ocr": math,
        })

        total["regions"] += regions
        total["math_ocr"] += math

        print(
            f"student_{student:02d}/cie_{cie}: "
            f"{regions:5d} regions, {math:4d} -> math_ocr"
        )

        if args.review:
            _review(routed, PREVIEW_DIR / "routed" / f"s{student:02d}_c{cie}.png")

    with open(INPUT_DIR / "index.json", "w") as handle:
        json.dump({"booklets": index, "totals": total}, handle, indent=4)

    share = 100 * total["math_ocr"] / max(1, total["regions"])

    print(
        f"\n{total['regions']} regions, {total['math_ocr']} routed to "
        f"math_ocr ({share:.1f}%)\n-> {ROUTED_DIR}"
    )


if __name__ == "__main__":
    main()
