"""
Produce Module 2 (Region Router) output to feed 03_ocr.

    02_segment/output/segmentation.json  (the geometry)
    02_segment/crops/manifest.csv        (the line images)
        -> 03_ocr/output/routed_regions.json

WHY THIS EXISTS
---------------
03_ocr does not read the corpus directly - it reads whatever Module 2
routed to the `ocr` processor. But Module 2 has never been run against
this corpus: its input/segmentation_output.json is a four-region
hand-written sample, and the classifier that would label real regions
does not exist yet. Without something in between, the recogniser could
not be built or tested at all.

This fills that gap, so recognise.py can be written against the real
router contract today and keep working unchanged when the real router
is wired up.

ONLY THE LABELS ARE INVENTED
----------------------------
This is deliberately not a reimplementation of the router. It builds
the router's *input* and then imports and runs the actual RegionRouter
from modules/module2_router - the real validate(), the real
sort_regions(), the real route() against the real ROUTING_TABLE. So
the ordering, the discard threshold and the label -> processor mapping
are not approximations of Module 2's behaviour, they ARE Module 2's
behaviour, and they change when it changes.

What is synthetic is one field: `label` (and the `confidence` attached
to it). Segmentation is geometry-only - it finds where the ink is, not
what it means - so a region's class has to come from somewhere, and
until the classifier exists that somewhere is the seeded heuristics
below. Everything else - bboxes, crop paths, page and line identity -
is real data measured off real pages.

The practical consequence: a region typed `equation` here is not
really an equation, it is a line whose shape made the heuristic guess
that. Do not measure anything about classification against this file.
What it is good for is exercising the *plumbing* - that the recogniser
correctly takes only what was routed to it, and correctly leaves
math_ocr's and diagram_parser's regions alone.

Deterministic: every label is seeded from the region's own line_uid,
so reruns are identical and two machines agree.

Run:
    python simulate_router.py                    # booklet s01_c1
    python simulate_router.py --student 5 --cie 2
    python simulate_router.py --pages 2          # first 2 pages only
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES_DIR = STAGE_DIR.parent

SEGMENT_DIR = MODULES_DIR / "02_segment"
MANIFEST_PATH = SEGMENT_DIR / "crops" / "manifest.csv"
SEGMENTATION_PATH = SEGMENT_DIR / "output" / "segmentation.json"

# Module 2 is being reorganised on another branch (modules/03_router/),
# so look for it by shape rather than by one hardcoded name: whichever
# candidate actually contains router/router.py is the real thing. If
# the rewrite changes the RegionRouter API rather than just its
# location, this will not paper over it - run_real_router() raises and
# says exactly what it looked for.
ROUTER_CANDIDATES = ("module2_router", "03_router", "02_router")

OUT_DIR = STAGE_DIR / "output"
ROUTED_PATH = OUT_DIR / "routed_regions.json"

# The router's own input format, kept next to the output so a reader
# can see exactly what was handed to it.
ROUTER_INPUT_PATH = OUT_DIR / "router_input.json"


# Label mix. These are rates, not truths - see the module docstring.
# Chosen so every downstream branch is exercised by a single booklet:
# something for math_ocr, something for diagram_parser, something the
# validator throws away, something that lands in manual_review.
EQUATION_RATE = 0.06
CROSSED_OUT_RATE = 0.015
UNKNOWN_RATE = 0.015

# A few regions come in under module2_router's DISCARD_THRESHOLD so
# that the real validator actually drops something and the reconciling
# counts below have to account for it.
LOW_CONFIDENCE_RATE = 0.02

# Geometry thresholds, as fractions of page width. A question number
# ("2a)") is a short mark at the left margin; a heading is short but
# need not be at the margin.
QUESTION_MAX_WIDTH = 0.10
QUESTION_MAX_X = 0.20
HEADING_MAX_WIDTH = 0.35


def _rng(line_uid):
    """Seeded by identity, so a region's label never depends on order."""

    digest = hashlib.sha256(("router:" + line_uid).encode()).digest()

    return random.Random(int.from_bytes(digest[:8], "big"))


def line_uid(row):
    """The corpus-wide key, derived exactly as 03_ocr/src/simulate.py does."""

    return (f"{row['page_id']}_b{int(row['block_id']):02d}"
            f"_l{int(row['line_id']):02d}")


def choose_label(row, page_width, first_in_block):
    """
    Guess a region class from its shape.

    SYNTHETIC. Segmentation is geometry-only, so there is nothing real
    to read here - this is a stand-in for the classifier that will
    eventually do it.
    """

    rng = _rng(line_uid(row))

    # `tall` is the one real signal available: 02_segment flags regions
    # that are too tall to be a row of writing - figures, braces, long
    # division. Those are genuinely not text.
    if int(row["tall"]):
        return "diagram", rng

    width = int(row["width"])
    x1 = int(row["x1"])

    draw = rng.random()

    if draw < CROSSED_OUT_RATE:
        return "crossed_out", rng

    if draw < CROSSED_OUT_RATE + UNKNOWN_RATE:
        return "unknown", rng

    if draw < CROSSED_OUT_RATE + UNKNOWN_RATE + EQUATION_RATE:
        return "equation", rng

    if width < QUESTION_MAX_WIDTH * page_width and x1 < QUESTION_MAX_X * page_width:
        return "question", rng

    if first_in_block and width < HEADING_MAX_WIDTH * page_width:
        return "heading", rng

    return "paragraph", rng


def choose_confidence(rng, label):
    """
    0.0-1.0, shaped like a detector's output rather than a flat 0.99,
    with a deliberate tail below module2_router's DISCARD_THRESHOLD.
    """

    if rng.random() < LOW_CONFIDENCE_RATE:
        return round(rng.uniform(0.05, 0.29), 3)

    base = rng.betavariate(9, 2)

    if label in ("unknown", "crossed_out"):
        base *= rng.uniform(0.55, 0.85)

    return round(min(0.999, max(0.05, base)), 3)


def build_router_input(rows, page_sizes):
    """Manifest rows -> the {"regions": [...]} shape RegionRouter.load reads."""

    # Group by page so "first line of a block" can be decided, and so
    # regions are emitted in a stable order.
    rows = sorted(rows, key=lambda r: (int(r["page"]),
                                       int(r["block_id"]),
                                       int(r["line_id"])))

    regions = []
    seen_blocks = set()

    for index, row in enumerate(rows, start=1):

        page_id = row["page_id"]
        page_width = page_sizes.get(page_id, [1598, 2177])[0]

        block_key = (page_id, row["block_id"])
        first_in_block = block_key not in seen_blocks
        seen_blocks.add(block_key)

        label, rng = choose_label(row, page_width, first_in_block)

        regions.append({
            "id": index,
            "page": int(row["page"]),
            "label": label,
            "confidence": choose_confidence(rng, label),
            "bbox": [int(row["x1"]), int(row["y1"]),
                     int(row["x2"]), int(row["y2"])],
            # Relative to 02_segment/crops/. recognise.py resolves it
            # and also parses page/block/line identity back out of it,
            # because the router's Region carries no metadata through.
            "crop_path": row["crop"],
        })

    return {"regions": regions}


def find_router_dir():
    """The Module 2 directory, identified by containing router/router.py."""

    for name in ROUTER_CANDIDATES:

        candidate = MODULES_DIR / name

        if (candidate / "router" / "router.py").exists():
            return candidate

    raise SystemExit(
        "could not find Module 2. Looked for router/router.py under:\n  "
        + "\n  ".join(str(MODULES_DIR / n) for n in ROUTER_CANDIDATES)
        + "\nIf the router was rewritten rather than moved, point "
          "ROUTER_CANDIDATES at it and check RegionRouter still has "
          "load/validate/sort_regions/route/save."
    )


def run_real_router(input_path, output_path):
    """
    Import Module 2 and run it, rather than reimplementing it.

    The router uses flat imports (`from config import ...`), so its
    directory has to lead sys.path. Importing its config also creates
    that module's outputs/ and outputs/debug/ as a side effect - empty
    directories, which git does not track. The routed file itself is
    written to OUR path, so Module 2's own committed
    outputs/routed_regions.json is never touched.
    """

    router_dir = find_router_dir()

    sys.path.insert(0, str(router_dir))

    try:
        from router.router import RegionRouter
    except ImportError as exc:
        raise SystemExit(
            f"found {router_dir} but could not import RegionRouter: {exc}\n"
            f"(it needs `rich` - see that module's requirements.txt)"
        )

    router = RegionRouter()

    router.load(input_path)
    router.validate()
    router.sort_regions()
    router.route()
    router.save(output_path)

    sys.path.remove(str(router_dir))


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--student", type=int, default=1)
    parser.add_argument("--cie", type=int, default=1)
    parser.add_argument("--pages", type=int, help="first N pages only")

    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"{MANIFEST_PATH} not found - run 02_segment/src/crop_lines.py"
        )

    with open(MANIFEST_PATH) as handle:
        rows = list(csv.DictReader(handle))

    # Only regions that became a crop. The rest were too small to hold
    # a glyph, so there is no image for anything downstream to read.
    booklet = [
        r for r in rows
        if r["crop"]
        and int(r["student"]) == args.student
        and int(r["cie"]) == args.cie
    ]

    if not booklet:
        raise SystemExit(
            f"no cropped regions for student_{args.student:02d} "
            f"cie_{args.cie}"
        )

    if args.pages:
        keep = sorted({int(r["page"]) for r in booklet})[:args.pages]
        booklet = [r for r in booklet if int(r["page"]) in keep]

    segmentation = json.load(open(SEGMENTATION_PATH))
    page_sizes = {p["page_id"]: p["size"] for p in segmentation}

    payload = build_router_input(booklet, page_sizes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ROUTER_INPUT_PATH, "w") as handle:
        json.dump(payload, handle, indent=1)

    print(f"student_{args.student:02d} cie_{args.cie}: "
          f"{len(payload['regions'])} region(s) over "
          f"{len({r['page'] for r in payload['regions']})} page(s)")
    print(f"Router input : {ROUTER_INPUT_PATH}\n")

    run_real_router(ROUTER_INPUT_PATH, ROUTED_PATH)

    # Reconcile: what went in, what the validator kept, where it went.
    routed = json.load(open(ROUTED_PATH))

    by_processor = {}
    kept = 0

    for page in routed["pages"]:
        for region in page["regions"]:
            kept += 1
            by_processor[region["processor"]] = (
                by_processor.get(region["processor"], 0) + 1
            )

    dropped = len(payload["regions"]) - kept

    print()
    print(f"Regions in    : {len(payload['regions'])}")
    print(f"Kept          : {kept}")
    print(f"Dropped       : {dropped} "
          f"(confidence below module2_router DISCARD_THRESHOLD)")
    print("Routed to     : " + ", ".join(
        f"{name}={count}" for name, count in sorted(by_processor.items())))
    print()
    print(f"Routed output : {ROUTED_PATH}")
    print()
    print("SIMULATED ROUTING - region labels were guessed from geometry, "
          "not classified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
