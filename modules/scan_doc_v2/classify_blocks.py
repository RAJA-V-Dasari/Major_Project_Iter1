"""
Step 3: assign each block one of

    paragraph | math | table | figure | crossed_out

Thresholds below were set by measuring real blocks on pages whose
content was checked by eye first (see README for the numbers). They are
ordered most-confident first, because the evidence quality differs a lot
between classes:

  table        crossings are decisive - 77 inside the cover sheet's
               table against 0-1 anywhere on a prose page
  crossed_out  strong, once strikethroughs are separated from
               underlines by what sits above and below the stroke
  figure       ink that is not organised into pitch-aligned rows
  math         WEAKEST. Rests mainly on component width: cursive prose
               connects into wide blobs (30-48 px measured) while maths
               is isolated symbols (24-29 px). The margin is thin and
               this is the first thing to distrust on new handwriting.
  paragraph    the default when nothing else fires

Run:
    python classify_blocks.py
    python classify_blocks.py --dump      # per-block feature table
"""

import argparse
import json
from pathlib import Path

import cv2

from normalize_page import INPUT_DIR, OUTPUT_DIR, page_sort_key, scaled
from features import page_masks, block_features


BLOCKS_PATH = OUTPUT_DIR / "blocks.json"
STRUCTURE_PATH = OUTPUT_DIR / "structure.json"
JSON_PATH = OUTPUT_DIR / "regions.json"

CLASSES = ["paragraph", "math", "table", "figure", "crossed_out"]


# --- table ---------------------------------------------------------
# Crossings of horizontal rules with vertical dividers. Measured: 77 in
# a real table, 0-1 in prose. 6 leaves wide headroom on both sides.
TABLE_MIN_CROSSINGS = 6

# --- crossed out ---------------------------------------------------
# Struck ink as a share of the block's ink. Measured 0.13-0.18 on truly
# struck blocks, 0.03-0.04 on merely underlined ones.
CROSSED_MIN_STRIKE = 0.08

# Crossed-out content is detected per LINE, not per block.
#
# A student cancels a line or a phrase, not usually a whole answer, so a
# strikethrough normally sits inside a block that is otherwise ordinary
# prose. Judging whole blocks either mislabels the surrounding good text
# as cancelled, or (with a tighter threshold) misses the cancellation
# because it is diluted by the rest of the block.
#
# So each line is tested on its own and emitted as its own region. A
# block whose lines are mostly struck becomes crossed_out as a whole.
BLOCK_CROSSED_LINE_SHARE = 0.60

# --- figure --------------------------------------------------------
# Tall regions whose rows do not repeat at the page pitch are drawings
# rather than writing. Only applied to blocks big enough for the
# periodicity measure to mean anything.
FIGURE_MIN_LINE_SPAN = 2.0
FIGURE_MAX_REGULARITY = 0.15

# --- math ----------------------------------------------------------
# Mean glyph-component width in px. Below this the block is isolated
# symbols rather than joined-up words.
# Measured at normalize_page.REFERENCE_WIDTH (2480px / 300 DPI) and
# scaled to whatever the page actually is. Left unscaled it silently
# means a different physical size on 200 DPI scans, where cursive blobs
# are ~1.46x narrower and would nearly all read as maths.
MATH_MAX_COMPONENT_WIDTH = 29.0

# Very sparse wide blocks are prose regardless of component width.
MATH_MAX_FILL = 0.25


def is_struck(features):
    """Whether a region's ink is predominantly cancelled."""

    strike = features["strike_ratio"]

    return strike >= CROSSED_MIN_STRIKE and strike >= features["underline_ratio"]


def classify(features, page_width):
    """
    Return (label, reason). The reason is kept so a wrong call can be
    traced to the measurement that caused it.

    `page_width` is needed because the component-width threshold is a
    physical length, so it must follow the scan resolution.

    Does not decide crossed_out - that is handled per line by the
    caller, for the reason documented at BLOCK_CROSSED_LINE_SHARE.
    """

    if features["crossings"] >= TABLE_MIN_CROSSINGS:
        return "table", f"crossings={features['crossings']}"

    if (
        features["line_span"] >= FIGURE_MIN_LINE_SPAN
        and features["row_regularity"] < FIGURE_MAX_REGULARITY
    ):
        return "figure", (
            f"span={features['line_span']} "
            f"regularity={features['row_regularity']}"
        )

    width = features["component_width"]

    limit = scaled(MATH_MAX_COMPONENT_WIDTH, page_width)

    if 0 < width <= limit and features["fill"] <= MATH_MAX_FILL:
        return "math", f"component_width={width}"

    return "paragraph", f"component_width={width}"


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dump",
        action="store_true",
        help="print the measured features for every block",
    )

    args = parser.parse_args()

    for path in (BLOCKS_PATH, STRUCTURE_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run normalize_page.py then "
                "segment_blocks.py first."
            )

    with open(STRUCTURE_PATH) as f:
        structures = json.load(f)

    with open(BLOCKS_PATH) as f:
        blocks_doc = json.load(f)

    document = {"pages": []}

    counts = {name: 0 for name in CLASSES}

    for page in blocks_doc["pages"]:

        image_name = page["image"]

        structure = structures[image_name]

        image = cv2.imread(str(INPUT_DIR / image_name))

        if image is None:
            print(f"  ! missing {image_name}")
            continue

        masks = page_masks(image, structure["margin_x"])

        pitch = structure["pitch"] or max(40, structure["height"] // 40)

        regions = []

        if args.dump:
            print(f"\n--- {image_name} ---")
            print(
                f"{'id':>3} {'label':>11} {'span':>5} {'cw':>6} "
                f"{'strike':>7} {'undr':>6} {'xing':>5} {'reg':>6} {'fill':>6}"
            )

        # Which individual lines are struck through. Done first, because
        # a block's own label depends on how many of its lines are.
        struck_lines = []

        for line in page["lines"]:

            line_features = block_features(
                masks,
                line["bbox"],
                pitch,
                structure["width"],
                structure["margin_x"],
            )

            if is_struck(line_features):
                struck_lines.append((line, line_features))

        next_id = max((b["id"] for b in page["blocks"]), default=0) + 1

        claimed = set()

        for block in page["blocks"]:

            features = block_features(
                masks,
                block["bbox"],
                pitch,
                structure["width"],
                structure["margin_x"],
            )

            bx1, by1, bx2, by2 = block["bbox"]

            inside = [
                item
                for item in struck_lines
                if by1 <= (item[0]["bbox"][1] + item[0]["bbox"][3]) / 2 <= by2
            ]

            share = len(inside) / max(1, block["line_count"])

            if inside and share >= BLOCK_CROSSED_LINE_SHARE:

                # nearly all of it is cancelled: the block itself is
                label, reason = "crossed_out", f"struck_lines={len(inside)}"

                claimed.update(id(item[0]) for item in inside)

            else:
                label, reason = classify(features, structure["width"])

            counts[label] += 1

            regions.append(
                {
                    "id": block["id"],
                    "label": label,
                    "bbox": block["bbox"],
                    "line_count": block["line_count"],
                    "reason": reason,
                    "features": features,
                }
            )

        # struck lines sitting inside an otherwise-normal block are
        # emitted on their own, so the good text around them keeps its
        # own label
        for line, line_features in struck_lines:

            if id(line) in claimed:
                continue

            counts["crossed_out"] += 1

            regions.append(
                {
                    "id": next_id,
                    "label": "crossed_out",
                    "bbox": line["bbox"],
                    "line_count": 1,
                    "reason": f"strike={line_features['strike_ratio']}",
                    "features": line_features,
                }
            )

            next_id += 1

            if args.dump:
                print(
                    f"{block['id']:>3} {label:>11} {features['line_span']:>5} "
                    f"{features['component_width']:>6} "
                    f"{features['strike_ratio']:>7} "
                    f"{features['underline_ratio']:>6} "
                    f"{features['crossings']:>5} "
                    f"{features['row_regularity']:>6} {features['fill']:>6}"
                )

        document["pages"].append(
            {
                "page": page["page"],
                "image": image_name,
                "regions": regions,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "w") as f:
        json.dump(document, f, indent=2)

    total = sum(counts.values())

    print()
    for name in CLASSES:
        share = 100.0 * counts[name] / total if total else 0.0
        print(f"{name:>12}: {counts[name]:4d}  ({share:4.1f}%)")

    print(f"\nRegions: {total}")
    print(f"Saved  : {JSON_PATH}")


if __name__ == "__main__":
    main()
