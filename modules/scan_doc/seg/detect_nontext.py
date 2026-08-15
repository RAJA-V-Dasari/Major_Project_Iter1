"""
Non-text region detection for handwritten answer sheets.

docTR (detect_lines.py) finds text extremely well but is blind to
anything that is not text: hand-drawn diagrams, the brackets and arrows
of a worked long division, boxes a student draws around a final answer,
logos and scanner watermarks.

This module covers that gap without needing any training. It isolates
ink, removes the printed rule lines, subtracts everything docTR already
claimed, and reports what is left over as candidate non-text regions.

Together the two detectors give complete page coverage, which is what
makes the CVAT pre-annotation pass worth doing.

Run:
    python detect_nontext.py --visualize
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from detect_lines import (
    INPUT_DIR,
    OUTPUT_DIR,
    LineDetector,
    page_sort_key,
)


JSON_PATH = OUTPUT_DIR / "nontext.json"
ANNOTATED_DIR = OUTPUT_DIR / "nontext_annotated"

# Ink smaller than this fraction of the page is treated as noise/speckle.
MIN_AREA_FRACTION = 0.00040

# A residual blob must be at least this many pixels on both sides to be a
# real drawing rather than a stray stroke fragment.
MIN_SIDE = 60

# Text boxes are grown by this many pixels before being subtracted, so
# ascenders/descenders don't leak through as fake non-text ink.
TEXT_PADDING = 18

# Residual strokes within this distance are merged into one region.
# Kept small: a large kernel chains unrelated fragments into page-wide blobs.
MERGE_KERNEL = 15

# These scans are slightly warped, so a printed rule is only locally
# straight. A shorter probe kernel tolerates that curvature, and the
# detected rules are then thickened before subtraction so no residue is
# left behind to chain fragments together.
RULE_PROBE_FRACTION = 25
RULE_THICKEN = 5

# A proposal covering this much of the page in both directions is a
# merge artefact (page border, leftover rules), not a drawing.
MAX_EXTENT = 0.60

# Proposals must actually contain ink. Sparse boxes are chained noise.
MIN_INK_DENSITY = 0.045


def ink_mask(image_bgr):
    """Binarise the page so ink is white (255) and paper is black."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )


def remove_ruled_lines(mask):
    """
    Strip the printed horizontal rules and the vertical margin rule.

    They span most of the page, so a morphological opening with a long
    thin kernel isolates them cleanly regardless of their printed colour.
    """

    height, width = mask.shape

    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(1, width // RULE_PROBE_FRACTION), 1)
        ),
    )

    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(1, height // RULE_PROBE_FRACTION))
        ),
    )

    rules = cv2.bitwise_or(horizontal, vertical)

    # grow the detected rules so their full printed thickness is removed;
    # leftover slivers are what chain fragments into page-wide blobs
    rules = cv2.dilate(
        rules,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (RULE_THICKEN, RULE_THICKEN)
        ),
    )

    return cv2.subtract(mask, rules)


def subtract_text(mask, text_boxes, padding=TEXT_PADDING):
    """Blank out every region docTR already accounted for."""

    height, width = mask.shape

    out = mask.copy()

    for x1, y1, x2, y2 in text_boxes:

        cv2.rectangle(
            out,
            (max(0, x1 - padding), max(0, y1 - padding)),
            (min(width - 1, x2 + padding), min(height - 1, y2 + padding)),
            0,
            thickness=-1,
        )

    return out


def residual_regions(mask):
    """Merge leftover strokes into regions and return their boxes."""

    height, width = mask.shape

    merged = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (MERGE_KERNEL, MERGE_KERNEL)
        ),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    min_area = MIN_AREA_FRACTION * height * width

    regions = []

    # label 0 is the background component
    for label in range(1, count):

        x, y, w, h, area = stats[label]

        if area < min_area:
            continue

        if w < MIN_SIDE or h < MIN_SIDE:
            continue

        # a box spanning the page in both directions is a merge artefact
        if w > MAX_EXTENT * width and h > MAX_EXTENT * height:
            continue

        # measure ink in the ORIGINAL residual, not the closed mask, so
        # dilation cannot inflate a sparse chain into a confident region
        ink = int(np.count_nonzero(mask[y:y + h, x:x + w]))

        if ink / float(w * h) < MIN_INK_DENSITY:
            continue

        regions.append((x, y, x + w, y + h, ink))

    return regions


class NonTextDetector:

    def __init__(self):

        # reuse the same docTR detector to know what is already text
        self.lines = LineDetector(level="word")

    def process_page(self, image_path):

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        word_regions, _ = self.lines.process_page(image_path)

        text_boxes = [r["bbox"] for r in word_regions]

        mask = ink_mask(image)
        mask = remove_ruled_lines(mask)
        mask = subtract_text(mask, text_boxes)

        regions = []

        for region_id, (x1, y1, x2, y2, area) in enumerate(
            residual_regions(mask), start=1
        ):

            regions.append(
                {
                    "id": region_id,
                    "label": "nontext",
                    # no model score here - this is a geometric proposal,
                    # so report ink density as a rough salience instead
                    "confidence": round(
                        float(area) / max(1, (x2 - x1) * (y2 - y1)), 4
                    ),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        return regions, image

    def run(self, visualize=False):

        images = sorted(INPUT_DIR.glob("*.png"), key=page_sort_key)

        if not images:
            print(f"No images found in {INPUT_DIR}")
            return

        if visualize:
            ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

        document = {"pages": []}

        for page_number, image_path in enumerate(images, start=1):

            regions, image = self.process_page(image_path)

            print(f"{image_path.name}: {len(regions)} non-text region(s)")

            document["pages"].append(
                {
                    "page": page_number,
                    "image": image_path.name,
                    "regions": regions,
                }
            )

            if visualize:

                vis = image.copy()

                for region in regions:
                    x1, y1, x2, y2 = region["bbox"]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 5)

                cv2.imwrite(str(ANNOTATED_DIR / image_path.name), vis)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(JSON_PATH, "w") as f:
            json.dump(document, f, indent=4)

        total = sum(len(p["regions"]) for p in document["pages"])

        print()
        print(f"Pages   : {len(document['pages'])}")
        print(f"Regions : {total}")
        print(f"Saved   : {JSON_PATH}")


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="write annotated images to output/nontext_annotated",
    )

    args = parser.parse_args()

    NonTextDetector().run(visualize=args.visualize)


if __name__ == "__main__":
    main()
