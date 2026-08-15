"""
Text-line detection for handwritten answer sheets.

Uses docTR's DBNet text detector, which - unlike DocLayout-YOLO or
Grounding DINO - actually generalises to handwriting (see README for the
comparison that led here). It emits word-level boxes, which this module
groups into lines and optionally into blocks.

The result is used two ways:

  1. As CVAT pre-annotations, so labeling is "assign a class to an
     existing box" rather than "draw every box by hand".
  2. As the geometric front-end for the eventual pipeline, where a
     fine-tuned classifier assigns each line a semantic class.

Run:
    python detect_lines.py                  # all pages -> output/lines.json
    python detect_lines.py --visualize      # also write annotated images
    python detect_lines.py --level word     # skip line grouping
    python detect_lines.py --level block    # group lines into blocks too
"""

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
from doctr.models import detection_predictor


# ==========================
# Configuration
# ==========================

INPUT_DIR = Path("input/pages")
OUTPUT_DIR = Path("output")

JSON_PATH = OUTPUT_DIR / "lines.json"
ANNOTATED_DIR = OUTPUT_DIR / "lines_annotated"

DETECTION_ARCH = "db_resnet50"

# Minimum detector score for a word box to be kept.
WORD_CONFIDENCE = 0.30

# Two words belong to the same line if their vertical centres are within
# this fraction of the page's median word height.
LINE_TOLERANCE = 0.60

# Two lines belong to the same block if the vertical gap between them is
# under this multiple of the median line height.
BLOCK_GAP = 1.20


# ==========================
# Geometry helpers
# ==========================

def detect_words(model, image_bgr, confidence=WORD_CONFIDENCE):
    """
    Run docTR detection and return word boxes in pixel coordinates.

    Returns:
        list[tuple]: (x1, y1, x2, y2, score)
    """

    height, width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = model([image_rgb])[0]

    raw = result["words"] if isinstance(result, dict) else result

    words = []

    for box in raw:

        x1, y1, x2, y2, score = box[:5]

        if score < confidence:
            continue

        words.append(
            (
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height),
                float(score),
            )
        )

    return words


def group_words_into_lines(words, tolerance=LINE_TOLERANCE):
    """
    Cluster word boxes into text lines.

    Words are sorted by vertical centre and swept top to bottom; a word
    joins the current line when its centre sits within `tolerance` x the
    median word height of that line's running centre. Ruled answer
    sheets have well-separated baselines, so this simple sequential
    clustering is reliable and has no training dependency.
    """

    if not words:
        return []

    heights = [y2 - y1 for _, y1, _, y2, _ in words]

    median_height = statistics.median(heights) or 1

    threshold = tolerance * median_height

    ordered = sorted(words, key=lambda w: (w[1] + w[3]) / 2)

    lines = []

    for word in ordered:

        centre = (word[1] + word[3]) / 2

        if lines and abs(centre - lines[-1]["centre"]) <= threshold:

            line = lines[-1]

            line["words"].append(word)

            # running mean keeps the line anchored as it extends sideways
            line["centre"] = sum(
                (w[1] + w[3]) / 2 for w in line["words"]
            ) / len(line["words"])

        else:

            lines.append({"centre": centre, "words": [word]})

    for line in lines:
        line["words"].sort(key=lambda w: w[0])

    return lines


def group_lines_into_blocks(lines, gap=BLOCK_GAP):
    """
    Merge vertically adjacent lines into blocks (answer paragraphs).

    A block break is declared when the gap to the next line exceeds
    `gap` x the median line height, which is what separates one answer
    part from the next on these sheets.
    """

    if not lines:
        return []

    boxes = [line_bbox(line) for line in lines]

    heights = [y2 - y1 for _, y1, _, y2 in boxes]

    median_height = statistics.median(heights) or 1

    threshold = gap * median_height

    blocks = []

    current = [0]

    for i in range(1, len(boxes)):

        previous_bottom = boxes[i - 1][3]
        current_top = boxes[i][1]

        if current_top - previous_bottom > threshold:
            blocks.append(current)
            current = [i]
        else:
            current.append(i)

    blocks.append(current)

    merged = []

    for block in blocks:

        words = []

        for index in block:
            words.extend(lines[index]["words"])

        merged.append({"words": words})

    return merged


def line_bbox(line):
    """Bounding box enclosing every word in a line or block."""

    xs1 = [w[0] for w in line["words"]]
    ys1 = [w[1] for w in line["words"]]
    xs2 = [w[2] for w in line["words"]]
    ys2 = [w[3] for w in line["words"]]

    return (min(xs1), min(ys1), max(xs2), max(ys2))


# ==========================
# Line Detector
# ==========================

class LineDetector:

    def __init__(self, arch=DETECTION_ARCH, level="line"):

        print(f"Loading docTR detector ({arch}) ...")

        self.model = detection_predictor(
            arch,
            pretrained=True,
            assume_straight_pages=True,
        )

        self.level = level

    def process_page(self, image_path):

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        words = detect_words(self.model, image)

        if self.level == "word":
            groups = [{"words": [w]} for w in words]
        else:
            groups = group_words_into_lines(words)

            if self.level == "block":
                groups = group_lines_into_blocks(groups)

        regions = []

        for region_id, group in enumerate(groups, start=1):

            x1, y1, x2, y2 = line_bbox(group)

            scores = [w[4] for w in group["words"]]

            regions.append(
                {
                    "id": region_id,
                    # every region starts life as a generic text line; the
                    # semantic class is assigned later, by a human in CVAT
                    # and then by the fine-tuned model
                    "label": self.level,
                    "confidence": round(sum(scores) / len(scores), 4),
                    "bbox": [x1, y1, x2, y2],
                    "word_count": len(group["words"]),
                }
            )

        return regions, image

    def run(self, visualize=False):

        images = sorted(INPUT_DIR.glob("*.png"), key=page_sort_key)

        if not images:
            print(f"No images found in {INPUT_DIR}")
            return

        document = {"pages": []}

        if visualize:
            ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

        for page_number, image_path in enumerate(images, start=1):

            regions, image = self.process_page(image_path)

            print(f"{image_path.name}: {len(regions)} {self.level}(s)")

            document["pages"].append(
                {
                    "page": page_number,
                    "image": image_path.name,
                    "regions": regions,
                }
            )

            if visualize:
                annotate(image, regions, ANNOTATED_DIR / image_path.name)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(JSON_PATH, "w") as f:
            json.dump(document, f, indent=4)

        total = sum(len(p["regions"]) for p in document["pages"])

        print()
        print(f"Pages   : {len(document['pages'])}")
        print(f"Regions : {total}")
        print(f"Saved   : {JSON_PATH}")

        if visualize:
            print(f"Annotated: {ANNOTATED_DIR}")


def page_sort_key(path):
    """Sort page_2.png before page_10.png."""

    stem = path.stem

    digits = "".join(c for c in stem if c.isdigit())

    return int(digits) if digits else 0


def annotate(image, regions, output_path):

    vis = image.copy()

    for region in regions:

        x1, y1, x2, y2 = region["bbox"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 4)

    cv2.imwrite(str(output_path), vis)


# ==========================
# Main
# ==========================

def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--level",
        choices=["word", "line", "block"],
        default="line",
        help="granularity of the emitted regions (default: line)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="also write annotated images to output/lines_annotated",
    )

    args = parser.parse_args()

    detector = LineDetector(level=args.level)

    detector.run(visualize=args.visualize)


if __name__ == "__main__":
    main()
