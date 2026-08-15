"""
Step 2: cut a page into text lines and paragraph blocks using
projection profiles.

Depends on output/structure.json from normalize_page.py - every
threshold here is expressed as a multiple of that page's measured
pitch, so nothing is hardcoded to one scan resolution or one notebook
ruling.

    horizontal profile  ->  bands of ink  ->  lines
    vertical gaps between lines           ->  blocks
    vertical profile within a block       ->  its left/right extent

Blocks are the unit the user cares about ("paragraphs"), but lines are
kept too: a block can mix prose and an inline calculation, and step 3
needs the finer unit to tell those apart.

Run:
    python segment_blocks.py --visualize
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from normalize_page import (
    INPUT_DIR,
    OUTPUT_DIR,
    ink_mask,
    horizontal_profile,
    page_sort_key,
    split_horizontal_strokes,
)


STRUCTURE_PATH = OUTPUT_DIR / "structure.json"
JSON_PATH = OUTPUT_DIR / "blocks.json"
DEBUG_DIR = OUTPUT_DIR / "blocks_debug"

# A row counts as "inked" when it holds this share of the busiest row's
# ink. Low, because a line of sparse handwriting still matters.
ROW_INK_THRESHOLD = 0.08

# Ignore bands thinner than this fraction of pitch - specks and the
# residue of a printed rule, not a line of writing.
MIN_LINE_HEIGHT_FRACTION = 0.18

# Lines separated by more than this multiple of pitch start a new block.
# Just over 1.0 so a blank ruled line splits paragraphs, while normal
# line-to-line spacing does not.
BLOCK_GAP_PITCH = 1.15

# Trim a block's left/right edges where a column holds less than this
# share of its busiest column.
COLUMN_INK_THRESHOLD = 0.04

# Ignore this fraction of the page at the far left and right when
# measuring a block's extent. These scans have binding holes and shadow
# down the outer edges, which otherwise stretch every box to the page
# border and destroy the left-alignment signal step 3 needs.
EDGE_IGNORE_FRACTION = 0.03

# Drop blocks with almost no ink. Rule removal leaves faint residue on
# blank ruled areas, which otherwise becomes an empty "region" that
# downstream classification still has to label - and gets wrong.
MIN_BLOCK_INK_DENSITY = 0.004


def find_line_bands(mask, pitch):
    """
    Rows of ink, grouped into contiguous bands = candidate text lines.

    Returns:
        list[tuple]: (y1, y2) per band
    """

    profile = horizontal_profile(mask)

    if profile.max() <= 0:
        return []

    inked = profile >= ROW_INK_THRESHOLD * profile.max()

    bands = []

    start = None

    for y, is_ink in enumerate(inked):

        if is_ink and start is None:
            start = y

        elif not is_ink and start is not None:
            bands.append((start, y))
            start = None

    if start is not None:
        bands.append((start, len(inked)))

    minimum = MIN_LINE_HEIGHT_FRACTION * pitch

    return [(y1, y2) for y1, y2 in bands if (y2 - y1) >= minimum]


def group_bands_into_blocks(bands, pitch):
    """Merge vertically adjacent line bands into paragraph blocks."""

    if not bands:
        return []

    threshold = BLOCK_GAP_PITCH * pitch

    blocks = []

    current = [bands[0]]

    for band in bands[1:]:

        gap = band[0] - current[-1][1]

        if gap > threshold:
            blocks.append(current)
            current = [band]
        else:
            current.append(band)

    blocks.append(current)

    return blocks


def horizontal_extent(mask, y1, y2):
    """
    Left and right edge of the ink between two rows, so a block's box
    hugs its content instead of spanning the full page width.
    """

    strip = mask[y1:y2]

    if strip.size == 0:
        return 0, mask.shape[1]

    width = mask.shape[1]

    columns = strip.sum(axis=0).astype(np.float64) / 255.0

    # zero out the binding-hole / shadow strip at either edge before
    # looking for content
    edge = int(EDGE_IGNORE_FRACTION * width)

    if edge > 0:
        columns[:edge] = 0
        columns[-edge:] = 0

    if columns.max() <= 0:
        return 0, width

    inked = np.where(columns >= COLUMN_INK_THRESHOLD * columns.max())[0]

    if len(inked) == 0:
        return 0, width

    return int(inked[0]), int(inked[-1]) + 1


def segment_page(image_path, structure):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    raw = ink_mask(image)

    # profile the page WITHOUT the printed ruling - with it, every ruled
    # row carries ink, the profile has no valleys, and the whole page
    # collapses into a single block
    mask, _, _ = split_horizontal_strokes(raw)

    pitch = structure.get("pitch")

    # Cover sheets have no dependable ruling, so fall back to a pitch
    # derived from the page height. They are still worth segmenting -
    # their tables are real content - just not via the ruling prior.
    if not pitch or not structure.get("is_ruled_page"):
        pitch = max(40, structure["height"] // 40)

    bands = find_line_bands(mask, pitch)

    lines = []

    for band_id, (y1, y2) in enumerate(bands, start=1):

        x1, x2 = horizontal_extent(mask, y1, y2)

        lines.append(
            {
                "id": band_id,
                "bbox": [x1, int(y1), x2, int(y2)],
            }
        )

    blocks = []

    for block_id, group in enumerate(group_bands_into_blocks(bands, pitch), start=1):

        top = group[0][0]
        bottom = group[-1][1]

        x1, x2 = horizontal_extent(mask, top, bottom)

        area = max(1, (x2 - x1) * (bottom - top))

        ink = float(np.count_nonzero(mask[top:bottom, x1:x2]))

        if ink / area < MIN_BLOCK_INK_DENSITY:
            continue

        blocks.append(
            {
                "id": block_id,
                "bbox": [x1, int(top), x2, int(bottom)],
                "line_count": len(group),
            }
        )

    return lines, blocks, image


def annotate(image, lines, blocks, output_path):

    vis = image.copy()

    for line in lines:
        x1, y1, x2, y2 = line["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)

    for block in blocks:
        x1, y1, x2, y2 = block["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 200), 5)

    small = cv2.resize(vis, (vis.shape[1] // 3, vis.shape[0] // 3))

    cv2.imwrite(str(output_path), small)


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    if not STRUCTURE_PATH.exists():
        raise FileNotFoundError(
            f"{STRUCTURE_PATH} not found. Run normalize_page.py first."
        )

    with open(STRUCTURE_PATH) as f:
        structures = json.load(f)

    images = sorted(INPUT_DIR.glob("*.png"), key=page_sort_key)

    if args.visualize:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    document = {"pages": []}

    for page_number, image_path in enumerate(images, start=1):

        structure = structures.get(image_path.name)

        if structure is None:
            print(f"  ! no structure for {image_path.name}, skipping")
            continue

        lines, blocks, image = segment_page(image_path, structure)

        print(
            f"{image_path.name:>14}: {len(lines):3d} lines, "
            f"{len(blocks):3d} blocks"
        )

        document["pages"].append(
            {
                "page": page_number,
                "image": image_path.name,
                "lines": lines,
                "blocks": blocks,
            }
        )

        if args.visualize:
            annotate(image, lines, blocks, DEBUG_DIR / image_path.name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "w") as f:
        json.dump(document, f, indent=2)

    total_lines = sum(len(p["lines"]) for p in document["pages"])
    total_blocks = sum(len(p["blocks"]) for p in document["pages"])

    print()
    print(f"Pages  : {len(document['pages'])}")
    print(f"Lines  : {total_lines}")
    print(f"Blocks : {total_blocks}")
    print(f"Saved  : {JSON_PATH}")


if __name__ == "__main__":
    main()
