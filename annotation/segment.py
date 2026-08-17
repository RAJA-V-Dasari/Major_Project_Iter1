"""
Locate content on every cleaned page: where it is, not what it is.

Emits a page -> block -> line hierarchy as JSON, plus flat CSV tables
ready for relational storage, plus a rendered image per page.

WHY LOCALISATION ONLY
---------------------
Classifying a region (paragraph / maths / figure / ...) needs training
data this corpus does not have - 27 figure, 10 code and 1 crossed_out
examples - and a wrong class is a claim the pipeline then has to
un-learn. Where the content sits is a far easier question and the one
OCR actually needs answered.

WHY RLSA AND NOT PROJECTION PROFILING
-------------------------------------
Projection profiling finds horizontal bands of ink, so it is blind to
anything that is not laid out in rows. Measured on cleaned pages it put
only 42-95% of the handwriting inside a box: it missed isolated
question numbers ("2b)") and every arrow and vertical stroke of a
diagram.

Run-length smoothing instead smears ink horizontally then vertically
and takes connected components, so every stroke joins some region by
construction. Same pages: 99.2% covered. Uncovered ink is content that
silently never reaches OCR, so that difference is the whole point.

The smear widths were swept rather than guessed:

    hgap  vgap   regions/page   coverage
      36    16          41.7      98.3%
      60    16          17.7      99.2%     <- chosen
      90    16          10.2      99.5%
     150    16           9.8      99.7%

60/16 sits at line granularity - roughly one region per line of
handwriting - which is the unit handwriting OCR reads. Wider smears
keep merging lines into paragraphs, which buys a little coverage and
loses the line structure that makes the output useful.

Run:
    python segment.py                 # every content page
    python segment.py --count 40      # a sample
    python segment.py --no-images
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR.parent / "modules" / "scan_doc_v2"))

from normalize_page import ink_mask, split_horizontal_strokes  # noqa: E402

CORPUS_DIR = BASE_DIR.parent / "preprocessing" / "cleaned"

OUT_DIR = BASE_DIR / "segmented"
IMAGE_OUT = OUT_DIR / "images"

COVER_PAGE = 1

# Run-length smoothing, sized to join the strokes of a WORD only.
# Words are grouped into lines afterwards by position, which is
# independent of how widely a given student spaces their writing.
H_GAP = 22
V_GAP = 10

# Components smaller than this are speckle, not content. One stroke of
# a pen is comfortably larger.
MIN_REGION_AREA = 120

# Lines closer than this many multiples of the rule pitch belong to the
# same block. 1.6 keeps normal paragraph spacing together while a blank
# line still starts a new block.
BLOCK_GAP_PITCH = 1.6

# The ruled-line pitch, in pixels. A CONSTANT, not a per-page
# measurement, because the pages are now canonical: one booklet
# format, one resolution, one size. Measured on pages where the rules
# were cleanly detected (33-35 rules found) it is 58-59 every time.
#
# Estimating it per page was tried and only added failure modes: the
# mask handed to the segmenter has the rules REMOVED, so autocorrelating
# it measures handwriting, which on sparse pages returned 119 and 177
# against a true 59 - inflating the block-grouping threshold until every
# page collapsed into a single block.
RULE_PITCH = 59

# Components whose vertical centres sit within this many pitches of a
# line's running centre belong to that line.
LINE_CLUSTER_PITCH = 0.6

# Taller than this and it is not a row of text - a diagram, a brace, a
# long division - so it is kept as its own region.
TALL_REGION_PITCH = 1.6

INK_THRESHOLD = 160

LINE_COLOUR = (32, 94, 27)       # dark green
BLOCK_COLOUR = (140, 20, 74)     # dark purple

HEADER_HEIGHT = 54
JPEG_QUALITY = 90


def content_pages():
    """Every page except each booklet's identity-bearing cover."""

    by_student = defaultdict(list)

    covers = 0

    for path in sorted(CORPUS_DIR.glob("student_*/cie_*/page_*.png")):

        student = int(re.search(r"student_(\d+)", path.parts[-3]).group(1))
        cie = int(re.search(r"cie_(\d+)", path.parts[-2]).group(1))
        number = int(re.search(r"(\d+)", path.stem).group(1))

        if number == COVER_PAGE:
            covers += 1
            continue

        page_id = f"s{student:02d}_c{cie}_p{number:02d}"

        by_student[student].append((page_id, student, cie, number, path))

    return by_student, covers


def round_robin(by_student, count):
    """One page per student per pass, so a partial run spans all writers."""

    students = sorted(by_student)

    picked = []
    depth = 0

    while len(picked) < count:

        added = False

        for student in students:

            if len(picked) >= count:
                break

            if depth < len(by_student[student]):
                picked.append(by_student[student][depth])
                added = True

        if not added:
            break

        depth += 1

    return picked


def find_components(mask):
    """
    Word-sized ink clusters: a light smear, then connected components.

    The smear is deliberately small. It joins the strokes of a word but
    not the gap between words, because word spacing varies far too much
    between writers here to be a reliable join - one page came out as
    77 fragments at a 60px smear while another merged whole paragraphs.
    Words are grouped into lines afterwards, by position instead.
    """

    smeared = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (H_GAP, 1)),
    )

    smeared = cv2.morphologyEx(
        smeared, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, V_GAP)),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(smeared, 8)

    boxes = []

    for index in range(1, count):

        x, y, w, h, area = stats[index]

        if area >= MIN_REGION_AREA:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))

    return boxes


def find_lines(mask):
    """
    One region per line of handwriting, plus tall regions on their own.

    Components are assigned to a line by clustering their vertical
    CENTRES against the known rule pitch. Two simpler rules were tried
    first and both failed:

      a wider horizontal smear - word spacing is writer-dependent, so
      no single gap gives lines on every page;

      merging components whose vertical ranges overlap - the merged
      band keeps growing as it absorbs each new component, so one tall
      item eventually swallows the page. It returned 3 lines for a full
      page of writing.

    Clustering against a fixed pitch cannot run away like that, because
    the tolerance never grows. Anything taller than a line and a half
    is not text laid out in rows - a diagram, a brace, a long division -
    and is kept as its own region rather than dragged into a line.
    """

    boxes = find_components(mask)

    tall = [b for b in boxes if (b[3] - b[1]) > TALL_REGION_PITCH * RULE_PITCH]

    flat = [b for b in boxes if (b[3] - b[1]) <= TALL_REGION_PITCH * RULE_PITCH]

    flat.sort(key=lambda b: (b[1] + b[3]) / 2)

    tolerance = LINE_CLUSTER_PITCH * RULE_PITCH

    clusters = []

    current = []
    centre = None

    for box in flat:

        box_centre = (box[1] + box[3]) / 2

        if centre is None or abs(box_centre - centre) <= tolerance:
            current.append(box)
            centre = float(np.mean([(b[1] + b[3]) / 2 for b in current]))
        else:
            clusters.append(current)
            current = [box]
            centre = box_centre

    if current:
        clusters.append(current)

    regions = [
        (min(b[0] for b in group), min(b[1] for b in group),
         max(b[2] for b in group), max(b[3] for b in group))
        for group in clusters
    ] + tall

    lines = []

    for x1, y1, x2, y2 in regions:

        lines.append({
            "bbox": [x1, y1, x2, y2],
            "ink": int(mask[y1:y2, x1:x2].sum()),
        })

    # reading order: top to bottom, then left to right within a band
    lines.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    return lines


def group_blocks(lines, pitch):
    """
    Group lines into blocks by vertical gap.

    A block is the unit a reconstruction step would treat as one
    paragraph or one figure; lines are the unit OCR reads.
    """

    if not lines:
        return []

    threshold = BLOCK_GAP_PITCH * pitch

    blocks = []

    current = [lines[0]]

    for line in lines[1:]:

        # the PREVIOUS line's bottom, not the running maximum over the
        # block. Using the maximum collapses the page: one tall region
        # - a diagram, or a brace spanning several lines - sets a
        # bottom that nothing afterwards can clear, so every remaining
        # line joins the same block and the page comes out as 1 block.
        previous_bottom = current[-1]["bbox"][3]

        if line["bbox"][1] - previous_bottom > threshold:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)

    blocks.append(current)

    grouped = []

    for members in blocks:

        x1 = min(m["bbox"][0] for m in members)
        y1 = min(m["bbox"][1] for m in members)
        x2 = max(m["bbox"][2] for m in members)
        y2 = max(m["bbox"][3] for m in members)

        grouped.append({"bbox": [x1, y1, x2, y2], "lines": members})

    return grouped


def ink_coverage(mask, lines):
    """Share of handwriting inside some region."""

    total = int(mask.sum())

    if total == 0:
        return 1.0

    covered = np.zeros(mask.shape, bool)

    for line in lines:
        x1, y1, x2, y2 = line["bbox"]
        covered[y1:y2, x1:x2] = True

    return float((mask.astype(bool) & covered).sum()) / total


def segment_page(path):
    """Returns (blocks, coverage, page_bgr, mask)."""

    page = cv2.imread(str(path))

    if page is None:
        return None, None, None, None

    clean, _, _ = split_horizontal_strokes(ink_mask(page))

    mask = (clean > 0).astype(np.uint8)

    lines = find_lines(mask)

    blocks = group_blocks(lines, RULE_PITCH)

    return blocks, ink_coverage(mask, lines), page, mask


def render(page, blocks, page_id, coverage, out_path):

    height, width = page.shape[:2]

    canvas = np.full((height + HEADER_HEIGHT, width, 3), 255, np.uint8)
    canvas[HEADER_HEIGHT:] = page

    for block in blocks:

        x1, y1, x2, y2 = block["bbox"]

        cv2.rectangle(canvas, (x1 - 6, y1 - 6 + HEADER_HEIGHT),
                      (x2 + 6, y2 + 6 + HEADER_HEIGHT), BLOCK_COLOUR, 3)

        for line in block["lines"]:

            lx1, ly1, lx2, ly2 = line["bbox"]

            cv2.rectangle(canvas, (lx1, ly1 + HEADER_HEIGHT),
                          (lx2, ly2 + HEADER_HEIGHT), LINE_COLOUR, 2)

    total_lines = sum(len(b["lines"]) for b in blocks)

    cv2.rectangle(canvas, (0, 0), (width, HEADER_HEIGHT), (35, 35, 35), -1)

    cv2.putText(
        canvas,
        f"{page_id}   |   {len(blocks)} blocks, {total_lines} lines   |   "
        f"ink covered {100 * coverage:.1f}%",
        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
    )

    cv2.imwrite(str(out_path), canvas,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])


def write_tables(pages):
    """Flat CSVs, one row per entity, ready to load into tables."""

    with open(OUT_DIR / "pages.csv", "w", newline="") as handle:

        writer = csv.writer(handle)
        writer.writerow([
            "page_id", "student_id", "cie", "page_no", "width", "height",
            "n_blocks", "n_lines", "ink_coverage",
        ])

        for page in pages:
            writer.writerow([
                page["page_id"], page["student_id"], page["cie"],
                page["page_no"], page["width"], page["height"],
                len(page["blocks"]),
                sum(len(b["lines"]) for b in page["blocks"]),
                page["ink_coverage"],
            ])

    with open(OUT_DIR / "blocks.csv", "w", newline="") as handle:

        writer = csv.writer(handle)
        writer.writerow([
            "block_id", "page_id", "block_order", "x", "y", "w", "h",
            "n_lines",
        ])

        for page in pages:
            for order, block in enumerate(page["blocks"]):
                x1, y1, x2, y2 = block["bbox"]
                writer.writerow([
                    f"{page['page_id']}_b{order:02d}", page["page_id"],
                    order, x1, y1, x2 - x1, y2 - y1, len(block["lines"]),
                ])

    with open(OUT_DIR / "lines.csv", "w", newline="") as handle:

        writer = csv.writer(handle)
        writer.writerow([
            "line_id", "page_id", "block_id", "line_order",
            "x", "y", "w", "h", "ink_pixels",
        ])

        for page in pages:
            for border, block in enumerate(page["blocks"]):
                for lorder, line in enumerate(block["lines"]):
                    x1, y1, x2, y2 = line["bbox"]
                    writer.writerow([
                        f"{page['page_id']}_b{border:02d}_l{lorder:02d}",
                        page["page_id"],
                        f"{page['page_id']}_b{border:02d}",
                        lorder, x1, y1, x2 - x1, y2 - y1, line["ink"],
                    ])


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--count", type=int)
    parser.add_argument("--no-images", action="store_true")

    args = parser.parse_args()

    if not CORPUS_DIR.exists():
        sys.exit(f"{CORPUS_DIR} not found - run preprocessing/clean.py")

    by_student, covers = content_pages()

    available = sum(len(v) for v in by_student.values())

    selected = round_robin(by_student, args.count or available)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_images:
        IMAGE_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Excluded  : {covers} cover page(s)")
    print(f"Segmenting: {len(selected)} of {available} content pages\n")

    pages = []

    for index, (page_id, student, cie, number, path) in enumerate(
            selected, start=1):

        if index % 100 == 0:
            print(f"  {index}/{len(selected)}", flush=True)

        blocks, coverage, page, _ = segment_page(path)

        if blocks is None:
            continue

        height, width = page.shape[:2]

        pages.append({
            "page_id": page_id,
            "student_id": student,
            "cie": cie,
            "page_no": number,
            "width": width,
            "height": height,
            "ink_coverage": round(coverage, 4),
            "blocks": blocks,
        })

        if not args.no_images:
            render(page, blocks, page_id, coverage,
                   IMAGE_OUT / f"{page_id}.jpg")

    document = {
        "info": {
            "description": (
                "Content localisation for the cleaned handwritten answer "
                "scripts. Regions mark WHERE content is; they carry no "
                "class. Hierarchy is page -> block -> line, in reading "
                "order."
            ),
            "date_created": str(date.today()),
            "h_gap": H_GAP,
            "v_gap": V_GAP,
        },
        "pages": pages,
    }

    with open(OUT_DIR / "segmentation.json", "w") as handle:
        json.dump(document, handle)

    write_tables(pages)

    coverages = [p["ink_coverage"] for p in pages]
    line_counts = [sum(len(b["lines"]) for b in p["blocks"]) for p in pages]
    block_counts = [len(p["blocks"]) for p in pages]

    summary = "\n".join([
        f"pages       : {len(pages)}",
        f"blocks      : {sum(block_counts)}  "
        f"({np.mean(block_counts):.1f} per page)",
        f"lines       : {sum(line_counts)}  "
        f"({np.mean(line_counts):.1f} per page)",
        "",
        f"ink coverage: mean {100 * np.mean(coverages):.1f}%   "
        f"median {100 * np.median(coverages):.1f}%   "
        f"worst {100 * min(coverages):.1f}%",
        f"pages under 95%: {sum(1 for c in coverages if c < 0.95)}",
        f"pages under 90%: {sum(1 for c in coverages if c < 0.90)}",
    ])

    (OUT_DIR / "summary.txt").write_text(summary + "\n")

    print("\n" + summary)
    print(f"\nJSON  : {OUT_DIR / 'segmentation.json'}")
    print(f"Tables: pages.csv, blocks.csv, lines.csv in {OUT_DIR}")

    if not args.no_images:
        print(f"Images: {IMAGE_OUT}")


if __name__ == "__main__":
    main()
