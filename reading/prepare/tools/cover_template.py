"""
Identify the printed table structure on the cover page (page_01 of
every CIE) and build a consensus template from multiple examples.

Every booklet uses the same printed form, so the table's line
positions should be near-identical relative to the table itself -
but the CROP's own anchor (x1, y1 from crop.py) shifts a few px per
scan (shadow/seam trim varies), so raw pixel positions cannot be
compared directly across pages. Each page is normalised relative to
its own top-left table corner (topmost horizontal line, leftmost
vertical line) before aggregating.

Detection: horizontal-opening / vertical-opening + Hough, same family
of technique as grid.py and the old clean.py, applied in both
orientations to isolate the table's printed box borders from
handwriting and background.

Run:
    python cover_template.py           # build + save the template
    python cover_template.py --preview # draw the template over one page
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
OUT_DIR = STAGE_DIR / "output"

TEMPLATE_PATH = STAGE_DIR / "cover_template.json"

HORIZ_KERNEL = 40
VERT_KERNEL = 40
HOUGH_THETA = np.pi / 720
H_THRESHOLD = 150
V_THRESHOLD = 100
MIN_LEN_FRACTION = 0.15   # of page width/height, minimum segment length
CLUSTER_GAP = 8           # px; segments closer than this are one line

# When matching a line across examples, positions within this many px
# (in the normalised, anchor-relative frame) are the same line.
MATCH_TOLERANCE = 12

# Keep a line in the template only if this fraction of examples had it -
# below this, treat it as scan noise (stray mark, ink, seal bleed)
# rather than a real printed feature.
MIN_SUPPORT_FRACTION = 0.6


def _cluster_1d(vals_with_extent, gap=CLUSTER_GAP):

    vals_with_extent.sort(key=lambda t: t[0])
    clusters = [[vals_with_extent[0]]]

    for v in vals_with_extent[1:]:
        if v[0] - clusters[-1][-1][0] < gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    out = []

    for c in clusters:
        pos = float(np.mean([x[0] for x in c]))
        lo = min(x[1] for x in c)
        hi = max(x[2] for x in c)
        out.append((pos, lo, hi))

    return out


def detect_table_lines(gray):
    """
    Returns (h_lines, v_lines): each a list of (position, start, end)
    - h_lines are (y, x_start, x_end), v_lines are (x, y_start, y_end).
    """

    height, width = gray.shape

    ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    horiz = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZ_KERNEL, 1)),
    )
    h_segments = cv2.HoughLinesP(
        horiz, 1, HOUGH_THETA, threshold=H_THRESHOLD,
        minLineLength=int(width * MIN_LEN_FRACTION), maxLineGap=20,
    )

    vert = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, VERT_KERNEL)),
    )
    v_segments = cv2.HoughLinesP(
        vert, 1, HOUGH_THETA, threshold=V_THRESHOLD,
        minLineLength=int(height * 0.01), maxLineGap=20,
    )

    h_raw = [
        ((s.ravel()[1] + s.ravel()[3]) / 2,
         int(min(s.ravel()[0], s.ravel()[2])),
         int(max(s.ravel()[0], s.ravel()[2])))
        for s in h_segments
    ] if h_segments is not None else []

    v_raw = [
        ((s.ravel()[0] + s.ravel()[2]) / 2,
         int(min(s.ravel()[1], s.ravel()[3])),
         int(max(s.ravel()[1], s.ravel()[3])))
        for s in v_segments
    ] if v_segments is not None else []

    h_lines = _cluster_1d(h_raw) if h_raw else []
    v_lines = _cluster_1d(v_raw) if v_raw else []

    return h_lines, v_lines


def normalise(h_lines, v_lines):
    """
    Shift lines so the table's own top-left corner (topmost horizontal
    line, leftmost vertical line) is the origin - removes the crop's
    per-page anchor offset so examples become comparable.
    """

    if not h_lines or not v_lines:
        return None

    origin_y = min(y for y, _, _ in h_lines)
    origin_x = min(x for x, _, _ in v_lines)

    h_norm = [(y - origin_y, x0 - origin_x, x1 - origin_x)
              for y, x0, x1 in h_lines]
    v_norm = [(x - origin_x, y0 - origin_y, y1 - origin_y)
              for x, y0, y1 in v_lines]

    return h_norm, v_norm


def _aggregate(all_examples, index):
    """
    all_examples: list of (h_norm, v_norm) tuples.
    index: 0 for horizontal lines, 1 for vertical.

    Groups matching lines (within MATCH_TOLERANCE of their primary
    coordinate) across examples, keeping only those seen in at least
    MIN_SUPPORT_FRACTION of examples.
    """

    n = len(all_examples)
    all_lines = []

    for example in all_examples:
        all_lines.extend(example[index])

    all_lines.sort(key=lambda t: t[0])

    groups = []

    for line in all_lines:
        placed = False
        for group in groups:
            if abs(np.mean([x[0] for x in group]) - line[0]) < MATCH_TOLERANCE:
                group.append(line)
                placed = True
                break
        if not placed:
            groups.append([line])

    template = []

    for group in groups:
        support = len(group)
        if support / n < MIN_SUPPORT_FRACTION:
            continue
        pos = float(np.median([g[0] for g in group]))
        start = float(np.median([g[1] for g in group]))
        end = float(np.median([g[2] for g in group]))
        template.append({
            "pos": round(pos, 1),
            "start": round(start, 1),
            "end": round(end, 1),
            "support": support,
            "n_examples": n,
        })

    template.sort(key=lambda d: d["pos"])

    return template


def build_template(cover_pages):

    examples = []
    skipped = []

    for rel in cover_pages:

        path = OUT_DIR / rel

        if not path.exists():
            skipped.append(rel)
            continue

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h_lines, v_lines = detect_table_lines(gray)
        result = normalise(h_lines, v_lines)

        if result is None:
            skipped.append(rel)
            continue

        examples.append(result)

    if not examples:
        raise SystemExit("no usable cover-page examples")

    h_template = _aggregate(examples, 0)
    v_template = _aggregate(examples, 1)

    return {
        "n_examples": len(examples),
        "skipped": skipped,
        "horizontal_lines": h_template,
        "vertical_lines": v_template,
    }


def cover_page_sample(n=15, seed=3):

    import random

    all_covers = sorted(OUT_DIR.glob("student_*/cie_*/page_01.png"))
    rels = [str(p.relative_to(OUT_DIR)) for p in all_covers]
    random.seed(seed)
    return random.sample(rels, min(n, len(rels)))


def preview(template, rel="student_01/cie_1/page_01.png"):

    path = OUT_DIR / rel
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h_lines, v_lines = detect_table_lines(gray)
    origin_y = min(y for y, _, _ in h_lines)
    origin_x = min(x for x, _, _ in v_lines)

    for line in template["horizontal_lines"]:
        y = int(line["pos"] + origin_y)
        x0 = int(line["start"] + origin_x)
        x1 = int(line["end"] + origin_x)
        cv2.line(color, (x0, y), (x1, y), (0, 0, 255), 3)

    for line in template["vertical_lines"]:
        x = int(line["pos"] + origin_x)
        y0 = int(line["start"] + origin_y)
        y1 = int(line["end"] + origin_y)
        cv2.line(color, (x, y0), (x, y1), (255, 0, 0), 3)

    out_path = STAGE_DIR / "preview" / "cover_template.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), color)
    print(f"Template overlay (aligned to {rel}): {out_path}")


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--n", type=int, default=15,
                         help="number of cover-page examples to use")
    args = parser.parse_args()

    cover_pages = cover_page_sample(args.n)
    print(f"Building template from {len(cover_pages)} cover pages")

    template = build_template(cover_pages)

    print(f"  examples used : {template['n_examples']}")
    print(f"  skipped       : {len(template['skipped'])}")
    print(f"  horizontal lines kept : {len(template['horizontal_lines'])}")
    print(f"  vertical lines kept   : {len(template['vertical_lines'])}")

    with open(TEMPLATE_PATH, "w") as handle:
        json.dump(template, handle, indent=1)

    print(f"\nTemplate saved: {TEMPLATE_PATH}")

    if args.preview:
        preview(template)


if __name__ == "__main__":
    main()
