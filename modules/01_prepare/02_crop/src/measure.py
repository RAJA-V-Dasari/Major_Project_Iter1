"""
Measure the physical booklet-sheet size across the corpus, before
committing to a crop target.

Every page is the same physical sheet, so its true paper bounding box
(scanner lip trimmed off) should cluster tightly on one (width,
height) - this just measures that and reports the distribution. It
does not crop or write any images; that is crop.py, once a target size
is confirmed.

Detection is the same edge logic as the old clean.py's paper_bbox() /
booklet_bottom(), carried over because it was already tuned against
this exact corpus:

  - top/left/right: walk inward from each edge while the row/column
    mean stays darker than a share of the page median (scanner lip is
    a dark band, sometimes a gradient rather than a sharp line).
  - bottom: a separate detector, because the area below the booklet's
    true bottom edge is BRIGHT desk/background, not dark - a generic
    darkness-based trim walks straight past it.

Unlike clean.py, no extra content-safety margin is subtracted off the
detected bottom edge - that margin existed there to guarantee zero
dark residue in a *cleaned* page. Here the goal is the true physical
boundary, so both trims use the same small TRIM_MARGIN.

Run:
    python measure.py                # full corpus
    python measure.py --limit 100    # a sample
"""

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

SOURCE_DIR = STAGE_DIR / "input"
REPORT_PATH = STAGE_DIR / "output" / "measurements.json"

# Same tuning as the old clean.py - see its comments for how these
# were arrived at (measured against specific worst-case pages).
EDGE_BRIGHTNESS_RATIO = 0.90
EDGE_SMOOTHING = 9
TRIM_MARGIN = 4

# 0.93 was the original value; widened to 0.90 after finding pages
# (e.g. student_61/cie_3/page_05, and several cover pages) whose true
# edge sits as early as ~90% down, which the narrower window missed
# entirely - it would then pick up something else further down (near
# the raw canvas edge) instead. Validated on a 200-page random sample:
# only 4 pages shift at all, and all 4 checked visually were genuine
# earlier edges, not text mistaken for one - kept conservative rather
# than widening further to minimise the chance of a false trigger on
# printed table borders (cover pages have several).
BOOKLET_EDGE_SEARCH_FROM = 0.90
BOOKLET_EDGE_DARKNESS = 0.85


def trim_edges(means, floor):
    """Walk inward from both ends while the line is darker than `floor`."""

    count = len(means)
    half = EDGE_SMOOTHING // 2
    padded = np.pad(means, (half, half), mode="edge")

    smoothed = np.lib.stride_tricks.sliding_window_view(
        padded, EDGE_SMOOTHING
    ).min(axis=1)

    start = 0
    while start < count and smoothed[start] < floor:
        start += 1

    end = count - 1
    while end > start and smoothed[end] < floor:
        end -= 1

    return start, end


def booklet_bottom(gray):
    """Row where the booklet's bottom edge sits, or None."""

    height, width = gray.shape

    start = int(height * BOOKLET_EDGE_SEARCH_FROM)
    means = gray[start:].mean(axis=1)
    floor = BOOKLET_EDGE_DARKNESS * float(np.median(gray))

    dark = np.where(means < floor)[0]

    if len(dark) == 0:
        return None

    return start + int(dark[0])


def paper_bbox(gray):
    """Bounding box of the physical sheet, scanner lip trimmed off."""

    height, width = gray.shape

    floor = EDGE_BRIGHTNESS_RATIO * float(np.median(gray))

    row_start, row_end = trim_edges(gray.mean(axis=1), floor)
    col_start, col_end = trim_edges(gray.mean(axis=0), floor)

    y1 = min(height, row_start + TRIM_MARGIN)
    y2 = max(0, row_end + 1 - TRIM_MARGIN)
    x1 = min(width, col_start + TRIM_MARGIN)
    x2 = max(0, col_end + 1 - TRIM_MARGIN)

    edge = booklet_bottom(gray)

    if edge is not None:
        y2 = min(y2, max(0, edge - TRIM_MARGIN))

    # this much trimming means detection failed; report the page as
    # untrimmed rather than pretend a wrong crop is real
    if (x2 - x1) * (y2 - y1) < 0.5 * width * height:
        return 0, 0, width, height

    return x1, y1, x2, y2


def _worker(relative):

    image = cv2.imread(str(SOURCE_DIR / relative), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return relative, None

    x1, y1, x2, y2 = paper_bbox(image)

    return relative, (x2 - x1, y2 - y1)


def page_list():

    return [
        str(p.relative_to(SOURCE_DIR))
        for p in sorted(SOURCE_DIR.glob("student_*/cie_*/page_*.png"))
    ]


def summarize(label, values):

    values = np.array(values)
    counts = Counter(values.tolist())
    mode_value, mode_count = counts.most_common(1)[0]
    median = int(np.median(values))

    within_2 = int(np.sum(np.abs(values - median) <= 2))

    print(f"{label}:")
    print(f"  min={values.min()}  max={values.max()}  "
          f"median={median}  mode={mode_value} ({mode_count} pages)")
    print(f"  within 2px of median: {within_2}/{len(values)} "
          f"({100 * within_2 / len(values):.1f}%)")

    return median, mode_value


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        raise SystemExit(f"source not found: {SOURCE_DIR}")

    pages = page_list()

    if args.limit:
        pages = pages[:args.limit]

    if not pages:
        raise SystemExit(f"no pages under {SOURCE_DIR}")

    print(f"Measuring {len(pages)} page(s) with {args.workers} worker(s)\n")

    widths, heights = [], []
    results = {}
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for relative, size in pool.map(_worker, pages, chunksize=16):
            if size is None:
                failures.append(relative)
                continue
            widths.append(size[0])
            heights.append(size[1])
            results[relative] = size

    print()
    w_median, w_mode = summarize("Width", widths)
    h_median, h_mode = summarize("Height", heights)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w") as handle:
        json.dump(
            {"sizes": results,
             "width_median": w_median, "width_mode": w_mode,
             "height_median": h_median, "height_mode": h_mode},
            handle, indent=1,
        )

    print(f"\nSuggested crop target: {w_mode} x {h_mode} (mode)  "
          f"/  {w_median} x {h_median} (median)")
    print(f"Full report: {REPORT_PATH}")

    if failures:
        print(f"\n{len(failures)} unreadable: {failures[:5]}")


if __name__ == "__main__":
    main()
