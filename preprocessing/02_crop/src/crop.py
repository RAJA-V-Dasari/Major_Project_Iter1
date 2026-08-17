"""
Crop every page to one fixed booklet-sheet size.

    02_crop/input/  (deskewed, variable scanner lip)  ->  02_crop/output/

The physical page is the same size on every scan; what varies is a
per-scan shadow/margin (mostly on the left, from the booklet binding)
around it. So each page still needs its own paper edge detected (to
find WHERE to anchor the crop), but the crop SIZE is one constant for
the whole corpus - CROP_SIZE, chosen as the 5th percentile of detected
paper size across the corpus (see measure.py), so it stays inside real
paper content on ~95% of pages and at worst leaves a thin shadow strip
on the rest - it never cuts into real content.

Edge detection (paper_bbox / booklet_bottom) is the same logic as
measure.py - see there for how it was validated.

Run:
    python crop.py --preview        # a few pages, crop box overlaid
    python crop.py                  # crop everything
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from measure import (
    SOURCE_DIR, paper_bbox, page_list,
)


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

OUT_DIR = STAGE_DIR / "output"
PREVIEW_DIR = STAGE_DIR / "preview"

# Every booklet is bound (stapled/sewn) a short distance in from the
# true physical left edge - visible as a column of small perforation
# dots, not caught by the brightness-based edge trim since it's paper-
# coloured apart from the dots themselves.
#
# A single fixed offset cannot both fully clear the seam AND never cut
# real content: the seam's measured extent (blob-clustering the dots,
# 150-page sample) reaches ~150px on some pages, but real handwritten
# margin labels ("Q1)", "2a)") start as close as x=22 on others - and
# they are not reliably the same pages, so there is no offset that is
# simultaneously safe for both. Content loss is the worse failure
# mode, so this stays small and conservative: it clears typical
# edge noise but will NOT fully remove the seam on heavily-bound
# pages. A real per-page seam-vs-text discriminator would be needed
# to do better; not attempted here (tried once, too unreliable - see
# git history for that attempt).
SEAM_MARGIN = 15

# (width, height). Width is the 5th-percentile-safe paper width from
# measure.py's report (1613), minus SEAM_MARGIN since the left edge
# now starts that much further in - the right boundary stays at the
# same physical location.
CROP_SIZE = (1613 - SEAM_MARGIN, 2177)

# The one page documented (PREPROCESSING.md) as a genuine ~37%-scale
# outlier - far smaller than CROP_SIZE, so it cannot be cropped like
# the rest without upscaling first. Flagged rather than silently
# mishandled.
KNOWN_UNDERSIZED = {"student_19/cie_2/page_14.png"}


def crop_page(path):
    """Returns (cropped, anchor) or (None, None)."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None, None

    height, width = image.shape
    target_w, target_h = CROP_SIZE

    x1, y1, x2, y2 = paper_bbox(image)

    # skip past the binding seam, then anchor; clamp so the fixed
    # window never runs off the raw image on a page whose detected
    # region is smaller than the target in this dimension
    x1 = min(x1 + SEAM_MARGIN, max(0, width - target_w))
    y1 = min(y1, max(0, height - target_h))

    # Some pages' true content (x2, y2 from paper_bbox) is shorter
    # than the fixed target in one dimension - most often height, when
    # the booklet's bottom edge sits earlier than usual. Grabbing the
    # full target-sized window regardless would run past that edge
    # into the scanner background, pulling in the edge line itself.
    # Stop at the true edge and pad white instead - found on
    # student_61/cie_3/page_05, whose bottom edge sits ~75px short of
    # a full target_h below its top anchor.
    avail_w = max(0, min(target_w, x2 - x1))
    avail_h = max(0, min(target_h, y2 - y1))

    cropped = np.full((target_h, target_w), 255, np.uint8)
    cropped[:avail_h, :avail_w] = image[y1:y1 + avail_h, x1:x1 + avail_w]

    return cropped, (int(x1), int(y1))


def _worker(relative):

    if relative in KNOWN_UNDERSIZED:
        return relative, None, "undersized"

    cropped, anchor = crop_page(SOURCE_DIR / relative)

    if cropped is None:
        return relative, None, "unreadable"

    if cropped.shape[:2] != (CROP_SIZE[1], CROP_SIZE[0]):
        return relative, None, f"short crop {cropped.shape[:2]}"

    target = OUT_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), cropped)

    return relative, anchor, None


def preview():

    picks = [
        "student_12/cie_1/page_03.png",   # heavy left shadow (~70px)
        "student_58/cie_3/page_01.png",   # ~no shadow, seam clearly visible
        "student_01/cie_1/page_01.png",   # cover page
        "student_02/cie_1/page_09.png",   # content reaches near right edge
        "student_03/cie_2/page_04.png",   # blank margin, bleed-through only
        "student_20/cie_2/page_11.png",   # worst-case measured seam extent (155px)
        "student_08/cie_1/page_06.png",   # worst-case measured seam extent (155px)
        "student_59/cie_1/page_02.png",   # no seam cluster detected
        "student_17/cie_2/page_02.png",   # no seam cluster detected
        "student_61/cie_3/page_05.png",   # reported: bottom edge residue
        "student_61/cie_3/page_03.png",   # same booklet, same issue
    ]

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    target_w, target_h = CROP_SIZE

    for relative in picks:

        source = SOURCE_DIR / relative

        if not source.exists():
            continue

        original = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        cropped, (x, y) = crop_page(source)

        if cropped is None:
            continue

        annotated = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(annotated, (x, y), (x + target_w, y + target_h),
                      (0, 0, 255), 3)

        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

        h = max(annotated.shape[0], cropped_bgr.shape[0])

        def pad(img):
            out = np.full((h, img.shape[1], 3), 200, np.uint8)
            out[:img.shape[0]] = img
            return out

        gap = np.full((h, 14, 3), 90, np.uint8)
        pair = np.hstack([pad(annotated), gap, pad(cropped_bgr)])

        cv2.imwrite(str(PREVIEW_DIR / relative.replace("/", "_")), pair)

        print(f"  {relative:<32} anchor=({x},{y})  "
              f"raw={original.shape[1]}x{original.shape[0]} "
              f"-> crop={target_w}x{target_h}")

    print(f"\nLeft: original with crop box overlaid. Right: the crop.")
    print(f"Saved to: {PREVIEW_DIR}")


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    if args.preview:
        preview()
        return 0

    if not SOURCE_DIR.exists():
        raise SystemExit(f"source not found: {SOURCE_DIR}")

    pages = page_list()

    if args.limit:
        pages = pages[:args.limit]

    if not pages:
        raise SystemExit(f"no pages under {SOURCE_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cropping {len(pages)} page(s) to "
          f"{CROP_SIZE[0]}x{CROP_SIZE[1]} with {args.workers} worker(s)\n")

    ok = 0
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (relative, anchor, error) in enumerate(
                pool.map(_worker, pages, chunksize=16), start=1):

            if index % 200 == 0:
                print(f"  {index}/{len(pages)}", flush=True)

            if error:
                failures.append((relative, error))
            else:
                ok += 1

    print(f"\nCropped : {ok}")
    print(f"Output  : {OUT_DIR}")

    if failures:
        print(f"\n{len(failures)} skipped/failed: {failures[:10]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
