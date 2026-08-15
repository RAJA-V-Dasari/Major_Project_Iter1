"""
Check every converted page is the right way up - individually.

Written after two failed attempts to fix orientation by sampling a few
pages and applying one blanket transform. Both were wrong. This script
measures each page on its own and reports the outliers, so the question
"are they all correct?" is answered by evidence rather than by spot
checks.

The signal
----------
Handwriting on a ruled page is not vertically symmetric. Within each
ruled line the ink sits mostly just ABOVE the rule it is written on -
letters rest on the baseline and ascenders reach up. Flip the page and
that asymmetry inverts.

So for each detected text band, this measures where the ink's centre of
mass sits within the band. Upright pages skew one way, flipped pages
the other. `ink_balance` is the mean of that, roughly:

    < 0.5  ink concentrated in the upper part of its band
    > 0.5  ink concentrated in the lower part

The absolute direction is calibrated from the corpus itself (the
majority orientation), so this reports pages that DISAGREE WITH THE
MAJORITY rather than relying on a hardcoded threshold that may not
transfer to a different notebook ruling.

This flags candidates for a human to eyeball - it is not authoritative.
Use --contact-sheet to render the outliers for checking.

Run:
    python check_orientation.py
    python check_orientation.py --contact-sheet
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
REPORT_DIR = OUTPUT_DIR / "_orientation_check"

# Bands thinner than this fraction of the median band height are noise.
MIN_BAND_FRACTION = 0.35

# Pages whose balance sits this far from the corpus median are reported.
OUTLIER_MARGIN = 0.04


def ink_balance(image_path):
    """
    Mean vertical position of ink within its own text band, 0..1.

    Returns None when the page has too little structure to judge.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    mask = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )

    rows = mask.sum(axis=1).astype(np.float64) / 255.0

    if rows.max() <= 0:
        return None

    inked = rows >= 0.08 * rows.max()

    bands = []
    start = None

    for y, flag in enumerate(inked):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            bands.append((start, y))
            start = None

    if start is not None:
        bands.append((start, len(inked)))

    if not bands:
        return None

    heights = [b - a for a, b in bands]

    floor = MIN_BAND_FRACTION * float(np.median(heights))

    bands = [(a, b) for a, b in bands if (b - a) >= floor]

    if len(bands) < 4:
        return None

    positions = []

    for top, bottom in bands:

        segment = rows[top:bottom]

        total = segment.sum()

        if total <= 0:
            continue

        centre = float((segment * np.arange(len(segment))).sum() / total)

        positions.append(centre / max(1.0, len(segment) - 1))

    if not positions:
        return None

    return float(np.mean(positions))


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="render flagged pages to output/_orientation_check",
    )

    args = parser.parse_args()

    pages = sorted(OUTPUT_DIR.glob("student_*/cie_*/page_*.png"))

    if not pages:
        sys.exit(f"no pages under {OUTPUT_DIR}")

    print(f"Measuring {len(pages)} page(s) ...\n")

    scored = []
    unmeasurable = []

    for index, page in enumerate(pages, start=1):

        value = ink_balance(page)

        if value is None:
            unmeasurable.append(page)
        else:
            scored.append((value, page))

        if index % 200 == 0:
            print(f"  {index}/{len(pages)}")

    if not scored:
        sys.exit("could not measure any page")

    values = np.array([v for v, _ in scored])

    median = float(np.median(values))

    outliers = [
        (v, p) for v, p in scored if abs(v - median) > OUTLIER_MARGIN
    ]

    print(f"\nmedian ink_balance : {median:.4f}")
    print(f"spread (5-95 pct)  : {np.percentile(values,5):.4f} "
          f"- {np.percentile(values,95):.4f}")
    print(f"pages measured     : {len(scored)}")
    print(f"unmeasurable       : {len(unmeasurable)}")
    print(f"disagree w/ median : {len(outliers)}")

    if outliers:
        print("\n--- pages to eyeball (worst first) ---")
        for value, page in sorted(
            outliers, key=lambda t: -abs(t[0] - median)
        )[:40]:
            print(f"  {value:.4f}  ({value-median:+.4f})  "
                  f"{page.relative_to(OUTPUT_DIR)}")

    if unmeasurable:
        print("\n--- unmeasurable (blank or unusual) ---")
        for page in unmeasurable[:20]:
            print(f"  {page.relative_to(OUTPUT_DIR)}")

    if args.contact_sheet and outliers:

        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        for value, page in sorted(
            outliers, key=lambda t: -abs(t[0] - median)
        )[:24]:

            image = cv2.imread(str(page))

            if image is None:
                continue

            height, width = image.shape[:2]

            small = cv2.resize(image, (width // 4, height // 4))

            name = "__".join(page.relative_to(OUTPUT_DIR).parts)

            cv2.imwrite(str(REPORT_DIR / f"{value:.3f}__{name}"), small)

        print(f"\nWrote contact sheet to {REPORT_DIR}")


if __name__ == "__main__":
    main()
