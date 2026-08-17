"""
Fix page rotation using the printed rule lines.

    deskew/input/  (raw, as converted from HF)  ->  deskew/output/

The rules are the one landmark on every page, and they are dead
straight, so their angle IS the page angle. Detected with a
probabilistic Hough transform over horizontally-opened ink.

Hough was chosen over two alternatives after measuring all three on
the same pages:
  - connected components of long horizontals found between 2 and 37
    rules per page, because page curvature breaks a rule into
    fragments that individually fail a width test;
  - a rotation sweep maximising projection variance agreed with Hough
    to within 0.11 degrees but is quantised to the sweep step and
    ~80x more work.
Hough finds 67-138 segments on every page tested and returns a
continuous angle.

Carried over as-is from the previous clean.py: only the deskew step.
Trim, tone and canonicalise are separate stages to be rebuilt next.

Run:
    python deskew.py --preview        # before/after pairs, no writes
    python deskew.py                  # deskew everything
    python deskew.py --limit 20       # a sample
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

SOURCE_DIR = STAGE_DIR / "input"
OUT_DIR = STAGE_DIR / "output"

TRANSFORMS_PATH = OUT_DIR / "angles.json"
PREVIEW_DIR = STAGE_DIR / "preview"

# Reference grid drawn only in --preview, so skew is visible against a
# known-horizontal line rather than by eye alone.
REFERENCE_SPACING = 200
REFERENCE_COLOR = (0, 0, 255)   # red, BGR

# Rules run the width of the page; this opening keeps them and drops
# handwriting.
RULE_KERNEL = 60

# A page is never tilted more than a couple of degrees; anything beyond
# this is a mis-detection, not a rotation.
MAX_SKEW = 4.0

# Hough needs enough collinear ink to call something a line.
HOUGH_THRESHOLD = 200
HOUGH_MIN_LENGTH_FRACTION = 5
HOUGH_MAX_GAP = 40

# Angular search step. HoughLinesP returns float pixel endpoints, not
# quantised (rho, theta) bins, and the final angle is a median over
# dozens-to-hundreds of detected segments - so the search step mainly
# controls speed, not final precision.
#
# pi/1440 (0.125 deg) was the original value and cost ~1-7s/page,
# almost entirely inside HoughLinesP. Validated against a 25-page
# random sample from a full pi/1440 corpus run: pi/720 (0.25 deg) cuts
# that to ~0.4s/page (~3x) with mean angle drift 0.026 deg, max 0.091
# deg, and segment counts unchanged or higher - well under the ~4px
# total shift (over a 2338px page) that a 0.1 deg error would cause.
# Going further to pi/360 started dropping real sub-degree corrections
# to 0.0 on pages with fewer rule segments - not worth the extra speed.
HOUGH_THETA = np.pi / 720

# Below this many agreeing segments the angle is not trustworthy and
# the page is left unrotated.
MIN_RULE_SEGMENTS = 12


def rule_angle(gray):
    """
    Page tilt in degrees, from the printed rule lines.

    Returns (angle, segment_count). angle is 0.0 when there is not
    enough evidence - refusing to rotate beats rotating on noise.
    """

    height, width = gray.shape

    ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    horizontals = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (RULE_KERNEL, 1)),
    )

    lines = cv2.HoughLinesP(
        horizontals,
        rho=1,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=width // HOUGH_MIN_LENGTH_FRACTION,
        maxLineGap=HOUGH_MAX_GAP,
    )

    if lines is None:
        return 0.0, 0

    angles = []

    for line in lines:

        x1, y1, x2, y2 = (float(v) for v in line.ravel()[:4])

        if x2 == x1:
            continue

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if abs(angle) <= MAX_SKEW:
            angles.append(angle)

    if len(angles) < MIN_RULE_SEGMENTS:
        return 0.0, len(angles)

    # median, not mean: a few handwriting strokes survive the opening
    # and would drag an average
    return round(float(np.median(angles)), 3), len(angles)


def deskew_page(path):
    """Returns (deskewed_page, transform) or (None, None)."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None, None

    height, width = image.shape

    angle, segments = rule_angle(image)

    if angle != 0.0:

        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

        image = cv2.warpAffine(
            image, matrix, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )

    transform = {
        "angle": angle,
        "rule_segments": segments,
        "source_size": [int(width), int(height)],
    }

    return image, transform


def _worker(relative):

    deskewed, transform = deskew_page(SOURCE_DIR / relative)

    if deskewed is None:
        return relative, None, "unreadable"

    target = OUT_DIR / relative

    target.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(target), deskewed)

    return relative, transform, None


def page_list():

    return [
        str(p.relative_to(SOURCE_DIR))
        for p in sorted(SOURCE_DIR.glob("student_*/cie_*/page_*.png"))
    ]


def with_reference_lines(gray):
    """Grayscale -> BGR with horizontal red guide lines overlaid.

    The lines are dead horizontal by construction, so comparing the
    page's own rule lines against them makes skew (and its correction)
    visible at a glance instead of requiring a protractor.
    """

    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for y in range(REFERENCE_SPACING, color.shape[0], REFERENCE_SPACING):
        cv2.line(color, (0, y), (color.shape[1], y), REFERENCE_COLOR, 1)

    return color


def preview():

    picks = [
        "student_57/cie_1/page_08.png",
        "student_23/cie_2/page_05.png",
        "student_09/cie_2/page_07.png",
        "student_35/cie_3/page_09.png",
        "student_17/cie_3/page_08.png",
        "student_34/cie_3/page_06.png",
    ]

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for relative in picks:

        source = SOURCE_DIR / relative

        if not source.exists():
            continue

        before = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)

        after, transform = deskew_page(source)

        if after is None:
            continue

        before = with_reference_lines(before)
        after = with_reference_lines(after)

        gap = np.full((before.shape[0], 14, 3), 90, np.uint8)

        pair = np.hstack([before, gap, after])

        cv2.imwrite(str(PREVIEW_DIR / relative.replace("/", "_")), pair)

        print(f"  {relative:<32} angle={transform['angle']:+.3f} "
              f"({transform['rule_segments']} segs)")

    print(f"\nBefore | after pairs: {PREVIEW_DIR}")


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

    print(f"Deskewing {len(pages)} page(s) with {args.workers} worker(s)\n")

    transforms = {}
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        for index, (relative, transform, error) in enumerate(
                pool.map(_worker, pages, chunksize=8), start=1):

            if index % 200 == 0:
                print(f"  {index}/{len(pages)}", flush=True)

            if error:
                failures.append((relative, error))
            else:
                transforms[relative] = transform

    with open(TRANSFORMS_PATH, "w") as handle:
        json.dump(transforms, handle, indent=1)

    angles = [t["angle"] for t in transforms.values()]
    segments = [t["rule_segments"] for t in transforms.values()]

    rotated = sum(1 for a in angles if a != 0.0)
    weak = sum(1 for s in segments if s < MIN_RULE_SEGMENTS)

    print(f"\nDeskewed : {len(transforms)}")
    print(f"Rotated  : {rotated}")
    print(f"Skew     : min {min(angles):+.2f}  max {max(angles):+.2f}")
    print(f"Rule segments per page: median {int(np.median(segments))}, "
          f"min {min(segments)}")

    if weak:
        print(f"  {weak} page(s) had too few rules to trust an angle "
              f"and were left unrotated")

    print(f"\nOutput     : {OUT_DIR}")
    print(f"Transforms : {TRANSFORMS_PATH}")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures[:5]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
