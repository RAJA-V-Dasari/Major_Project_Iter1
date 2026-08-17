"""
Detect the printed ruled-line grid on a cropped page.

Reusable per-page tool, not part of the crop itself: the grid's phase
(position of line 1 relative to the true page top) varies by up to
~30px page to page - real variance in paper printing/binding, not
detector noise - so it cannot anchor a corpus-wide-consistent crop.
What it IS reliable for for is describing the ruled lines on ONE page
precisely, e.g. for a future de-ruling stage that needs to know
exactly where each printed line sits so it can be masked out without
touching handwriting.

Built on the same horizontal-opening + Hough approach as deskew.py's
rule_angle() and clean.py's original trim logic, run here on the
already-cropped page rather than the raw scan.

WHY THE EDGE-ARTIFACT FILTER
-----------------------------
Run naively (see session notes / git history), the very top and
bottom rows of a scan produce a SOLID band of "line" detections one
pixel apart (a residual scanner-edge artifact, not print) - a genuine
rule line clusters within a few px, an artifact band spans dozens.
Any cluster wider than MAX_CLUSTER_SPAN is dropped for exactly this
reason.
"""

import cv2
import numpy as np


RULE_KERNEL = 60
HOUGH_THETA = np.pi / 720
HOUGH_THRESHOLD = 200
HOUGH_MIN_LENGTH_FRACTION = 5
HOUGH_MAX_GAP = 40

# Segments within this many px (vertically) belong to the same
# printed line - pitch is ~58-59px corpus-wide, so this comfortably
# separates adjacent lines without merging them.
CLUSTER_GAP = 15

# A genuine rule line's segments cluster within a few px of each
# other. A residual scanner-edge artifact instead produces many
# 1px-apart detections spanning a wide band - reject those.
MAX_CLUSTER_SPAN = 6

# Accepted range for the corpus's rule pitch (~58-59px measured);
# a gap outside this is either a missed blank line (multiple) or
# noise, not a single pitch step.
PITCH_MIN, PITCH_MAX = 45, 70


def detect_rule_lines(gray):
    """
    Ruled-line rows on a page.

    Returns (rows, pitch): `rows` is a sorted array of detected line
    centre y-positions (float, image coordinates), `pitch` is the
    median line-to-line spacing (None if fewer than 2 lines found).
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
        return np.array([]), None

    ys = sorted((seg.ravel()[1] + seg.ravel()[3]) / 2 for seg in lines)

    clusters = [[ys[0]]]

    for y in ys[1:]:
        if y - clusters[-1][-1] < CLUSTER_GAP:
            clusters[-1].append(y)
        else:
            clusters.append([y])

    rows = np.array([
        np.mean(c) for c in clusters
        if (max(c) - min(c)) <= MAX_CLUSTER_SPAN
    ])

    if len(rows) < 2:
        return rows, None

    diffs = np.diff(rows)
    steps = diffs[(diffs >= PITCH_MIN) & (diffs <= PITCH_MAX)]

    pitch = float(np.median(steps)) if len(steps) else None

    return rows, pitch


def preview():

    from pathlib import Path

    stage_dir = Path(__file__).resolve().parent.parent
    out_dir = stage_dir / "output"

    picks = [
        "student_61/cie_3/page_05.png",
        "student_12/cie_1/page_03.png",
        "student_02/cie_1/page_09.png",
    ]

    for relative in picks:

        path = out_dir / relative

        if not path.exists():
            print(f"  {relative}: not found (run crop.py first)")
            continue

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rows, pitch = detect_rule_lines(gray)

        print(f"{relative:30s} n={len(rows):3d}  pitch={pitch}  "
              f"first={rows[0]:.1f}  last={rows[-1]:.1f}" if len(rows)
              else f"{relative}: no lines found")


if __name__ == "__main__":
    preview()
