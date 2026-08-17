"""
Flatten the illumination and normalise the tone of every cropped page.

    03_tone/input/  (deskewed + cropped)  ->  03_tone/output/

Two steps, in this order:

1. FLATTEN - divide each pixel by a local background estimate
   (morphological closing), so paper lands near white everywhere.
2. TONE - a linear stretch mapping ink to black and paper to white,
   keeping grey levels in between.

WHY THIS ALSO KILLS BLEED-THROUGH
---------------------------------
Show-through from the reverse side is the dominant defect on this
corpus, and the interesting part is that a global intensity threshold
CANNOT remove it. Measured on the cropped pages: light real strokes
reach ~195 while bleed-through sits ~195-222 - the two populations
overlap, and page histograms show no valley between ink and paper
(a continuous smear from 0 to ~230, then a paper peak at 240-249).

What separates them is not brightness but SHARPNESS. Bleed-through
has diffused through the paper, so it is spread out and sits close to
its local surroundings; a pen stroke is sharp and sits far below them.
Step 1 is therefore doing local background estimation, not just
global illumination correction: dividing by the local background
pushes diffuse bleed-through up to ~249 (within a few % of local
paper) where step 2's clip erases it, while a sharp stroke stays far
below the clip and survives.

This is why the order matters and why flatten cannot be skipped even
though these pages are already fairly evenly lit - measured
within-page spread is only ~8 grey levels (25 at worst), because
02_crop already removed the scanner lip and binding shadow where the
real gradient lived. Flatten earns its place here for the local
effect, not the global one.

GEOMETRY IS UNCHANGED
---------------------
Pixels only. Size, deskew and crop are earlier stages' work, so
anything mapped onto the cropped pages stays valid.

Run:
    python tone.py --preview      # before/after pairs, no writes
    python tone.py                # whole corpus
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

SOURCE_DIR = STAGE_DIR / "input"
OUT_DIR = STAGE_DIR / "output"
PREVIEW_DIR = STAGE_DIR / "preview"

# Size of the local background estimate. Must comfortably exceed the
# widest dark mark, or that mark becomes its own background and fades
# out. Measured across 200 random pages, the thickest marks in the
# corpus survive a 13x13 erosion but are no more than ~20px through,
# so 51 clears them with a wide margin.
#
# Checked visually at 21 / 51 / 81 on the heaviest-marked page: all
# three behave identically there, so this is chosen for the margin on
# thick strokes (21 would be borderline against a 20px mark) rather
# than for any visible difference. 81 is slower for no gain.
BACKGROUND_KERNEL = 51

# The stretch. After flattening, paper sits near 255 everywhere, so
# these are meaningful across pages rather than per-scan.
#
# PAPER_WHITE is the one that matters: everything at or above it is
# clipped to white, i.e. erased. 215 was confirmed by rendering the
# hazard pages (heavy bleed-through, faint writing, densest text,
# worst illumination, cover page) - it removes the show-through while
# keeping light-but-real strokes.
#
# Do NOT tighten this in the hope of a cleaner background. Tried
# 100-190: it erases a little more bleed-through but visibly thins and
# breaks genuinely light handwriting, because narrowing the range also
# lifts the mid-tones (a pixel at 150 maps to 141 instead of 122).
# Losing real strokes is far worse than leaving faint residue for a
# later stage.
INK_BLACK = 90
PAPER_WHITE = 215


def flatten(gray):
    """Divide out the local background; paper lands near 255."""

    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (BACKGROUND_KERNEL, BACKGROUND_KERNEL)
        ),
    )

    background = np.maximum(background, 1).astype(np.float32)

    flat = gray.astype(np.float32) / background * 255.0

    return np.clip(flat, 0, 255).astype(np.uint8)


def tone(gray):
    """
    Ink to black, paper to white, mid-tones stretched.

    Grey levels are kept rather than binarised: stroke weight carries
    information an OCR model can use, and a hard threshold throws it
    away. Binarising is also irreversible - a later stage can always
    threshold this, but cannot recover what a threshold discarded.
    """

    out = gray.astype(np.float32)

    out = (out - INK_BLACK) / float(PAPER_WHITE - INK_BLACK)

    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def process_page(path):
    """Returns the toned page, or None if unreadable."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    return tone(flatten(image))


def _worker(relative):

    result = process_page(SOURCE_DIR / relative)

    if result is None:
        return relative, "unreadable"

    target = OUT_DIR / relative

    target.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(target), result)

    return relative, None


def page_list():

    return [
        str(p.relative_to(SOURCE_DIR))
        for p in sorted(SOURCE_DIR.glob("student_*/cie_*/page_*.png"))
    ]


def preview():
    """Before/after on the pages this stage was tuned against."""

    picks = [
        ("student_08/cie_1/page_11.png", "pen + heavy bleed-through"),
        ("student_33/cie_1/page_04.png", "genuinely light writing"),
        ("student_61/cie_3/page_03.png", "worst illumination spread"),
        ("student_55/cie_2/page_07.png", "densest text"),
        ("student_30/cie_2/page_01.png", "thickest marks"),
        ("student_01/cie_1/page_01.png", "cover page, grey printed text"),
    ]

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for relative, why in picks:

        source = SOURCE_DIR / relative

        if not source.exists():
            continue

        before = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        after = process_page(source)

        if after is None:
            continue

        gap = np.full((before.shape[0], 14), 90, np.uint8)

        cv2.imwrite(str(PREVIEW_DIR / relative.replace("/", "_")),
                    np.hstack([before, gap, after]))

        print(f"  {relative:<32} {why}")

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

    print(f"Toning {len(pages)} page(s) with {args.workers} worker(s)")
    print(f"Background kernel {BACKGROUND_KERNEL}, "
          f"stretch {INK_BLACK}-{PAPER_WHITE}\n")

    done = 0
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        for index, (relative, error) in enumerate(
                pool.map(_worker, pages, chunksize=8), start=1):

            if index % 200 == 0:
                print(f"  {index}/{len(pages)}", flush=True)

            if error:
                failures.append((relative, error))
            else:
                done += 1

    print(f"\nToned  : {done}")
    print(f"Output : {OUT_DIR}")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures[:5]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
