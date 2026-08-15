"""
Check every cleaned page for leftover dark border, and rank the worst.

Written after the third round of "there is still black on page X":
finding these one at a time by eye does not scale to 1385 pages, and
each miss was a different defect (scanner lip, booklet edge, shadow
gradient). This looks at all of them at once and says which pages are
still wrong, so the threshold can be judged against the whole corpus
rather than against whichever page happened to be opened.

A cleaned page should be pure white to the edge: cleaning trims the
border and then pads back to canonical size with white, so anything
dark in the outer band is residue that escaped the trim.

Run:
    python audit_borders.py
    python audit_borders.py --band 40 --worst 25
    python audit_borders.py --dump      # write the worst offenders out
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

CLEAN_DIR = BASE_DIR / "cleaned"
DUMP_DIR = BASE_DIR / "border_audit"

# Outer band inspected on each side, in pixels.
BAND = 30

# A border pixel this dark is residue. Padding writes exactly 255, and
# real content never reaches the very edge after trimming.
DARK = 200

# Share of the band allowed to be dark before a page is reported. A
# little tolerance absorbs JPEG-free PNG rounding and the odd speck.
TOLERANCE = 0.002


def border_darkness(gray, band):
    """Fraction of the outer band that is darker than DARK."""

    top = gray[:band, :]
    bottom = gray[-band:, :]
    left = gray[:, :band]
    right = gray[:, -band:]

    parts = [top, bottom, left, right]

    dark = sum(int((p < DARK).sum()) for p in parts)
    total = sum(p.size for p in parts)

    per_side = {
        name: float((p < DARK).mean())
        for name, p in zip(("top", "bottom", "left", "right"), parts)
    }

    return dark / max(1, total), per_side


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--band", type=int, default=BAND)
    parser.add_argument("--worst", type=int, default=15)
    parser.add_argument("--dump", action="store_true")

    args = parser.parse_args()

    pages = sorted(CLEAN_DIR.glob("student_*/cie_*/page_*.png"))

    if not pages:
        raise SystemExit(f"nothing under {CLEAN_DIR}")

    results = []

    for index, path in enumerate(pages, start=1):

        if index % 300 == 0:
            print(f"  {index}/{len(pages)}", flush=True)

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            continue

        score, per_side = border_darkness(gray, args.band)

        results.append((score, per_side, path))

    results.sort(key=lambda r: -r[0])

    failing = [r for r in results if r[0] > TOLERANCE]

    print(f"\nPages checked : {len(results)}")
    print(f"Border band   : {args.band}px on each side")
    print(f"Clean         : {len(results) - len(failing)}")
    print(f"With residue  : {len(failing)}  "
          f"({100 * len(failing) / max(1, len(results)):.1f}%)")

    if failing:

        print(f"\nWorst {min(args.worst, len(failing))}:")
        print(f"  {'page':<34}{'dark%':>7}   per-side dark%")

        for score, per_side, path in failing[:args.worst]:

            sides = "  ".join(
                f"{name}={100 * value:.1f}"
                for name, value in per_side.items() if value > 0.001
            )

            relative = path.relative_to(CLEAN_DIR)

            print(f"  {str(relative):<34}{100 * score:>6.2f}   {sides}")

    if args.dump and failing:

        DUMP_DIR.mkdir(parents=True, exist_ok=True)

        for stale in DUMP_DIR.glob("*.png"):
            stale.unlink()

        for score, _, path in failing[:args.worst]:

            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            name = str(path.relative_to(CLEAN_DIR)).replace("/", "_")

            cv2.imwrite(str(DUMP_DIR / name), image)

        print(f"\nWorst pages written to {DUMP_DIR}")

    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
