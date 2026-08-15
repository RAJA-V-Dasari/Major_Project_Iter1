"""
Check the converted tree is complete, correctly named, and readable.

Catches the failure modes that matter here: a gap in the page sequence
(which would mean a page silently vanished), an unreadable PNG, or a
page whose resolution does not match the rest of its booklet.

Run:
    python verify_output.py
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


OUTPUT_DIR = Path(__file__).resolve().parent / "output"

EXPECTED_STUDENTS = range(1, 62)
EXPECTED_CIES = (1, 2, 3)

# Students 32-61 sat only two of the three CIEs, and which two varies.
# A missing third is therefore expected for them, not a gap, and no
# folder should exist for an exam that was never written.
PARTIAL_FROM = 32

# Known-absent in the source; reported as expected rather than as errors.
KNOWN_MISSING = {
    19: "entire student absent from source repo",
    18: "only CIE-1 present in source",
}


def main():

    if not OUTPUT_DIR.exists():
        sys.exit(f"{OUTPUT_DIR} not found - run convert_dataset.py first")

    problems = []

    sizes = Counter()

    total_pages = 0

    booklets = 0

    for student in EXPECTED_STUDENTS:

        student_dir = OUTPUT_DIR / f"student_{student:02d}"

        if not student_dir.exists():
            note = KNOWN_MISSING.get(student, "UNEXPECTED - not in output")
            print(f"student_{student:02d}: absent ({note})")
            continue

        summary = []

        expects_all = student < PARTIAL_FROM

        for cie in EXPECTED_CIES:

            cie_dir = student_dir / f"cie_{cie}"

            if not cie_dir.exists():
                if expects_all and student not in KNOWN_MISSING:
                    problems.append(
                        f"student_{student:02d}/cie_{cie}: MISSING (unexpected)"
                    )
                continue

            pages = sorted(
                cie_dir.glob("page_*.png"),
                key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
            )

            if not pages:
                problems.append(f"student_{student:02d}/cie_{cie}: no pages")
                continue

            booklets += 1

            numbers = [
                int(re.search(r"(\d+)", p.stem).group(1)) for p in pages
            ]

            # page numbering must be 1..N with no gaps and no repeats
            if numbers != list(range(1, len(numbers) + 1)):
                problems.append(
                    f"student_{student:02d}/cie_{cie}: page numbering broken "
                    f"-> {numbers}"
                )

            page_sizes = set()

            for page in pages:
                try:
                    with Image.open(page) as image:
                        image.verify()
                    with Image.open(page) as image:
                        page_sizes.add(image.size)
                        sizes[image.size] += 1
                except Exception as exc:
                    problems.append(
                        f"{page.relative_to(OUTPUT_DIR)}: unreadable ({exc})"
                    )

            if len(page_sizes) > 1:
                problems.append(
                    f"student_{student:02d}/cie_{cie}: mixed page sizes "
                    f"{sorted(page_sizes)}"
                )

            total_pages += len(pages)

            summary.append(f"cie_{cie}={len(pages)}p")

        print(f"student_{student:02d}: {'  '.join(summary)}")

    print()
    print(f"Booklets : {booklets}")
    print(f"Pages    : {total_pages}")

    print("\nResolutions:")
    for size, count in sizes.most_common(8):
        print(f"  {size[0]:>5} x {size[1]:<5}  {count:>4} pages")

    if problems:
        print(f"\n--- {len(problems)} PROBLEM(S) ---")
        for problem in problems:
            print(" *", problem)
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
