"""
Build contact sheets of the identity block from each student's cover
page, so the name and USN can be read off and recorded.

The name and USN are HANDWRITTEN, so there is no reliable automatic way
to read them - general OCR is unreliable on cursive, and a wrong USN is
worse than no USN. This script therefore does the mechanical part
(locate the cover page, crop the identity block, tile them with the
student id burned in) and leaves the reading to a human or a vision
model.

The student id is burned into each crop deliberately: it is what stops
a name being attributed to the wrong student when transcribing.

Run:
    python extract_identity.py            # write contact sheets
    python extract_identity.py --counts   # print page counts only
"""

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
SHEET_DIR = OUTPUT_DIR / "_identity_sheets"

# Vertical slice of the cover page holding "Name of the Candidate"
# through the USN row, as a fraction of page height. Measured on the
# real cover sheets; they are a printed form so the layout is fixed.
CROP_TOP = 0.255
CROP_BOTTOM = 0.395

# Crops per contact sheet. Kept low so the handwriting stays large
# enough to read without ambiguity - a misread USN is worse than one
# extra sheet to look at.
PER_SHEET = 5

SHEET_WIDTH = 1300

CIES = (1, 2, 3)


def student_dirs():

    found = []

    for path in sorted(OUTPUT_DIR.glob("student_*")):

        match = re.fullmatch(r"student_(\d+)", path.name)

        if match:
            found.append((int(match.group(1)), path))

    return found


def page_counts(student_dir):
    """Pages per CIE, 0 when that CIE was not sat."""

    counts = {}

    for cie in CIES:

        cie_dir = student_dir / f"cie_{cie}"

        counts[cie] = (
            len(list(cie_dir.glob("page_*.png"))) if cie_dir.exists() else 0
        )

    return counts


def cover_page(student_dir):
    """
    First page of the earliest CIE the student actually sat.

    Students did not all sit CIE 1, so this cannot assume cie_1 exists.
    """

    for cie in CIES:

        candidate = student_dir / f"cie_{cie}" / "page_01.png"

        if candidate.exists():
            return candidate, cie

    return None, None


def identity_crop(page_path, label, target_width=SHEET_WIDTH):

    with Image.open(page_path) as image:

        image = image.convert("RGB")

        width, height = image.size

        crop = image.crop(
            (0, int(height * CROP_TOP), width, int(height * CROP_BOTTOM))
        )

    scale = target_width / crop.width

    crop = crop.resize(
        (target_width, int(crop.height * scale)), Image.LANCZOS
    )

    # burn the student id in, so a transcription cannot be misattributed
    banner = 34

    canvas = Image.new(
        "RGB", (crop.width, crop.height + banner), (255, 255, 255)
    )

    canvas.paste(crop, (0, banner))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, crop.width, banner], fill=(0, 0, 0))
    draw.text((8, 10), label, fill=(255, 255, 255))

    return canvas


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--counts", action="store_true")
    parser.add_argument(
        "--students",
        help="comma-separated ids to re-check, e.g. 7,13,23. Each is "
             "written as its own full-width crop rather than tiled, "
             "which is what a disputed digit needs.",
    )
    parser.add_argument(
        "--width", type=int, default=SHEET_WIDTH,
        help="crop width in px; raise it to settle an ambiguous digit",
    )

    args = parser.parse_args()

    students = student_dirs()

    if args.students:

        wanted = {int(part) for part in args.students.split(",")}

        students = [(n, p) for n, p in students if n in wanted]

        missing = wanted - {n for n, _ in students}

        if missing:
            raise SystemExit(f"no such student(s): {sorted(missing)}")

    if not students:
        raise SystemExit(f"no students under {OUTPUT_DIR}")

    rows = []

    for number, path in students:

        counts = page_counts(path)

        cover, cie = cover_page(path)

        rows.append(
            {
                "student_id": number,
                "dir": path,
                "cover": cover,
                "cover_cie": cie,
                "cie_1": counts[1],
                "cie_2": counts[2],
                "cie_3": counts[3],
            }
        )

    if args.counts:
        for row in rows:
            print(
                f"student_{row['student_id']:02d}  "
                f"cie1={row['cie_1']:2d} cie2={row['cie_2']:2d} "
                f"cie3={row['cie_3']:2d}  cover=cie_{row['cover_cie']}"
            )
        return

    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    # one file per student, full width - for settling disputed readings
    if args.students:

        for row in rows:

            if row["cover"] is None:
                print(f"  ! student_{row['student_id']:02d}: no cover page")
                continue

            label = (
                f"student_{row['student_id']:02d}   "
                f"(from cie_{row['cover_cie']} page_01)"
            )

            crop = identity_crop(row["cover"], label, args.width)

            path = SHEET_DIR / f"check_{row['student_id']:02d}.png"

            crop.save(path)

            print(f"  {path}")

        return

    for old in SHEET_DIR.glob("*.png"):
        old.unlink()

    batch = []
    sheet_index = 1

    def flush(batch, index):

        if not batch:
            return

        width = max(c.width for c in batch)
        height = sum(c.height for c in batch) + 6 * (len(batch) - 1)

        sheet = Image.new("RGB", (width, height), (210, 210, 210))

        y = 0

        for crop in batch:
            sheet.paste(crop, (0, y))
            y += crop.height + 6

        sheet.save(SHEET_DIR / f"sheet_{index:02d}.png")

    for row in rows:

        if row["cover"] is None:
            print(f"  ! student_{row['student_id']:02d}: no cover page")
            continue

        label = (
            f"student_{row['student_id']:02d}   "
            f"(from cie_{row['cover_cie']} page_01)"
        )

        batch.append(identity_crop(row["cover"], label))

        if len(batch) == PER_SHEET:
            flush(batch, sheet_index)
            sheet_index += 1
            batch = []

    flush(batch, sheet_index)

    # page counts are mechanical - write them now, names get filled in
    # from the sheets afterwards
    partial = OUTPUT_DIR / "_identity_counts.csv"

    with open(partial, "w", newline="") as handle:

        writer = csv.writer(handle)

        writer.writerow(
            ["student_id", "usn", "name", "cie_1_pages", "cie_2_pages",
             "cie_3_pages"]
        )

        for row in rows:
            writer.writerow(
                [
                    f"student_{row['student_id']:02d}",
                    "",
                    "",
                    row["cie_1"],
                    row["cie_2"],
                    row["cie_3"],
                ]
            )

    print(f"Sheets : {SHEET_DIR}  ({sheet_index} file(s))")
    print(f"Counts : {partial}")


if __name__ == "__main__":
    main()
