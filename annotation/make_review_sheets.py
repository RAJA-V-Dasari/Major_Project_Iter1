"""
Tile the annotation sample into contact sheets for a fast visual survey.

The point is to decide the LABEL SCHEMA before anyone starts drawing
boxes. Writing a guide that lists five classes, then discovering the
data contains three, wastes the annotation pass - and a class with no
examples cannot be learned no matter how good the model is.

Structure is visible at a small scale even when words are not, which is
what this is for: spotting diagrams, tables and struck-out passages, not
reading answers.

Run:
    python make_review_sheets.py
    python make_review_sheets.py --cols 3 --rows 2 --width 700   # closer
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
REVIEW_DIR = BASE_DIR / "review"
MANIFEST_PATH = BASE_DIR / "manifest.csv"

BANNER = 26
GAP = 8


def load_manifest():

    with open(MANIFEST_PATH) as handle:
        return list(csv.DictReader(handle))


def cell(row, width):

    path = IMAGE_DIR / row["file_name"]

    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = width / image.width
        thumb = image.resize((width, int(image.height * scale)))

    canvas = Image.new("RGB", (width, thumb.height + BANNER), (255, 255, 255))
    canvas.paste(thumb, (0, BANNER))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, BANNER], fill=(0, 0, 0))
    draw.text((6, 8), f"{row['file_name']}  [{row['page_kind']}]",
              fill=(255, 255, 255))

    return canvas


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--width", type=int, default=460)
    parser.add_argument("--only", help="substring filter on file_name")

    args = parser.parse_args()

    rows = load_manifest()

    if args.only:
        rows = [r for r in rows if args.only in r["file_name"]]

    if not rows:
        raise SystemExit("no pages matched")

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for stale in REVIEW_DIR.glob("*.png"):
        stale.unlink()

    per_sheet = args.cols * args.rows

    sheets = 0

    for start in range(0, len(rows), per_sheet):

        batch = rows[start:start + per_sheet]

        cells = [cell(r, args.width) for r in batch]

        cell_h = max(c.height for c in cells)

        width = args.cols * args.width + (args.cols - 1) * GAP

        used_rows = (len(cells) + args.cols - 1) // args.cols

        height = used_rows * cell_h + (used_rows - 1) * GAP

        sheet = Image.new("RGB", (width, height), (150, 150, 150))

        for index, item in enumerate(cells):

            x = (index % args.cols) * (args.width + GAP)
            y = (index // args.cols) * (cell_h + GAP)

            sheet.paste(item, (x, y))

        sheets += 1

        sheet.save(REVIEW_DIR / f"sheet_{sheets:02d}.png")

    print(f"{sheets} sheet(s) of up to {per_sheet} pages -> {REVIEW_DIR}")


if __name__ == "__main__":
    main()
