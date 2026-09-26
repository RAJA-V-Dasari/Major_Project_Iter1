"""
Cut the faculty marks grid out of each booklet's cover page.

    handoff/data/<booklet>/pages/page_01.png
        -> gold/covers/<booklet>.png          the marks table, and only that
        -> gold/covers/_contact_sheet_NN.png  50 crops on a few sheets

WHY CROP AT ALL
---------------
The cover carries the student's name, USN, signature and booklet number
in its upper half, and the faculty's marks grid in its lower half. We
need the second and have no business handling the first, so the crop is
the privacy control: it is taken BELOW the identity block, and what
comes out the other side cannot identify anyone even if it leaks.

That is why the default window starts at 0.50 of page height. The name
row sits at about 0.30 and the USN row at about 0.37 on these scans, so
half a page of clearance sits between the last identifying field and the
top of the crop. Do not lower TOP to "get a bit more context" - there is
nothing above the grid worth having and a student's name below it.

WHY IT REACHES TO 0.96
----------------------
The bottom of the cover repeats the total in a separate "Marks Obtained
/ Maximum Marks" box. That repeat is free redundancy: it is a second,
independently written copy of the number the grid already totals, so a
misread grid can be caught by disagreeing with it. It costs nothing to
include and it is the cheapest check we have.

WHY A CONTACT SHEET
-------------------
A fixed fractional window is a guess that happens to be right on the
covers that were looked at. Across 50 phone scans some pages sit higher
or lower in frame, and a crop that clips the top row of the grid loses
question 1's marks silently - the file still exists, still opens, and is
still wrong. So every crop is tiled onto a contact sheet and looked at
before a single one is read.

Run:
    python src/crop_covers.py
    python src/crop_covers.py --top 0.46 --bottom 0.98   # if the sheet says so
    python src/crop_covers.py --booklet student_19_cie_2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths

# Fractions of page height. See the module docstring before changing.
TOP = 0.50
BOTTOM = 0.96

# Contact sheet layout.
COLUMNS = 2
ROWS = 5
THUMB_WIDTH = 700


def cover_page(booklet_dir, booklet):
    """The cover image for one booklet, from its own page list.

    Read out of booklet.json rather than assuming page_01: the handoff
    records a `role` per page, and one booklet in the corpus has a page
    that does not exist at all. Trusting the manifest costs nothing and
    means a renumbered booklet cannot silently hand us an answer page.
    """

    for page in booklet["pages"]:
        if page.get("role") == "cover":
            image = page.get("image")
            if not image:
                return None
            return booklet_dir / image
    return None


def crop_one(source, destination, top, bottom):
    from PIL import Image

    with Image.open(source) as image:
        width, height = image.size
        box = (0, int(height * top), width, int(height * bottom))
        crop = image.crop(box)
        crop.save(destination)
        return crop.size


def contact_sheets(crops, out_dir):
    """Tile the crops so a person can check all 50 in a few glances."""

    from PIL import Image, ImageDraw

    per_sheet = COLUMNS * ROWS
    written = []

    for index in range(0, len(crops), per_sheet):
        batch = crops[index:index + per_sheet]
        thumbs = []

        for path in batch:
            with Image.open(path) as image:
                scale = THUMB_WIDTH / image.width
                thumb = image.convert("RGB").resize(
                    (THUMB_WIDTH, int(image.height * scale))
                )
                draw = ImageDraw.Draw(thumb)
                # Label each crop, or a sheet of near-identical forms is
                # impossible to act on when one of them is wrong.
                draw.rectangle([0, 0, THUMB_WIDTH, 26], fill="black")
                draw.text((6, 6), path.stem, fill="white")
                thumbs.append(thumb)

        cell_height = max(t.height for t in thumbs)
        sheet = Image.new(
            "RGB",
            (COLUMNS * THUMB_WIDTH, ROWS * cell_height),
            "white",
        )
        for position, thumb in enumerate(thumbs):
            column, row = position % COLUMNS, position // COLUMNS
            sheet.paste(thumb, (column * THUMB_WIDTH, row * cell_height))

        number = index // per_sheet + 1
        out = out_dir / f"_contact_sheet_{number:02d}.png"
        sheet.save(out)
        written.append(out)

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=float, default=TOP)
    parser.add_argument("--bottom", type=float, default=BOTTOM)
    parser.add_argument("--booklet", help="just this one")
    parser.add_argument("--no-sheet", action="store_true")
    args = parser.parse_args()

    if not 0.45 <= args.top < args.bottom <= 1.0:
        raise SystemExit(
            f"refusing a crop window of {args.top}-{args.bottom}: the top "
            "must stay at or below 0.45 of page height or the crop starts "
            "eating the student's USN. See the module docstring."
        )

    paths.require(paths.HANDOFF_INDEX, "handoff index.json")
    out_dir = paths.COVER_CROPS
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(paths.HANDOFF_INDEX, encoding="utf-8") as handle:
        index = json.load(handle)

    rows = index["booklets"]
    if args.booklet:
        rows = [r for r in rows if r["booklet_id"] == args.booklet]
        if not rows:
            raise SystemExit(f"no booklet {args.booklet} in the handoff")

    crops, missing = [], []

    for row in rows:
        booklet_id = row["booklet_id"]
        booklet_dir = paths.HANDOFF_DATA / booklet_id

        with open(booklet_dir / "booklet.json", encoding="utf-8") as handle:
            booklet = json.load(handle)

        source = cover_page(booklet_dir, booklet)
        if source is None or not source.exists():
            missing.append(booklet_id)
            continue

        destination = out_dir / f"{booklet_id}.png"
        size = crop_one(source, destination, args.top, args.bottom)
        crops.append(destination)
        print(f"{booklet_id}  {size[0]}x{size[1]}")

    print(f"\n{len(crops)} crops -> {out_dir}")
    if missing:
        print(f"no cover image: {', '.join(missing)}")

    if not args.no_sheet and crops:
        sheets = contact_sheets(sorted(crops), out_dir)
        print(f"\n{len(sheets)} contact sheet(s):")
        for sheet in sheets:
            print(f"  {sheet}")
        print(
            "\nLook at these BEFORE reading any crop. Every grid must show "
            "its top row (question 1) and its TOTAL column. If any crop "
            "clips, re-run with a different --top/--bottom."
        )


if __name__ == "__main__":
    main()
