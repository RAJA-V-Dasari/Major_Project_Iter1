"""Draw the reader's figure boxes back onto the pages, for eyeballing.

A box is only useful if it actually encloses the figure. The
coordinates come back as round numbers in the model's own resized
space, which is the signature of a guess rather than a measurement, so
they have to be looked at rather than trusted.

Writes one PNG per page carrying a box, into --out.

    python modules/02_read/src/render_boxes.py modules/02_read/output/batch00 --out <dir>
"""
import argparse, pathlib, re, sys

from PIL import Image, ImageDraw

PAGES = pathlib.Path("modules/01_prepare/03_tone/output")
STEM = re.compile(r"^s(\d+)_c(\d+)_p(\d+)$")
BOX = re.compile(r"!\[(diagram|table)\]\((\d+),(\d+),(\d+),(\d+)\)")


def page_path(stem):
    m = STEM.match(stem)
    if not m:
        return None
    s, c, p = m.groups()
    return PAGES / f"student_{s}" / f"cie_{c}" / f"page_{p}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--scale", type=float, default=0.55)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    made = 0

    for f in sorted(args.batch.glob("*.md")):
        boxes = BOX.findall(f.read_text(encoding="utf-8"))
        if not boxes:
            continue
        img = page_path(f.stem)
        if img is None or not img.exists():
            print(f"  {f.stem}: no source image", file=sys.stderr)
            continue

        page = Image.open(img).convert("RGB")
        draw = ImageDraw.Draw(page)
        for kind, *xy in boxes:
            x1, y1, x2, y2 = map(int, xy)
            colour = (220, 30, 30) if kind == "diagram" else (30, 90, 220)
            # inset slightly so a full-page box stays visible at the edge
            draw.rectangle([x1 + 3, y1 + 3, x2 - 3, y2 - 3],
                           outline=colour, width=9)

        w, h = page.size
        page = page.resize((int(w * args.scale), int(h * args.scale)),
                           Image.LANCZOS)
        page.save(args.out / f"{f.stem}.png")
        made += 1

    print(f"rendered {made} pages with boxes to {args.out}")


if __name__ == "__main__":
    main()
