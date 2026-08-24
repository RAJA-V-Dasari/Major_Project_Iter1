"""Export prepared pages as scaled PNGs for visual review.

    python modules/02_read/src/show_page.py s08_c2_p03 s08_c2_p06 --out DIR
"""
import argparse, pathlib, re
from PIL import Image

PAGES = pathlib.Path("modules/01_prepare/03_tone/output")
STEM = re.compile(r"^s(\d+)_c(\d+)_p(\d+)$")


def page_path(stem):
    m = STEM.match(stem)
    if not m:
        return None
    s, c, p = m.groups()
    return PAGES / f"student_{s}" / f"cie_{c}" / f"page_{p}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--scale", type=float, default=0.55)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for stem in args.stems:
        p = page_path(stem)
        if p is None or not p.exists():
            print(f"  {stem}: missing")
            continue
        im = Image.open(p)
        w, h = im.size
        im.resize((int(w * args.scale), int(h * args.scale)),
                  Image.LANCZOS).save(args.out / f"{stem}.png")
        print(f"  {stem} -> {args.out / (stem + '.png')}")


if __name__ == "__main__":
    main()
