"""
Draw a page with its line boxes numbered, for transcribing against.

    02_segment/input/<page>
        -> 04_ocr/finetune/numbered/<page_id>.png
        -> 04_ocr/finetune/numbered/<page_id>.boxes.json

WHY NUMBER THE BOXES INSTEAD OF ALIGNING AFTERWARDS
---------------------------------------------------
The obvious route - transcribe the page as prose, then match those
lines to the boxes 02_segment found - was built first and does not
work. Measured over four pages the counts never agreed: 28 text lines
against 19 boxes, 9 against 17, 22 against 24, 13 against 29. The
reasons are structural rather than fixable. A diagram is several boxes
and no text. A stacked fraction is two boxes and one line. A sparse
page picks up boxes on smudges that carry no writing at all.

Any alignment over counts that far apart is a guess, and a wrong guess
pairs a crop with the text of a DIFFERENT line - training data that is
confidently wrong, which is worse for a model than having less of it.

So the boxes are numbered on the image and transcribed against
directly. The pairing is then exact by construction, whatever the
segmentation did: a box over a diagram gets an empty transcription and
is dropped, and nothing is ever silently misaligned.

Run:
    python render_numbered.py --limit 10
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES = STAGE_DIR.parent

sys.path.insert(0, str(MODULES / "02_segment" / "src"))

import segment as S   # noqa: E402

CLEAN = MODULES / "02_segment" / "input"
OUT_DIR = STAGE_DIR / "finetune" / "numbered"

BOX_COLOUR = (40, 120, 40)
TAG_BG = (255, 255, 255)
TAG_FG = (200, 0, 0)


def numbered(path):

    record, gray, _ = S.segment_page(path)

    if record is None:
        return None, []

    boxes = []
    for block in record["blocks"]:
        for line in block["lines"]:
            boxes.append(line["bbox"])

    boxes.sort(key=lambda b: (b[1], b[0]))

    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for index, (x1, y1, x2, y2) in enumerate(boxes):

        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOUR, 2)

        tag = str(index)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # the number sits in the left margin, clear of the writing, so
        # it never covers the thing it is labelling
        tx = max(2, x1 - tw - 10)
        ty = y1 + th

        cv2.rectangle(canvas, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 4),
                      TAG_BG, -1)
        cv2.putText(canvas, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    TAG_FG, 2, cv2.LINE_AA)

    return canvas, boxes


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="*",
                        help="relative paths under 02_segment/input")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.pages:
        relatives = args.pages
    else:
        import json as _json
        bench = _json.load(
            open(MODULES / "06_evaluation" / "bench_pages.json",
                 encoding="utf-8"))
        relatives = [e["path"] for e in bench][:args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for relative in relatives:

        path = CLEAN / relative

        if not path.exists():
            print(f"  missing: {relative}")
            continue

        canvas, boxes = numbered(path)

        if canvas is None:
            print(f"  unreadable: {relative}")
            continue

        parts = Path(relative).parts
        student = int(parts[0].split("_")[1])
        cie = int(parts[1].split("_")[1])
        page = int(Path(parts[2]).stem.split("_")[1])
        key = f"s{student:02d}_c{cie}_p{page:02d}"

        cv2.imwrite(str(OUT_DIR / f"{key}.png"), canvas)

        with open(OUT_DIR / f"{key}.boxes.json", "w", encoding="utf-8") as h:
            json.dump({"page": relative, "boxes": boxes}, h)

        print(f"  {key}: {len(boxes)} boxes")

    print(f"\nNumbered pages: {OUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
