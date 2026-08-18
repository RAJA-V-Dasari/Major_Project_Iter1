"""
Cut every located line region out of its page as its own image.

    02_segment/input/   (the prepared pages)
    02_segment/output/  (the geometry segment.py found)
        -> 02_segment/crops/   one PNG per line, plus manifest.csv

WHY CROP AT ALL, RATHER THAN OCR AT COORDINATES
-----------------------------------------------
Two-stage detect-then-recognise is what OCR engines actually expect:
Tesseract, TrOCR, PaddleOCR and any CRNN take a single line image, not
a page plus a box. Handing them a page means they crop internally on
every call - uncached, unpadded, and repeated on every re-run. Doing
it once here makes the crop a real artefact that can be looked at,
cached and shipped, and lets each line be padded properly.

NATIVE RESOLUTION, NO RESIZING
------------------------------
Every recogniser wants a different input height, and downscaling is
the one step that cannot be undone. These come out at source
resolution and greyscale, exactly as cut; whichever engine is chosen
resizes at load time.

NOTHING IS SILENTLY DROPPED
---------------------------
A region too small to hold a glyph is not written - a 28x1px sliver
of leftover rule is not a line and an OCR queue full of them helps
nobody. But every region segment.py found appears in manifest.csv
either way, with the reason it was skipped, so the count always
reconciles against the segmentation and a mistake here is visible
rather than invisible.

Run:
    python crop_lines.py               # every page
    python crop_lines.py --limit 20    # a sample
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

SOURCE_DIR = STAGE_DIR / "input"
GEOMETRY_DIR = STAGE_DIR / "output"

CROP_DIR = STAGE_DIR / "crops"
MANIFEST_PATH = CROP_DIR / "manifest.csv"

# White margin left around each line, in rule-pitch units so it
# follows the resolution like everything else in this module.
# Recognisers do better with a little space around the glyphs than
# with ink flush to the border, and the ascender/descender of a
# neighbouring line is a pitch away, so this cannot pull one in.
PAD_PITCH = 0.10

# Below either of these a region cannot hold a character and is
# recorded but not written. Measured over the 37,966 regions
# segment.py found: this drops 4,389 of them (11.6%), which are the
# leftover rule fragments and specks - heights run down to 1px, and
# 7.3% of all regions are under 8px tall. The floor is deliberately
# low rather than tidy: a narrow "1" or "l" is only a few px wide, so
# widening it to something that looks neater would start discarding
# real digits.
MIN_HEIGHT_PITCH = 0.20
MIN_WIDTH_PITCH = 0.08

# Paper is white after 03_tone, so pad with white rather than black -
# a black border would read as ink.
PAD_VALUE = 255


def load_pages():
    """Page records from segment.py, which carry the source path."""

    path = GEOMETRY_DIR / "segmentation.json"

    if not path.exists():
        raise SystemExit(f"{path} not found - run segment.py first")

    with open(path) as handle:
        return json.load(handle)


def crop_name(page_id, block_index, line_index):

    return f"{page_id}_b{block_index:02d}_l{line_index:02d}.png"


def _worker(page):
    """One page: read it once, cut every line out of it."""

    image = cv2.imread(str(SOURCE_DIR / page["source"]), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return page["page_id"], [], "unreadable"

    height, width = image.shape

    pitch = page["rule_pitch"]

    pad = int(round(PAD_PITCH * pitch))
    min_h = MIN_HEIGHT_PITCH * pitch
    min_w = MIN_WIDTH_PITCH * pitch

    # crops/student_NN/cie_M/ - mirrors the corpus layout, and keeps
    # any one directory to a few hundred files rather than 33k
    relative_dir = Path(page["source"]).parent
    target_dir = CROP_DIR / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for block_index, block in enumerate(page["blocks"]):

        for line_index, line in enumerate(block["lines"]):

            x1, y1, x2, y2 = line["bbox"]

            row = {
                "page_id": page["page_id"],
                "student": page["student"],
                "cie": page["cie"],
                "page": page["page"],
                "block_id": block_index,
                "line_id": line_index,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "tall": int(line["tall"]),
                "crop": "",
                "skipped": "",
            }

            if (y2 - y1) < min_h or (x2 - x1) < min_w:
                row["skipped"] = "too small to hold a glyph"
                rows.append(row)
                continue

            # pad, then clamp to the page - a line touching the edge
            # gets less padding rather than a wrapped or shifted crop
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(width, x2 + pad)
            py2 = min(height, y2 + pad)

            crop = image[py1:py2, px1:px2]

            if crop.size == 0:
                row["skipped"] = "empty after clamping"
                rows.append(row)
                continue

            # if the clamp ate padding on one side, put it back as
            # white so every crop really does have a margin
            top = pad - (y1 - py1)
            bottom = pad - (py2 - y2)
            left = pad - (x1 - px1)
            right = pad - (px2 - x2)

            if any(v > 0 for v in (top, bottom, left, right)):
                crop = cv2.copyMakeBorder(
                    crop, max(0, top), max(0, bottom),
                    max(0, left), max(0, right),
                    cv2.BORDER_CONSTANT, value=PAD_VALUE,
                )

            name = crop_name(page["page_id"], block_index, line_index)

            cv2.imwrite(str(target_dir / name), crop)

            row["crop"] = str(relative_dir / name)

            rows.append(row)

    return page["page_id"], rows, None


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))

    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        raise SystemExit(f"source not found: {SOURCE_DIR}")

    pages = load_pages()

    if args.limit:
        pages = pages[:args.limit]

    CROP_DIR.mkdir(parents=True, exist_ok=True)

    expected = sum(len(b["lines"]) for p in pages for b in p["blocks"])

    print(f"Cropping {expected} region(s) from {len(pages)} page(s) "
          f"with {args.workers} worker(s)")
    print(f"Padding {PAD_PITCH:.2f} x pitch, native resolution\n")

    all_rows = []
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        for index, (page_id, rows, error) in enumerate(
                pool.map(_worker, pages, chunksize=8), start=1):

            if index % 200 == 0:
                print(f"  {index}/{len(pages)}", flush=True)

            if error:
                failures.append((page_id, error))
            else:
                all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["page_id"], r["block_id"], r["line_id"]))

    with open(MANIFEST_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    written = [r for r in all_rows if r["crop"]]
    skipped = [r for r in all_rows if not r["crop"]]

    widths = np.array([r["width"] for r in written])
    heights = np.array([r["height"] for r in written])

    print(f"\nRegions found : {len(all_rows)}")
    print(f"Crops written : {len(written)}")
    print(f"Skipped       : {len(skipped)} "
          f"({len(skipped) / max(len(all_rows), 1) * 100:.1f}%, "
          f"too small to hold a glyph)")
    print(f"Tall/diagram  : {sum(r['tall'] for r in written)} "
          f"(kept and flagged, not text lines)")
    print(f"Crop size     : median {int(np.median(widths))}x"
          f"{int(np.median(heights))}, "
          f"largest {widths.max()}x{heights.max()}")

    print(f"\nCrops    : {CROP_DIR}")
    print(f"Manifest : {MANIFEST_PATH}")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures[:5]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
