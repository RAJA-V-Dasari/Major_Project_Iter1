"""
Read the regions the router sent to math_ocr.

    05_math/input/routed/**/routed_regions.json   (from the router)
    02_segment/crops/                             (the images)
        -> 05_math/output/   one reading per expression, JSON + CSV
        -> 05_math/preview/  the same, drawn, to check by eye

    router --(processor == "math_ocr")--> prepare --> engine --> LaTeX

Everything the router sends here is kept in the output, including
the expressions that came back empty or unreadable. A recogniser
that silently drops what it cannot read looks far better than it is,
and the count has to reconcile against the routing.

COORDINATES SURVIVE THE WHOLE TRIP
----------------------------------
Each reading carries the box it came from IN PAGE COORDINATES, not
crop coordinates. Three frames are involved - the page, the padded
crop, and the expression cut out of it - and the mapping between
them is exact, so nothing here needs to be re-found later:

    page_x = region.bbox.x1 - pad + expression.offset_x

`pad` is crop_lines.py's margin, 0.10 of the page's rule pitch,
which that stage guarantees on every side of every crop (it pads
white where the page edge ate the margin). The pitch travels in the
routed region's metadata, so this stays right on a page scanned at a
different scale.

That is what makes a reading placeable back on the page - for
overlaying on the original, for feeding a document reconstruction,
or for a marker to check against the script.

Run:
    python math_ocr.py --engine none              # plumbing, no model
    python math_ocr.py --engine sumen --limit 40  # a sample
    python math_ocr.py --engine sumen             # everything routed
    python math_ocr.py --engine trocr --limit 40  # the control
    python math_ocr.py --engine sumen --limit 40 --preview
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import engines
import prepare


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

MODULES_DIR = STAGE_DIR.parent
SEGMENT_DIR = MODULES_DIR / "02_segment"
CROP_DIR = SEGMENT_DIR / "crops"

INPUT_DIR = STAGE_DIR / "input"
ROUTED_DIR = INPUT_DIR / "routed"
OUTPUT_DIR = STAGE_DIR / "output"
PREVIEW_DIR = STAGE_DIR / "preview"

PROCESSOR = "math_ocr"

# crop_lines.py:PAD_PITCH - the margin it leaves around every crop.
# Duplicated deliberately: this module reads the crops, so it needs
# the constant that made them, and importing across stages would
# couple two pipelines that are meant to be runnable apart.
CROP_PAD_PITCH = 0.10


def routed_regions():
    """Every region the router addressed to this module, in order."""

    if not ROUTED_DIR.exists():
        raise SystemExit(
            f"{ROUTED_DIR} not found - run simulate_router.py first "
            "(or point the real router's output here)"
        )

    for path in sorted(ROUTED_DIR.glob("*/*/routed_regions.json")):

        with open(path) as handle:
            document = json.load(handle)

        booklet = f"{path.parent.parent.name}/{path.parent.name}"

        for page in document["pages"]:
            for region in page["regions"]:

                if region["processor"] != PROCESSOR or region["ignored"]:
                    continue

                yield booklet, region


def read_region(region, engine):
    """
    One routed region: prepare it, then read every expression on it.

    A region fans out to N expressions - usually one, sometimes two
    when a student wrote two workings side by side on the same line.
    """

    crop = cv2.imread(str(CROP_DIR / region["crop_path"]), cv2.IMREAD_GRAYSCALE)

    if crop is None:
        return {"error": "crop missing", "expressions": [], "images": []}

    pitch = region["metadata"]["rule_pitch"]

    expressions, stats = prepare.prepare(crop, pitch)

    pad = int(round(CROP_PAD_PITCH * pitch))

    readings = []

    for index, expression in enumerate(expressions, start=1):

        reading = engine.read(expression["image"])

        offset_x, offset_y = expression["offset"]

        height, width = expression["image"].shape

        readings.append({
            "index": index,
            "of": len(expressions),
            "latex": reading["latex"],
            "confidence": reading["confidence"],
            "seconds": reading["seconds"],
            # Where this expression sits on the page, not on the crop.
            "bbox": {
                "x1": region["bbox"]["x1"] - pad + offset_x,
                "y1": region["bbox"]["y1"] - pad + offset_y,
                "x2": region["bbox"]["x1"] - pad + offset_x + width,
                "y2": region["bbox"]["y1"] - pad + offset_y + height,
            },
            "ink_pixels": expression["ink_pixels"],
        })

    return {
        "error": None,
        "expressions": readings,
        # Kept for --preview, which draws each reading under the
        # image it was read from. Returned rather than re-derived:
        # preparing twice would be both wasteful and a chance for
        # the picture and the reading to disagree.
        "images": [expression["image"] for expression in expressions],
        "rule_pixels_erased": stats["rule_pixels_erased"],
        "x_height": stats["x_height"],
    }


def _preview(rows, engine_name):
    """
    Every reading drawn under the image it was read from.

    The only honest way to look at this stage's output. A LaTeX
    string in a JSON file gives no clue whether it matches the
    handwriting; side by side, a wrong reading is obvious in a
    second.
    """

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    width = 1500
    tiles = []

    for row in rows:

        image = row["image"]

        scale = min(1.0, (width - 40) / image.shape[1])

        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(image.shape[1] * scale), int(image.shape[0] * scale)),
            )

        tile = np.full((image.shape[0] + 62, width), 255, np.uint8)
        tile[8:8 + image.shape[0], 20:20 + image.shape[1]] = image

        head = (
            f"{row['page_id']}  region {row['id']}  "
            f"expression {row['index']}/{row['of']}  "
            f"conf {row['confidence']:.3f}"
        )

        cv2.putText(
            tile, head, (20, image.shape[0] + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, 90, 1, cv2.LINE_AA,
        )

        cv2.putText(
            tile, row["latex"][:150] or "(empty)", (20, image.shape[0] + 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, 0, 1, cv2.LINE_AA,
        )

        tile[-1, :] = 205
        tiles.append(tile)

    if not tiles:
        return

    path = PREVIEW_DIR / f"readings_{engine_name}.png"
    cv2.imwrite(str(path), np.vstack(tiles))

    print(f"preview -> {path}")


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", default="none", choices=sorted(engines.ENGINES),
        help="recogniser to run (default: none - plumbing only)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="stop after this many routed regions (0 = all)",
    )
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument(
        "--preview", action="store_true",
        help="also draw every reading under its image",
    )

    args = parser.parse_args()

    kwargs = {} if args.engine == "none" else {"beams": args.beams}

    engine = engines.load(args.engine, **kwargs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    regions = list(routed_regions())

    if args.limit:
        regions = regions[:args.limit]

    if not regions:
        raise SystemExit("nothing routed to math_ocr - run simulate_router.py")

    print(
        f"{len(regions)} regions routed to {PROCESSOR}, "
        f"engine={args.engine}\n"
    )

    started = time.time()

    results = []
    preview_rows = []

    totals = {"expressions": 0, "empty": 0, "errors": 0}

    for position, (booklet, region) in enumerate(regions, start=1):

        outcome = read_region(region, engine)

        if outcome["error"]:
            totals["errors"] += 1

        page_id = region["metadata"]["page_id"]

        for reading in outcome["expressions"]:

            totals["expressions"] += 1

            if not reading["latex"]:
                totals["empty"] += 1

        results.append({
            "booklet": booklet,
            "page_id": page_id,
            "page": region["page"],
            "region_id": region["id"],
            "reading_order": region["reading_order"],
            "crop_path": region["crop_path"],
            "router_label": region["label"],
            "router_confidence": region["confidence"],
            "router_simulated": region["metadata"].get("simulated", False),
            "bbox": region["bbox"],
            "error": outcome["error"],
            "rule_pixels_erased": outcome.get("rule_pixels_erased"),
            "x_height": outcome.get("x_height"),
            "expressions": outcome["expressions"],
        })

        if args.preview and outcome["expressions"]:

            for reading, image in zip(outcome["expressions"], outcome["images"]):
                preview_rows.append({
                    "image": image,
                    "page_id": page_id,
                    "id": region["id"],
                    "index": reading["index"],
                    "of": reading["of"],
                    "latex": reading["latex"],
                    "confidence": reading["confidence"],
                })

        if position % 10 == 0 or position == len(regions):
            print(
                f"  {position}/{len(regions)} regions, "
                f"{totals['expressions']} expressions, "
                f"{time.time() - started:.0f}s",
            )

    elapsed = round(time.time() - started, 1)

    confidences = [
        reading["confidence"]
        for result in results for reading in result["expressions"]
        if reading["latex"]
    ]

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": engine.describe(),
        "regions": len(results),
        "expressions": totals["expressions"],
        "empty_readings": totals["empty"],
        "errors": totals["errors"],
        "seconds": elapsed,
        "mean_confidence": (
            round(float(np.mean(confidences)), 4) if confidences else 0.0
        ),
    }

    stem = f"math_ocr_{args.engine}"

    with open(OUTPUT_DIR / f"{stem}.json", "w") as handle:
        json.dump({"summary": summary, "regions": results}, handle, indent=4)

    with open(OUTPUT_DIR / f"{stem}.csv", "w", newline="") as handle:

        writer = csv.writer(handle)

        writer.writerow([
            "booklet", "page_id", "region_id", "reading_order",
            "expression", "of", "x1", "y1", "x2", "y2",
            "confidence", "seconds", "latex",
        ])

        for result in results:
            for reading in result["expressions"]:
                writer.writerow([
                    result["booklet"], result["page_id"], result["region_id"],
                    result["reading_order"], reading["index"], reading["of"],
                    reading["bbox"]["x1"], reading["bbox"]["y1"],
                    reading["bbox"]["x2"], reading["bbox"]["y2"],
                    reading["confidence"], reading["seconds"],
                    reading["latex"],
                ])

    if args.preview:
        _preview(preview_rows, args.engine)

    print(
        f"\n{summary['regions']} regions -> {summary['expressions']} "
        f"expressions in {elapsed}s "
        f"({summary['empty_readings']} empty, {summary['errors']} errors)"
    )
    print(f"mean confidence {summary['mean_confidence']}")
    print(f"-> {OUTPUT_DIR / (stem + '.json')}")
    print(f"-> {OUTPUT_DIR / (stem + '.csv')}")


if __name__ == "__main__":
    main()
