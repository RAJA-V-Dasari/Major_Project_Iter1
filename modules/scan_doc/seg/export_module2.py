"""
Emit segmentation results in the exact shape Module 2 (the region router)
expects, and write the region crops it references.

Module 1 and Module 2 do not currently share a format:

    Module 1 (segment.py, detect_lines.py) writes
        {"pages": [{"page": 1, "image": "...", "regions": [
            {"id", "label", "confidence", "bbox": [x1,y1,x2,y2]}]}]}

    Module 2 (router.load) reads
        {"regions": [
            {"id", "page", "label", "confidence",
             "bbox": [x1,y1,x2,y2], "crop_path"}]}

So Module 2 needs a flat region list, each region carrying its own page
number and a path to a cropped image. This script performs that
conversion and does the cropping.

Run:
    python export_module2.py                     # from output/lines.json
    python export_module2.py output/final.json   # from any regions file

Output:
    output/segmentation_output.json   -> copy to modules/module2_router/input/
    output/crops/page<N>/<id>_<label>.png
"""

import json
import shutil
import sys
from pathlib import Path

import cv2

from export_preannotations import SOURCE_LABEL_MAP


INPUT_DIR = Path("input/pages")
OUTPUT_DIR = Path("output")

CROP_DIR = OUTPUT_DIR / "crops"
OUTPUT_JSON = OUTPUT_DIR / "segmentation_output.json"

DEFAULT_SOURCE = OUTPUT_DIR / "lines.json"

# Pixels of context kept around each crop, so downstream OCR is not fed
# glyphs clipped flush against the box edge.
CROP_PADDING = 8


def export(source_path, clean=True):

    with open(source_path) as f:
        document = json.load(f)

    if clean and CROP_DIR.exists():
        shutil.rmtree(CROP_DIR)

    CROP_DIR.mkdir(parents=True, exist_ok=True)

    regions_out = []

    region_id = 1

    for page in document["pages"]:

        page_number = page["page"]

        image_path = INPUT_DIR / page["image"]

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"  ! skipping missing image {image_path}")
            continue

        height, width = image.shape[:2]

        page_dir = CROP_DIR / f"page{page_number}"
        page_dir.mkdir(parents=True, exist_ok=True)

        for region in page["regions"]:

            x1, y1, x2, y2 = region["bbox"]

            cx1 = max(0, x1 - CROP_PADDING)
            cy1 = max(0, y1 - CROP_PADDING)
            cx2 = min(width, x2 + CROP_PADDING)
            cy2 = min(height, y2 + CROP_PADDING)

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            # detector-native labels ("line", "nontext", "figure") mean
            # nothing to Module 2's ROUTING_TABLE, so translate into its
            # vocabulary; anything unrecognised becomes manual_review
            label = SOURCE_LABEL_MAP.get(region["label"], region["label"])

            crop_name = f"{region['id']}_{label}.png"
            crop_path = page_dir / crop_name

            cv2.imwrite(str(crop_path), image[cy1:cy2, cx1:cx2])

            regions_out.append(
                {
                    "id": region_id,
                    "page": page_number,
                    "label": label,
                    "confidence": region.get("confidence", 0.0),
                    "bbox": [x1, y1, x2, y2],
                    # relative to this module's output dir, so Module 2
                    # can resolve it from wherever it is mounted
                    "crop_path": str(crop_path.relative_to(OUTPUT_DIR)),
                }
            )

            region_id += 1

        print(f"page {page_number}: {len(page['regions'])} region(s)")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"regions": regions_out}, f, indent=4)

    return regions_out


def main():

    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE

    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. Run detect_lines.py (or segment.py) first."
        )

    regions = export(source)

    print()
    print(f"Source : {source}")
    print(f"Regions: {len(regions)}")
    print(f"Crops  : {CROP_DIR}")
    print(f"Saved  : {OUTPUT_JSON}")
    print()
    print("Hand off to Module 2 with:")
    print(f"  cp {OUTPUT_JSON} ../../module2_router/input/segmentation_output.json")


if __name__ == "__main__":
    main()
