"""
Step 4: draw classified regions, one colour per class.

Run:
    python visualize.py
    python visualize.py --pages 5,12,29     # only these pages
    python visualize.py --scale 3           # downscale factor
"""

import argparse
import json
from pathlib import Path

import cv2

from normalize_page import INPUT_DIR, OUTPUT_DIR


REGIONS_PATH = OUTPUT_DIR / "regions.json"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"

# BGR. Chosen to stay distinguishable on a white page and from each
# other when several land close together.
COLORS = {
    "paragraph": (0, 170, 0),      # green
    "math": (220, 120, 0),         # blue
    "table": (0, 160, 255),        # orange
    "figure": (200, 0, 200),       # magenta
    "crossed_out": (0, 0, 230),    # red
}

LEGEND_ORDER = ["paragraph", "math", "table", "figure", "crossed_out"]


def draw_legend(image, scale):

    x = 20
    y = 20

    box = int(34 / scale) or 1
    step = int(46 / scale) or 1
    font_scale = 0.9 / scale
    thickness = max(1, int(2 / scale))

    for name in LEGEND_ORDER:

        cv2.rectangle(
            image, (x, y), (x + box, y + box), COLORS[name], thickness=-1
        )

        cv2.putText(
            image,
            name,
            (x + box + int(10 / scale), y + box),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )

        y += step


def annotate_page(image, regions, scale, show_labels=True):

    vis = image.copy()

    for region in regions:

        x1, y1, x2, y2 = region["bbox"]

        color = COLORS.get(region["label"], (128, 128, 128))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 6)

        if show_labels:
            cv2.putText(
                vis,
                region["label"],
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                color,
                3,
            )

    small = cv2.resize(vis, (vis.shape[1] // scale, vis.shape[0] // scale))

    draw_legend(small, 1)

    return small


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--pages", help="comma-separated page numbers")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--no-labels", action="store_true")

    args = parser.parse_args()

    if not REGIONS_PATH.exists():
        raise FileNotFoundError(
            f"{REGIONS_PATH} not found. Run classify_blocks.py first."
        )

    with open(REGIONS_PATH) as f:
        document = json.load(f)

    wanted = None

    if args.pages:
        wanted = {int(p) for p in args.pages.split(",") if p.strip()}

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    written = 0

    for page in document["pages"]:

        if wanted is not None and page["page"] not in wanted:
            continue

        image = cv2.imread(str(INPUT_DIR / page["image"]))

        if image is None:
            print(f"  ! missing {page['image']}")
            continue

        small = annotate_page(
            image, page["regions"], args.scale, not args.no_labels
        )

        output = ANNOTATED_DIR / page["image"]

        cv2.imwrite(str(output), small)

        written += 1

    print(f"Wrote {written} annotated page(s) to {ANNOTATED_DIR}")


if __name__ == "__main__":
    main()
