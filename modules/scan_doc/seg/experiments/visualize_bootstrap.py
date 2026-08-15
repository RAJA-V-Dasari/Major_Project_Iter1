"""
Visualize Grounding DINO bootstrap pseudo-labels (output/bootstrap_regions.json)
so they can be eyeballed before correcting them in CVAT.

Run:
    python visualize_bootstrap.py
"""

import json
from pathlib import Path

import cv2


COLORS = {
    "table": (0, 165, 255),
    "handwritten sentence": (0, 255, 0),
    "diagram": (255, 0, 0),
    "code snippet": (255, 0, 255),
    "mathematical equation": (203, 192, 255),
    "crossed out text": (0, 0, 255),
}


INPUT_DIR = Path("input/pages")
OUTPUT_DIR = Path("output/bootstrap_annotated")
JSON_PATH = Path("output/bootstrap_regions.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    with open(JSON_PATH) as f:
        data = json.load(f)

    for page in data["pages"]:

        image = cv2.imread(str(INPUT_DIR / page["image"]))

        for region in page["regions"]:

            label = region["label"]
            color = COLORS.get(label, (128, 128, 128))

            x1, y1, x2, y2 = region["bbox"]

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            text = f"{label} ({region['confidence']:.2f})"

            cv2.putText(
                image,
                text,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        output = OUTPUT_DIR / page["image"]
        cv2.imwrite(str(output), image)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
