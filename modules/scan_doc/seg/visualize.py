import json
from pathlib import Path

import cv2


COLORS = {
    "title": (0, 0, 255),
    "plain text": (0, 255, 0),
    "abandon": (0, 255, 255),
    "figure": (255, 0, 0),
    "figure_caption": (255, 0, 255),
    "table": (0, 165, 255),
    "table_caption": (255, 255, 0),
    "table_footnote": (42, 42, 165),
    "isolate_formula": (203, 192, 255),
    "formula_caption": (0, 0, 0),
}


INPUT_DIR = Path("input/pages")
OUTPUT_DIR = Path("output/annotated")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with open("output/regions.json") as f:
    data = json.load(f)


for page in data["pages"]:

    image = cv2.imread(str(INPUT_DIR / page["image"]))

    for region in page["regions"]:

        label = region["label"]

        color = COLORS.get(label, (128, 128, 128))

        x1, y1, x2, y2 = region["bbox"]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

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