"""
Layout segmentation for handwritten answer sheets.

This is the production inference entry point. It prefers the fine-tuned
YOLO model produced by train_yolo.py, and falls back to the pretrained
DocLayout-YOLO baseline when no fine-tuned weights exist yet.

The distinction matters: the baseline was trained on printed documents
(DocStructBench) and mislabels handwriting badly - on this project's
sample booklet it called 67% of all regions "figure". Treat any baseline
run as a smoke test, not as real output.

Run:
    python segment.py                       # auto-select best weights
    python segment.py --model path/to.pt    # force a specific checkpoint
    python segment.py --baseline            # force the DocLayout baseline
"""

import argparse
import json
from pathlib import Path


INPUT_DIR = Path("input/pages")
OUTPUT_DIR = Path("output")

JSON_PATH = OUTPUT_DIR / "regions.json"

# Where train_yolo.py leaves its best checkpoint.
FINETUNED_WEIGHTS = Path("runs/answer_sheet_seg/weights/best.pt")

BASELINE_WEIGHTS = Path("models/doclayout_yolo_docstructbench_imgsz1024.pt")

IMAGE_SIZE = 1024
CONFIDENCE = 0.20
DEVICE = "cpu"  # change to "cuda:0" / "0" on a GPU box


def page_sort_key(path):
    """Sort page_2.png before page_10.png."""

    digits = "".join(c for c in path.stem if c.isdigit())

    return int(digits) if digits else 0


class DocumentSegmenter:
    """
    Wraps whichever detector is available behind one interface.

    `backend` is either "ultralytics" (the fine-tuned YOLO11 model) or
    "doclayout" (the pretrained baseline). They have slightly different
    prediction APIs but the same result shape, so only construction and
    the predict call differ.
    """

    def __init__(self, model_path=None, backend=None):

        self.model_path, self.backend = self._resolve(model_path, backend)

        if self.backend == "ultralytics":
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))

        else:
            from doclayout_yolo import YOLOv10

            self.model = YOLOv10(str(self.model_path))

            print(
                "WARNING: using the DocLayout-YOLO baseline, which was "
                "trained on printed documents.\n"
                "         Its labels are unreliable on handwriting - see "
                "README. Run train_yolo.py to replace it."
            )

        print(f"Backend: {self.backend}")
        print(f"Weights: {self.model_path}")

    @staticmethod
    def _resolve(model_path, backend):

        if model_path is not None:
            path = Path(model_path)

            if not path.exists():
                raise FileNotFoundError(path)

            # an explicit path is assumed to be a fine-tuned checkpoint
            # unless the caller said otherwise
            return path, backend or "ultralytics"

        if backend == "doclayout":
            return BASELINE_WEIGHTS, "doclayout"

        if FINETUNED_WEIGHTS.exists():
            return FINETUNED_WEIGHTS, "ultralytics"

        if BASELINE_WEIGHTS.exists():
            return BASELINE_WEIGHTS, "doclayout"

        raise FileNotFoundError(
            "No weights found. Train a model with train_yolo.py, or place "
            f"the baseline checkpoint at {BASELINE_WEIGHTS}."
        )

    def detect_page(self, image_path):

        results = self.model.predict(
            str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None:
            return []

        regions = []

        for region_id, box in enumerate(result.boxes, start=1):

            cls = int(box.cls.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            regions.append(
                {
                    "id": region_id,
                    "label": result.names[cls],
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        return regions

    def run(self):

        images = sorted(INPUT_DIR.glob("*.png"), key=page_sort_key)

        if not images:
            print(f"No images found in {INPUT_DIR}")
            return

        document = {"pages": []}

        for page_number, image in enumerate(images, start=1):

            regions = self.detect_page(image)

            print(f"{image.name}: {len(regions)} region(s)")

            document["pages"].append(
                {
                    "page": page_number,
                    "image": image.name,
                    "regions": regions,
                }
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(JSON_PATH, "w") as f:
            json.dump(document, f, indent=4)

        total = sum(len(p["regions"]) for p in document["pages"])

        print()
        print(f"Pages   : {len(document['pages'])}")
        print(f"Regions : {total}")
        print(f"Saved   : {JSON_PATH}")


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        help="path to a specific checkpoint (default: auto-select)",
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="force the pretrained DocLayout-YOLO baseline",
    )

    args = parser.parse_args()

    segmenter = DocumentSegmenter(
        model_path=args.model,
        backend="doclayout" if args.baseline else None,
    )

    segmenter.run()


if __name__ == "__main__":
    main()
