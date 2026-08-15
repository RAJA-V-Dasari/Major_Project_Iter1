"""
Zero-shot bootstrap labeling for handwritten answer sheets.

Uses Grounding DINO (open-vocabulary object detector, free / Apache-2.0,
via HuggingFace transformers) to propose regions for classes that
DocLayout-YOLO cannot reliably identify on handwritten pages
(e.g. "code snippet", "crossed out text").

This is NOT the final production model. It exists to generate a first
pass of labels that a human corrects in CVAT, which then become the
training set for a fine-tuned YOLOv11-seg model (see train_yolo.py).

Run:
    python bootstrap_label.py
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# ==========================
# Configuration
# ==========================

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

INPUT_DIR = Path("input/pages")
OUTPUT_DIR = Path("output")
JSON_PATH = OUTPUT_DIR / "bootstrap_regions.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grounding DINO takes a single lowercase, period-separated prompt string.
# Keep phrasing close to natural language; it matters more than single words.
CLASS_PROMPTS = {
    "table": "table.",
    "sentence": "handwritten sentence.",
    "diagram": "diagram.",
    "code": "code snippet.",
    "equation": "mathematical equation.",
    "crossed_out": "crossed out text.",
}

TEXT_QUERY = " ".join(CLASS_PROMPTS.values())

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25


# ==========================
# Bootstrap Labeler
# ==========================

class BootstrapLabeler:

    def __init__(self, model_id=MODEL_ID, device=DEVICE):

        print(f"Loading {model_id} on {device} ...")

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    def detect_page(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=TEXT_QUERY,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]

        regions = []

        for region_id, (label, score, box) in enumerate(
            zip(results["labels"], results["scores"], results["boxes"]), start=1
        ):

            x1, y1, x2, y2 = box.tolist()

            regions.append(
                {
                    "id": region_id,
                    "label": label.strip().rstrip("."),
                    "confidence": round(float(score), 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        return regions

    def run(self):

        images = sorted(INPUT_DIR.glob("*.png"))

        if not images:
            print("No images found in", INPUT_DIR)
            return

        document = {"pages": []}

        for page_number, image_path in enumerate(images, start=1):

            print(f"Processing {image_path.name}")

            page = {
                "page": page_number,
                "image": image_path.name,
                "regions": self.detect_page(image_path),
            }

            document["pages"].append(page)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(JSON_PATH, "w") as f:
            json.dump(document, f, indent=4)

        print(f"\nSaved: {JSON_PATH}")
        print("Next: correct these pseudo-labels in CVAT, export as YOLO-seg format,")
        print("then run train_yolo.py to fine-tune a lightweight detector on them.")


if __name__ == "__main__":

    labeler = BootstrapLabeler()
    labeler.run()
