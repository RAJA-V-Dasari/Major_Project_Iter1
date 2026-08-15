"""
Fine-tune a lightweight YOLOv11-seg model on the corrected answer-sheet
dataset (exported from CVAT in YOLO segmentation format).

This is the model that actually gets deployed (segment.py loads its
weights) — Grounding DINO (bootstrap_label.py) is only used offline to
pre-label training data faster.

Expected dataset layout (standard Ultralytics YOLO-seg format), exported
from CVAT into `dataset/`:

    dataset/
        data.yaml
        images/train/*.png
        images/val/*.png
        labels/train/*.txt
        labels/val/*.txt

data.yaml should list your classes, e.g.:

    names: [table, sentence, diagram, code, equation, crossed_out]

Run:
    python train_yolo.py
"""

from pathlib import Path

from ultralytics import YOLO


DATA_YAML = Path("dataset/data.yaml")

# yolo11n-seg = nano segmentation model: smallest/fastest, best fit for
# "lightweight execution" once fine-tuned on a narrow, consistent domain
# like answer-sheet layouts. Bump to yolo11s-seg if accuracy is lacking.
BASE_MODEL = "yolo11n-seg.pt"

EPOCHS = 100
IMG_SIZE = 1024
BATCH = 8


def main():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found. Export your corrected CVAT annotations "
            "in YOLO segmentation format into dataset/ first."
        )

    model = YOLO(BASE_MODEL)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=20,
        project="runs",
        name="answer_sheet_seg",
    )

    # Best weights land at runs/answer_sheet_seg/weights/best.pt —
    # point segment.py's MODEL_PATH at that file once training is done.


if __name__ == "__main__":
    main()
