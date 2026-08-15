"""
Fine-tune a YOLO detector on the annotated answer-script layouts.

Detection, not segmentation: the pages were annotated with boxes, and
the downstream consumer routes regions to handlers rather than needing
pixel-accurate outlines.

ABOUT THIS MACHINE
------------------
There is no NVIDIA GPU here (Intel UHD integrated, 6 CPU threads,
7 GB RAM), so training runs on CPU. At the defaults below that is
roughly a few minutes per epoch on ~83 training pages - fine overnight,
painful to iterate on. If you are tuning rather than producing a final
model, either drop --imgsz or move to a GPU box; the same command works
unchanged there and will pick the GPU up automatically.

`--imgsz 1024` is not arbitrary: the pages are 1700x2338, and the
regions that matter least-visibly (a struck word) are only ~40px wide
at full size. Below about 640 they stop being resolvable at all.

Run:
    python train_layout.py --smoke        # 2 epochs, prove the plumbing
    python train_layout.py                # real run
    python train_layout.py --imgsz 640 --epochs 60
"""

import argparse
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA = BASE_DIR / "dataset" / "data.yaml"

# yolo11n = nano. Smallest of the family, and the right starting point
# for a narrow, visually consistent domain with ~100 training images -
# a larger backbone has more capacity to overfit a set this size, not
# more signal to learn from. Move up to yolo11s only if val recall
# plateaus low with train recall high.
BASE_MODEL = "yolo11n.pt"

DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 1024

# Small dataset, small machine. Batch is what usually blows the 7 GB.
DEFAULT_BATCH = 4

PROJECT_DIR = BASE_DIR / "runs"


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--name", default="layout")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 epochs at 320px - checks the pipeline runs, learns nothing",
    )

    args = parser.parse_args()

    data_path = Path(args.data)

    if not data_path.exists():
        sys.exit(
            f"{data_path} not found.\n"
            f"Run:  python build_dataset.py labels/instances_default.json"
        )

    from ultralytics import YOLO

    epochs = 2 if args.smoke else args.epochs
    imgsz = 320 if args.smoke else args.imgsz

    print(f"Model  : {args.model}")
    print(f"Data   : {data_path}")
    print(f"Epochs : {epochs}   imgsz: {imgsz}   batch: {args.batch}")

    if args.smoke:
        print("\nSMOKE RUN - proves the pipeline end-to-end. The resulting\n"
              "weights are worthless; do not evaluate them.\n")

    run_name = args.name + ("_smoke" if args.smoke else "")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=args.batch,
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=True,
        # Geometric augmentation is kept mild and one-sided on purpose:
        #
        #   fliplr=0  a mirrored answer page is not a thing that exists,
        #             and mirroring turns handwriting into a script the
        #             model will never meet. The originals were already
        #             flipped once by the scanner; that was a bug, not a
        #             source of variety.
        #   flipud=0  same, more so.
        #   degrees   real scans are tilted by a couple of degrees, so a
        #             little rotation is honest augmentation.
        fliplr=0.0,
        flipud=0.0,
        degrees=3.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,   # stitches 4 pages into one; destroys page layout,
                      # which is the entire signal here
        erasing=0.0,
        patience=25,
        seed=0,
        plots=True,
    )

    weights = PROJECT_DIR / run_name / "weights" / "best.pt"

    print(f"\nWeights: {weights}")

    if args.smoke:
        print("Smoke weights - discard them.")
    else:
        print(f"Evaluate with:  python evaluate.py --weights {weights}")


if __name__ == "__main__":
    main()
