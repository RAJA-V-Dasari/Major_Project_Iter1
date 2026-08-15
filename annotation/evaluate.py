"""
Score a trained layout model on a held-out split.

Defaults to `val`. Reporting on `test` requires --test, deliberately:
the test split exists to be looked at once, at the end, after the
model and its thresholds are frozen. Every extra look at it turns a
held-out measurement into a tuning signal, and the number stops meaning
what it claims to mean. The flag is there so choosing to spend it is a
conscious act.

Run:
    python evaluate.py                       # val
    python evaluate.py --test                # spend the test split
    python evaluate.py --weights runs/layout/weights/best.pt
"""

import argparse
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA = BASE_DIR / "dataset" / "data.yaml"
DEFAULT_WEIGHTS = BASE_DIR / "runs" / "layout" / "weights" / "best.pt"

# Classes we already expect to be data-starved. Called out in the
# report so a low score there is read as "too few examples" rather
# than "the model cannot learn this".
SCARCE = ("code", "figure", "crossed_out")


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--test",
        action="store_true",
        help="score on the test split instead of val - see the docstring",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    weights_path = Path(args.weights)

    if not data_path.exists():
        sys.exit(f"{data_path} not found - run build_dataset.py")

    if not weights_path.exists():
        sys.exit(f"{weights_path} not found - run train_layout.py")

    split = "test" if args.test else "val"

    if args.test:
        print("Scoring on TEST. Freeze the model first - anything tuned "
              "after seeing this number is no longer held out.\n")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_path),
        split=split,
        imgsz=args.imgsz,
        plots=True,
        # without these ultralytics writes into ./runs relative to the
        # working directory, which lands in the repo root
        project=str(BASE_DIR / "runs"),
        name=f"eval_{split}",
        exist_ok=True,
    )

    names = metrics.names if hasattr(metrics, "names") else {}

    print(f"\n=== {split} ===")
    print(f"mAP50    : {metrics.box.map50:.3f}")
    print(f"mAP50-95 : {metrics.box.map:.3f}")

    print("\nPer class:")
    print(f"  {'class':<14}{'P':>7}{'R':>8}{'mAP50':>8}   n")

    for index, class_id in enumerate(metrics.box.ap_class_index):

        name = names.get(int(class_id), str(class_id))

        precision = metrics.box.p[index]
        recall = metrics.box.r[index]
        ap50 = metrics.box.ap50[index]

        note = "  (scarce)" if name in SCARCE else ""

        print(f"  {name:<14}{precision:>7.3f}{recall:>8.3f}"
              f"{ap50:>8.3f}{note}")

    print(
        "\nRecall matters more than precision here: a region the model "
        "misses is never routed anywhere downstream, while a region it "
        "over-calls still reaches a handler that can reject it."
    )


if __name__ == "__main__":
    main()
