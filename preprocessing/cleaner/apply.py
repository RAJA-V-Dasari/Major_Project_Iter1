"""
Run the trained cleaner over the corpus.

Produces crisp black handwriting on white paper: the network decides
per pixel whether it is looking at a pen stroke, so shadows, printed
ruling, show-through and binding smudges all fall away together
regardless of how hard the student pressed.

WHOLE PAGE IN ONE PASS
----------------------
The network is fully convolutional, so despite training on small crops
it can take a whole page at once. That is both faster and better than
tiling:

    tiled, stride 128, 5 threads   11.1 s/page   (221 tiles, ~4x
                                                  redundant compute)
    whole page,        2 threads    7.5 s/page   peak RSS 1.57 GB

and a single pass has no tile seams to blend away at all. The first
version here tiled with a raised-cosine blend to hide those seams;
that machinery is gone because the seams are gone.

Only the padding matters: three pooling levels mean the input has to be
a multiple of 8, so the page is padded by replication and cropped back
afterwards.

GEOMETRY IS UNCHANGED
---------------------
This only rewrites pixels. Deskew, trimming and the canonical size are
clean.py's job and have already been applied, so annotations mapped
onto the cleaned pages stay valid.

Run:
    python apply.py --preview        # a few before/after pairs
    python apply.py                  # whole corpus
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train import UNet


BASE_DIR = Path(__file__).resolve().parent

WEIGHTS_PATH = BASE_DIR / "runs" / "cleaner.pt"

SOURCE_DIR = BASE_DIR.parent / "cleaned"
OUT_DIR = BASE_DIR.parent / "vectorised"

PREVIEW_DIR = BASE_DIR / "preview"

# Three MaxPool2d levels in the U-Net, so each side of the input must
# be divisible by 8 for the skip connections to line up.
POOL_MULTIPLE = 8

# Probability above which a pixel is ink. 0.5 is the natural cut for a
# balanced sigmoid; lower it to keep fainter strokes.
THRESHOLD = 0.5

# Below this many ink pixels a component is speckle. A dot on an i is
# far larger at 200 DPI.
MIN_COMPONENT = 12


@torch.no_grad()
def clean_page(model, gray):
    """Ink probability for a whole page, in a single forward pass."""

    height, width = gray.shape

    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]

    # three pooling levels, so both sides must be a multiple of 8.
    # replicate rather than zero-pad: a black border would look like
    # ink to the network and bleed a dark edge into the result.
    pad_h = (-height) % POOL_MULTIPLE
    pad_w = (-width) % POOL_MULTIPLE

    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

    probability = torch.sigmoid(model(tensor))

    return probability[0, 0, :height, :width].numpy()


def to_page(probability):
    """Probability map -> black ink on white paper, speckle removed."""

    ink = (probability > THRESHOLD).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)

    keep = np.zeros(count, bool)

    for index in range(1, count):
        keep[index] = stats[index, cv2.CC_STAT_AREA] >= MIN_COMPONENT

    ink = keep[labels]

    return np.where(ink, 0, 255).astype(np.uint8)


def load_model():

    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"{WEIGHTS_PATH} not found - run train.py")

    blob = torch.load(WEIGHTS_PATH, map_location="cpu")

    model = UNet(blob.get("width", 8))

    model.load_state_dict(blob["state_dict"])

    model.eval()

    return model


def pages():

    return sorted(SOURCE_DIR.glob("student_*/cie_*/page_*.png"))


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true",
                        help="skip pages already written")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() - 1))

    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    model = load_model()

    if args.preview:

        picks = [
            "student_05/cie_3/page_05.png",   # binding smudge
            "student_57/cie_1/page_08.png",   # faint diagram labels
            "student_02/cie_2/page_10.png",
            "student_23/cie_2/page_05.png",
        ]

        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

        for relative in picks:

            path = SOURCE_DIR / relative

            if not path.exists():
                continue

            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            result = to_page(clean_page(model, gray))

            gap = np.full((gray.shape[0], 14), 90, np.uint8)

            cv2.imwrite(str(PREVIEW_DIR / relative.replace("/", "_")),
                        np.hstack([gray, gap, result]))

            print(f"  {relative}")

        print(f"\nBefore | after: {PREVIEW_DIR}")
        return

    targets = pages()

    if args.limit:
        targets = targets[:args.limit]

    if not targets:
        raise SystemExit(f"no pages under {SOURCE_DIR}")

    print(f"Cleaning {len(targets)} page(s) with the trained model")
    print(f"Threads: {args.threads}\n")

    done = 0

    for index, path in enumerate(targets, start=1):

        if index % 50 == 0:
            print(f"  {index}/{len(targets)}", flush=True)

        out_path = OUT_DIR / path.relative_to(SOURCE_DIR)

        # Whole-page and tiled inference were measured to agree to
        # 0.001% of pixels, so pages written by the earlier tiled run
        # are not worth recomputing.
        if args.resume and out_path.exists():
            done += 1
            continue

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            continue

        result = to_page(clean_page(model, gray))

        out_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_path), result)

    if done:
        print(f"\nSkipped {done} page(s) already written")

    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
