"""
Build training pairs for the cleaning model.

THE LABEL PROBLEM
-----------------
There is no hand-cleaned ground truth for these scans, and making some
by hand is not realistic. Training on the heuristic cleaner's output
would only teach the network to copy that cleaner, including the
artefacts it leaves behind - the binding smudge on
student_05/cie_3/page_05 being the case that started this.

So the pairs are COMPOSITED, which makes the label exact by
construction:

    background : a crop of real paper - real texture, real printed
                 rules, real shadows, real binding smudges, real
                 bleed-through - taken from regions that carry almost
                 no handwriting.

    foreground : a real handwriting stroke mask lifted from a
                 different page.

    input      = background with those strokes darkened onto it
    target     = the stroke mask

The network therefore sees exactly one thing it is asked to keep, and
everything else in the picture is something it must remove. Rules,
smudges, shadows and show-through are all present in the background and
all absent from the target, so "remove them" is learned rather than
hand-coded.

Both halves are real photographs, so there is none of the
synthetic-to-real gap that comes from drawing fake paper.

WHY INTENSITY IS RANDOMISED
---------------------------
Students press very differently, and a fixed threshold is exactly what
fails on the light writers. Each composite darkens its strokes by a
random factor over a wide range, so the network has to infer "this is a
stroke" from shape and local contrast rather than absolute darkness.

Run:
    python make_data.py                 # build the patch set
    python make_data.py --preview       # sample pairs as one image
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR.parent.parent / "modules" / "scan_doc_v2"))

from normalize_page import ink_mask, split_horizontal_strokes  # noqa: E402

SOURCE_DIR = BASE_DIR.parent / "output"

DATA_DIR = BASE_DIR / "data"

PATCH = 256

# How many source pages to mine. More pages beats more patches per page:
# the variation that matters is between writers and between scans.
SOURCE_PAGES = 260

BACKGROUNDS_PER_PAGE = 6
FOREGROUNDS_PER_PAGE = 6

# A crop counts as background if this little of it is handwriting.
# Not zero: a faint trace is realistic and keeps the model from
# assuming backgrounds are perfectly empty.
BACKGROUND_MAX_INK = 0.004

# A crop is usable handwriting if it has at least this much ink -
# otherwise the pair teaches nothing.
FOREGROUND_MIN_INK = 0.02

# Stroke darkening range. 0.35 is a heavy pen, 0.85 barely marks the
# paper; real pages span roughly this.
INTENSITY_RANGE = (0.35, 0.85)

SEED = 11


def page_paths(limit):
    """
    Content pages only.

    Cover pages are excluded deliberately. They carry the printed
    booklet form, and its text is ink, so it lands in the handwriting
    masks and would train the network to preserve printed matter -
    while the same text appearing in a BACKGROUND crop would train it
    to remove printed matter. Mining only content pages removes the
    contradiction instead of trying to arbitrate it.
    """

    paths = [
        p for p in sorted(SOURCE_DIR.glob("student_*/cie_*/page_*.png"))
        if not p.stem.endswith("_01")
    ]

    rng = random.Random(SEED)

    rng.shuffle(paths)

    return paths[:limit]


def handwriting_mask(page_bgr):
    """
    Ink with the printed ruling taken out, horizontal AND vertical.

    split_horizontal_strokes only removes the horizontal rules, so the
    vertical margin rule survived into the first version of the targets
    and would have taught the network to keep it. It is removed here by
    the same logic turned on its side.
    """

    clean, _, _ = split_horizontal_strokes(ink_mask(page_bgr))

    mask = (clean > 0).astype(np.uint8)

    height = mask.shape[0]

    verticals = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 12)),
    )

    return cv2.subtract(mask, verticals)


def crops(image, mask, count, rng, want_ink, min_ink=None, max_ink=None):
    """Random PATCH-sized crops meeting an ink criterion."""

    height, width = mask.shape

    found = []

    for _ in range(count * 40):

        if len(found) >= count:
            break

        if height <= PATCH or width <= PATCH:
            break

        y = rng.randrange(0, height - PATCH)
        x = rng.randrange(0, width - PATCH)

        patch_mask = mask[y:y + PATCH, x:x + PATCH]

        share = float(patch_mask.mean())

        if want_ink and share < min_ink:
            continue

        if not want_ink and share > max_ink:
            continue

        found.append((image[y:y + PATCH, x:x + PATCH],
                      patch_mask.copy()))

    return found


def composite(background, strokes, rng):
    """
    Darken `strokes` onto `background` at a random intensity.

    Multiplicative rather than additive, because ink absorbs light:
    a stroke over a dark shadow stays dark, and a stroke over bright
    paper takes the full intensity. Adding would let strokes wash out
    the very shadows the model needs to learn to ignore.
    """

    intensity = rng.uniform(*INTENSITY_RANGE)

    soft = cv2.GaussianBlur(strokes.astype(np.float32), (0, 0), 0.6)

    soft = np.clip(soft, 0.0, 1.0)

    factor = 1.0 - soft * (1.0 - intensity)

    out = background.astype(np.float32) * factor

    # sensor noise, so the network cannot key on unnaturally clean edges
    out += rng.uniform(0.0, 3.0) * np.random.randn(*out.shape)

    return np.clip(out, 0, 255).astype(np.uint8)


def build(limit, preview):

    rng = random.Random(SEED)

    paths = page_paths(limit)

    backgrounds = []
    foregrounds = []

    print(f"Mining {len(paths)} page(s) for backgrounds and strokes")

    for index, path in enumerate(paths, start=1):

        if index % 40 == 0:
            print(f"  {index}/{len(paths)}", flush=True)

        page = cv2.imread(str(path))

        if page is None:
            continue

        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

        mask = handwriting_mask(page)

        backgrounds += [
            b for b, _ in crops(gray, mask, BACKGROUNDS_PER_PAGE, rng,
                                want_ink=False, max_ink=BACKGROUND_MAX_INK)
        ]

        foregrounds += [
            m for _, m in crops(gray, mask, FOREGROUNDS_PER_PAGE, rng,
                                want_ink=True, min_ink=FOREGROUND_MIN_INK)
        ]

    print(f"\nBackgrounds: {len(backgrounds)}")
    print(f"Stroke sets : {len(foregrounds)}")

    if not backgrounds or not foregrounds:
        sys.exit("not enough material - lower the ink thresholds")

    pairs = min(len(backgrounds), len(foregrounds)) * 2

    inputs = np.zeros((pairs, PATCH, PATCH), np.uint8)
    targets = np.zeros((pairs, PATCH, PATCH), np.uint8)

    for i in range(pairs):

        background = backgrounds[rng.randrange(len(backgrounds))]
        strokes = foregrounds[rng.randrange(len(foregrounds))]

        if rng.random() < 0.5:
            strokes = np.fliplr(strokes).copy()

        inputs[i] = composite(background, strokes, rng)
        targets[i] = strokes * 255

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(DATA_DIR / "patches.npz",
                        inputs=inputs, targets=targets)

    print(f"\nPairs written: {pairs}  -> {DATA_DIR / 'patches.npz'}")

    if preview:

        rows = []

        for i in range(0, 24, 2):
            row = np.hstack([inputs[i], targets[i],
                             inputs[i + 1], targets[i + 1]])
            rows.append(row)

        sheet = np.vstack(rows[:6])

        cv2.imwrite(str(DATA_DIR / "preview.png"), sheet)

        print(f"Preview: {DATA_DIR / 'preview.png'} "
              f"(input, target, input, target)")


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--pages", type=int, default=SOURCE_PAGES)
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()

    build(args.pages, args.preview)


if __name__ == "__main__":
    main()
