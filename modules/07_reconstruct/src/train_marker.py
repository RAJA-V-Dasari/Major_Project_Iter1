"""
Train the marker reader: a small CNN over a closed label set.

    07_reconstruct/markers/crops/ + manifest.csv
        -> 07_reconstruct/markers/marker_cnn.pt
        -> 07_reconstruct/markers/confusion.png

WHY A CLASSIFIER AND NOT A FINE-TUNED TrOCR
-------------------------------------------
TrOCR is a sequence model: an encoder-decoder that emits arbitrary
text. That is the right shape for a line of prose and the wrong shape
here, for two reasons.

The label set is CLOSED. `question_schema` fixes it - eight top-level
questions, ten roman and a few letter sub-parts, plus a reject class
for the stars, ticks and `Ans:-` that share the margin. Twenty-odd
outcomes, known in advance. Generating free text and then checking
whether it happens to land in that set throws away the constraint
instead of using it.

And there is no GPU on this machine. Fine-tuning trocr-base (334M
parameters) on CPU is hours per epoch; this model is ~400k parameters
and trains in minutes, which is the difference between an experiment
that can be iterated on and one that can be run once.

If accuracy stalls, the labelled crops transfer to a TrOCR fine-tune
unchanged - nothing here forecloses that.

WHY IT IS SPLIT BY STUDENT
--------------------------
Same reason `annotation/` does it: one student's markers share a
hand. Splitting by crop would put near-identical marks either side of
the train/test line and report a score the model has not earned. The
split is on student id, and it is fixed before any measurement.

Run:
    python train_marker.py                 # train and score
    python train_marker.py --epochs 40
    python train_marker.py --smoke         # 2 epochs, proves plumbing
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

import question_schema as Q   # noqa: E402

MARKER_DIR = STAGE_DIR / "markers"
CROP_DIR = MARKER_DIR / "crops"
MANIFEST = MARKER_DIR / "manifest.csv"
WEIGHTS = MARKER_DIR / "marker_cnn.pt"

# Every crop is squashed to this before it reaches the net. Small on
# purpose: a marker is 1-3 glyphs, and the information is in the
# stroke layout, not in fine texture.
SIZE = 48

REJECT = "_reject"

MIN_EXAMPLES = 8

SEED = 11
VAL_FRACTION = 0.25


def load_rows():

    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found - run extract_markers.py first")

    with open(MANIFEST, encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["label"].strip()]

    if not rows:
        raise SystemExit("no labelled rows in the manifest")

    return rows


def load_image(identifier):

    path = CROP_DIR / f"{identifier}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    # pad to square before resizing, so a wide "2a" and a narrow "1"
    # keep their true aspect instead of both being stretched to fill
    h, w = image.shape
    side = max(h, w)
    square = np.full((side, side), 255, np.uint8)
    square[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = image

    return cv2.resize(square, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def augment(image, rng):
    """Small affine jitter. Real variation here is slant and stroke
    weight, not flips - a mirrored '2a' is not a thing that exists."""

    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.88, 1.12)
    tx, ty = rng.uniform(-2, 2), rng.uniform(-2, 2)

    matrix = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    return cv2.warpAffine(image, matrix, (SIZE, SIZE),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def build(rows):
    """(images, labels, students) as arrays, dropping unreadable crops."""

    images, labels, students = [], [], []

    for row in rows:
        image = load_image(row["id"])
        if image is None:
            continue
        images.append(image)
        labels.append(row["label"].strip())
        students.append(int(row["student"]))

    return images, labels, students


def split(students, rng):
    """Held-out students, not held-out crops. See the module note."""

    unique = sorted(set(students))
    rng.shuffle(unique)

    cut = max(1, int(len(unique) * VAL_FRACTION))

    return set(unique[:cut])


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--binary", action="store_true",
                        help="marker vs reject only")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 2

    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    rows = load_rows()
    images, labels, students = build(rows)

    if args.binary:
        # The immediate question is "cut here or not", which does not
        # need the question's identity. Reported separately from the
        # multi-class score because they fail differently: confusing 3a
        # with 3b costs a label, confusing a marker with a star costs a
        # chunk boundary.
        labels = [REJECT if l == REJECT else "_marker" for l in labels]

    counts = Counter(labels)

    # A class seen a handful of times cannot be learned OR scored - with
    # a student-level split it may not even appear on both sides of the
    # line. Dropped rather than left in to inflate the denominator.
    rare = {c for c, n in counts.items() if n < MIN_EXAMPLES}

    if rare:
        keep = [i for i, l in enumerate(labels) if l not in rare]
        print(f"Dropping {len(labels) - len(keep)} crop(s) in "
              f"{len(rare)} class(es) under {MIN_EXAMPLES} examples: "
              f"{' '.join(sorted(rare))}\n")
        images = [images[i] for i in keep]
        labels = [labels[i] for i in keep]
        students = [students[i] for i in keep]
        counts = Counter(labels)

    classes = sorted(counts)
    index_of = {c: i for i, c in enumerate(classes)}

    print(f"Crops    : {len(images)}")
    print(f"Classes  : {len(classes)}")
    for name in classes:
        print(f"    {name:<10} {counts[name]:>5}"
              + ("   <- too few to learn" if counts[name] < 10 else ""))

    held = split(students, rng)

    train_i = [i for i, s in enumerate(students) if s not in held]
    val_i = [i for i, s in enumerate(students) if s in held]

    print(f"\nSplit    : {len(train_i)} train / {len(val_i)} val "
          f"({len(held)} students held out)")

    if not train_i or not val_i:
        raise SystemExit("split left one side empty - need more students")

    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(64 * (SIZE // 8) ** 2, 128), nn.ReLU(),
        nn.Linear(128, len(classes)),
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def batch_tensor(indices, jitter):
        pixels = []
        for i in indices:
            image = augment(images[i], rng) if jitter else images[i]
            pixels.append(image.astype(np.float32) / 255.0)
        x = torch.from_numpy(np.stack(pixels)).unsqueeze(1)
        y = torch.tensor([index_of[labels[i]] for i in indices])
        return x, y

    best = 0.0

    for epoch in range(1, args.epochs + 1):

        model.train()
        rng.shuffle(train_i)

        total = 0.0

        for at in range(0, len(train_i), args.batch):
            window = train_i[at:at + args.batch]
            x, y = batch_tensor(window, jitter=True)
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(window)

        model.eval()
        with torch.no_grad():
            x, y = batch_tensor(val_i, jitter=False)
            predicted = model(x).argmax(1)
            accuracy = (predicted == y).float().mean().item()

        if accuracy > best:
            best = accuracy
            torch.save({"state": model.state_dict(), "classes": classes,
                        "size": SIZE}, WEIGHTS)

        print(f"  epoch {epoch:>3}  loss {total / len(train_i):.4f}  "
              f"val {accuracy * 100:.1f}%"
              + ("  *" if accuracy >= best else ""), flush=True)

    print(f"\nBest val accuracy : {best * 100:.1f}%")
    print(f"Weights           : {WEIGHTS}")

    # what it confuses, which is the number that decides whether this is
    # usable - an overall score hides a class that never fires
    model.eval()
    with torch.no_grad():
        x, y = batch_tensor(val_i, jitter=False)
        predicted = model(x).argmax(1).numpy()

    confusion = Counter()
    for actual, guess in zip(y.numpy(), predicted):
        if actual != guess:
            confusion[(classes[actual], classes[guess])] += 1

    if confusion:
        print("\nTop confusions (actual -> predicted):")
        for (actual, guess), n in confusion.most_common(10):
            print(f"    {actual:<8} -> {guess:<8} {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
