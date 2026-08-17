"""
Train the cleaning model: scanned patch -> handwriting mask.

A small U-Net. The task is local (is this pixel part of a stroke?) and
the training set is a few thousand patches, so capacity is deliberately
low - a bigger network would memorise the 240 pages it was mined from
rather than learn the shape of a pen stroke.

WHY A NETWORK AND NOT ANOTHER THRESHOLD
---------------------------------------
Every threshold-based cleaner tried here failed on the same two things,
and both are structural rather than a matter of tuning:

  intensity   students press very differently. A cut that keeps a light
              writer's strokes also keeps shadow and show-through on a
              heavy writer's page.

  smudges     the binding leaves irregular dark streaks INSIDE the page
              (student_05/cie_3/page_05). They are as dark as ink and
              nowhere near an edge, so neither brightness nor position
              separates them. Shape does, which is what a convolutional
              network can use and a threshold cannot.

LOSS
----
BCE plus a soft Dice term. Strokes are ~3-6% of pixels, so BCE alone
drifts toward predicting "background everywhere" - it scores well and
is useless. Dice is computed on the positive class and stays sensitive
when the class is small, so the two together keep thin strokes alive.

Run:
    python train.py                  # ~25 min on 5 CPU threads
    python train.py --epochs 40
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "patches.npz"
RUN_DIR = BASE_DIR / "runs"

WEIGHTS_PATH = RUN_DIR / "cleaner.pt"

VAL_FRACTION = 0.12

# Patches are stored at 256 but trained on a random 128 crop of each.
#
# Measured on this machine (5 CPU threads), per training epoch:
#
#     256px width 16   39.8 min      <- first attempt, infeasible
#     256px width  8   15.9 min
#     128px width 16   11.7 min
#     128px width  8    6.2 min      <- chosen
#
# 128 px still spans four text lines at 200 DPI, which is far more
# context than deciding "is this pixel a pen stroke" needs, and the
# random crop doubles as augmentation.
CROP = 128

BATCH = 8
EPOCHS = 15
LEARNING_RATE = 2e-3

SEED = 7


class Patches(Dataset):

    def __init__(self, inputs, targets, augment):

        self.inputs = inputs
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        image = self.inputs[index].astype(np.float32) / 255.0
        target = self.targets[index].astype(np.float32) / 255.0

        size = image.shape[0]

        if size > CROP:

            # random crop while training, centre crop for validation so
            # the metric is comparable between epochs
            if self.augment:
                y = np.random.randint(0, size - CROP + 1)
                x = np.random.randint(0, size - CROP + 1)
            else:
                y = x = (size - CROP) // 2

            image = image[y:y + CROP, x:x + CROP]
            target = target[y:y + CROP, x:x + CROP]

        if self.augment:

            if np.random.rand() < 0.5:
                image, target = np.fliplr(image).copy(), np.fliplr(target).copy()

            # brightness/contrast jitter on the INPUT only: the label is
            # a shape, and must not move when the exposure does
            image = np.clip(
                image * np.random.uniform(0.85, 1.15)
                + np.random.uniform(-0.06, 0.06),
                0.0, 1.0,
            )

        return (torch.from_numpy(image)[None],
                torch.from_numpy(target)[None])


def block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Three levels. Enough context to see a whole stroke, no more."""

    def __init__(self, width=8):

        super().__init__()

        self.down1 = block(1, width)
        self.down2 = block(width, width * 2)
        self.down3 = block(width * 2, width * 4)

        self.middle = block(width * 4, width * 8)

        self.up3 = block(width * 8 + width * 4, width * 4)
        self.up2 = block(width * 4 + width * 2, width * 2)
        self.up1 = block(width * 2 + width, width)

        self.head = nn.Conv2d(width, 1, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))

        m = self.middle(self.pool(d3))

        u3 = self.up3(torch.cat([F.interpolate(m, scale_factor=2), d3], 1))
        u2 = self.up2(torch.cat([F.interpolate(u3, scale_factor=2), d2], 1))
        u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2), d1], 1))

        return self.head(u1)


def loss_fn(logits, target):

    bce = F.binary_cross_entropy_with_logits(logits, target)

    probability = torch.sigmoid(logits)

    intersection = (probability * target).sum(dim=(1, 2, 3))

    union = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = 1.0 - ((2 * intersection + 1.0) / (union + 1.0)).mean()

    return bce + dice


@torch.no_grad()
def evaluate(model, loader):
    """Mean loss and stroke-F1 on the held-out patches."""

    model.eval()

    losses = []

    true_positive = false_positive = false_negative = 0.0

    for image, target in loader:

        logits = model(image)

        losses.append(float(loss_fn(logits, target)))

        predicted = (torch.sigmoid(logits) > 0.5).float()

        true_positive += float((predicted * target).sum())
        false_positive += float((predicted * (1 - target)).sum())
        false_negative += float(((1 - predicted) * target).sum())

    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)

    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    return float(np.mean(losses)), precision, recall, f1


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--width", type=int, default=8)

    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found - run make_data.py")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    blob = np.load(DATA_PATH)

    inputs, targets = blob["inputs"], blob["targets"]

    count = len(inputs)

    order = np.random.permutation(count)

    split = int(count * (1 - VAL_FRACTION))

    train_index, val_index = order[:split], order[split:]

    train_loader = DataLoader(
        Patches(inputs[train_index], targets[train_index], augment=True),
        batch_size=args.batch, shuffle=True,
    )

    val_loader = DataLoader(
        Patches(inputs[val_index], targets[val_index], augment=False),
        batch_size=args.batch,
    )

    model = UNet(args.width)

    parameters = sum(p.numel() for p in model.parameters())

    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs
    )

    print(f"Patches : {count}  (train {len(train_index)}, "
          f"val {len(val_index)})")
    print(f"Model   : U-Net width {args.width}, {parameters/1000:.0f}k params")
    print(f"Epochs  : {args.epochs}\n")

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    best = 0.0

    for epoch in range(1, args.epochs + 1):

        model.train()

        started = time.time()

        losses = []

        for image, target in train_loader:

            optimiser.zero_grad()

            loss = loss_fn(model(image), target)

            loss.backward()

            optimiser.step()

            losses.append(float(loss))

        schedule.step()

        val_loss, precision, recall, f1 = evaluate(model, val_loader)

        marker = ""

        if f1 > best:
            best = f1
            torch.save({"state_dict": model.state_dict(),
                        "width": args.width}, WEIGHTS_PATH)
            marker = "  <- saved"

        print(f"  epoch {epoch:3d}/{args.epochs}  "
              f"train {np.mean(losses):.4f}  val {val_loss:.4f}  "
              f"P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}  "
              f"({time.time() - started:.0f}s){marker}", flush=True)

    print(f"\nBest stroke F1: {best:.3f}")
    print(f"Weights: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
