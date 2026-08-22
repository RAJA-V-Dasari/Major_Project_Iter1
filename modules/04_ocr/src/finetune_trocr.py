"""
Fine-tune TrOCR on this corpus's handwriting.

    04_ocr/finetune/pairs.csv + lines/
        -> 04_ocr/finetune/trocr_tuned/

WHAT THIS IS FOR
----------------
06_evaluation measured the BASE model on real pages and got CER 0.573
with fabrication - "the destination port number is the next" came back
as "Although the situation had not yet yet been". That is a model
reading a hand it has never seen, and it is the gap a fine-tune
closes: trocr-base-handwritten was trained on IAM, which is a
different pen, paper, scanner and script style from a Bangalore exam
booklet at 200 DPI.

MEMORY IS THE BINDING CONSTRAINT, NOT TIME
-------------------------------------------
trocr-base is 334M parameters. A full fine-tune holds weights,
gradients and two Adam moments - roughly 5GB in fp32 - and this
machine has 7.9GB total with about 2.8GB free. So `--freeze-encoder`
is the default: the vision encoder is left alone and only the text
decoder trains. That is also the right call on the merits, because the
encoder's job (turn strokes into features) transfers across hands far
better than the decoder's (turn features into this corpus's
vocabulary), and it is the decoder that is currently inventing words.

If it still will not fit, train on Colab - the checkpoint this writes
loads identically either way.

Run:
    python finetune_trocr.py --smoke        # 2 steps, proves it runs
    python finetune_trocr.py --epochs 8
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

FINETUNE = STAGE_DIR / "finetune"
LINE_DIR = FINETUNE / "lines"
PAIRS = FINETUNE / "pairs.csv"
OUT_DIR = FINETUNE / "trocr_tuned"

MODEL_NAME = "microsoft/trocr-base-handwritten"

SEED = 17
MAX_TARGET = 64


def load_pairs():

    if not PAIRS.exists():
        raise SystemExit(f"{PAIRS} not found - run build_finetune_set.py")

    with open(PAIRS, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("pairs.csv is empty")

    return rows


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--full", dest="freeze_encoder", action="store_false",
                        help="train the encoder too (needs far more RAM)")
    args = parser.parse_args()

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    rows = load_pairs()

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]

    print(f"Pairs   : {len(train)} train / {len(val)} val")

    if args.smoke:
        train = train[:4]
        args.epochs = 1

    print(f"Loading {MODEL_NAME} ...", flush=True)

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f"Encoder frozen; {trainable/1e6:.0f}M trainable parameters")

    model.train()

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def batch_of(rows_):
        images, texts = [], []
        for row in rows_:
            image = cv2.imread(str(LINE_DIR / row["crop"]))
            if image is None:
                continue
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            texts.append(row["text"])
        if not images:
            return None, None
        pixels = processor(images=images, return_tensors="pt").pixel_values
        labels = processor.tokenizer(
            texts, padding="max_length", max_length=MAX_TARGET,
            truncation=True, return_tensors="pt").input_ids
        # padding must not be learned as content
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return pixels, labels

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import time
    started = time.time()

    for epoch in range(1, args.epochs + 1):

        rng.shuffle(train)
        total, steps = 0.0, 0

        for at in range(0, len(train), args.batch):

            pixels, labels = batch_of(train[at:at + args.batch])
            if pixels is None:
                continue

            optimiser.zero_grad()
            loss = model(pixel_values=pixels, labels=labels).loss
            loss.backward()
            optimiser.step()

            total += loss.item()
            steps += 1

            print(f"  epoch {epoch} step {steps} "
                  f"loss {loss.item():.4f} "
                  f"({time.time() - started:.0f}s)", flush=True)

        if steps:
            print(f"epoch {epoch}: mean loss {total / steps:.4f}", flush=True)

    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)

    print(f"\nSaved: {OUT_DIR}")
    print(f"Total: {time.time() - started:.0f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
