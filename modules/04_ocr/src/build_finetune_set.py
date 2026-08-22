"""
Turn box-level transcriptions into TrOCR training pairs.

    04_ocr/finetune/numbered/<page_id>.png        rendered by render_numbered
    04_ocr/finetune/numbered/<page_id>.boxes.json the boxes it drew
    04_ocr/finetune/numbered/transcriptions.json  {page_id: {box_index: text}}
        -> 04_ocr/finetune/lines/<page_id>_NN.png
        -> 04_ocr/finetune/pairs.csv

WHY TRANSCRIPTIONS ARE KEYED BY BOX INDEX
-----------------------------------------
Pairing a crop with the wrong string is the one error that cannot be
recovered downstream - it teaches the model to read one line as
another, and nothing later can tell that it happened. Transcribing
against numbered boxes makes the pairing exact by construction rather
than by inference. The earlier version of this file matched whole-page
prose to boxes by order and count, and over four pages the counts
never once agreed (28 vs 19, 9 vs 17, 22 vs 24, 13 vs 29), because a
diagram is boxes with no text and a stacked fraction is two boxes for
one line.

An index with no entry is simply not training data - a box over a
diagram, a smudge, the page number, or a box that swallowed two
written lines (TrOCR is a single-line model, so those are excluded
rather than joined with a space).

THE SPLIT IS BY STUDENT, AND IT IS THE SAME SPLIT AS THE BENCHMARK
------------------------------------------------------------------
A page used for training must never appear in 06_evaluation's
benchmark, or the CER it reports afterwards is not a measurement. Any
page in bench_pages.json is refused outright, loudly.

Run:
    python build_finetune_set.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES = STAGE_DIR.parent

CLEAN = MODULES / "02_segment" / "input"
BENCH = MODULES / "06_evaluation" / "bench_pages.json"

FINETUNE = STAGE_DIR / "finetune"
NUMBERED = FINETUNE / "numbered"
TRANSCRIPTS = NUMBERED / "transcriptions.json"
LINE_DIR = FINETUNE / "lines"
PAIRS = FINETUNE / "pairs.csv"

PAD_PITCH = 0.10
FALLBACK_PITCH = 58.5

VAL_FRACTION = 0.2


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-bench", action="store_true",
                        help="permit benchmark pages (invalidates the CER)")
    args = parser.parse_args()

    if not TRANSCRIPTS.exists():
        raise SystemExit(f"{TRANSCRIPTS} not found")

    transcripts = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))

    bench_keys = set()
    if BENCH.exists():
        for entry in json.loads(BENCH.read_text(encoding="utf-8")):
            bench_keys.add(f"s{entry['student']:02d}_c{entry['cie']}"
                           f"_p{entry['page']:02d}")

    LINE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in LINE_DIR.glob("*.png"):
        stale.unlink()

    rows = []
    leaked = []

    for key, boxes_text in sorted(transcripts.items()):

        if key in bench_keys and not args.allow_bench:
            leaked.append(key)
            continue

        meta_path = NUMBERED / f"{key}.boxes.json"

        if not meta_path.exists():
            print(f"  {key}: no boxes.json, skipped")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        boxes = meta["boxes"]

        gray = cv2.imread(str(CLEAN / meta["page"]), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            print(f"  {key}: unreadable page, skipped")
            continue

        height, width = gray.shape
        pad = int(round(PAD_PITCH * FALLBACK_PITCH))

        student = int(key[1:3])

        written = 0

        for index_text, text in boxes_text.items():

            text = text.strip()
            if not text:
                continue

            index = int(index_text)

            if not (0 <= index < len(boxes)):
                print(f"  {key}: box {index} out of range, skipped")
                continue

            x1, y1, x2, y2 = boxes[index]

            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(width, x2 + pad)
            cy2 = min(height, y2 + pad)

            crop = gray[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            name = f"{key}_{index:02d}.png"
            cv2.imwrite(str(LINE_DIR / name), crop)

            rows.append({"crop": name, "text": text, "page": key,
                         "student": student, "split": ""})
            written += 1

        print(f"  {key}: {written} pair(s)")

    if leaked:
        raise SystemExit(
            f"\nREFUSED: {len(leaked)} page(s) are in the benchmark and "
            f"would invalidate its CER: {' '.join(leaked)}\n"
            f"Transcribe different pages, or pass --allow-bench knowingly.")

    if not rows:
        raise SystemExit("no pairs built")

    students = sorted({r["student"] for r in rows})
    cut = max(1, int(len(students) * VAL_FRACTION)) if len(students) > 1 else 0
    held = set(students[:cut])

    for row in rows:
        row["split"] = "val" if row["student"] in held else "train"

    with open(PAIRS, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    train = sum(1 for r in rows if r["split"] == "train")

    print(f"\nPairs   : {len(rows)} ({train} train / {len(rows) - train} val)")
    print(f"Pages   : {len(transcripts)}   Students: {len(students)}")
    print(f"Crops   : {LINE_DIR}")
    print(f"Pairs   : {PAIRS}")

    if len(rows) < 500:
        print(f"\nNOTE: {len(rows)} pairs is thin for a fine-tune. TrOCR "
              f"adapts to a new hand on order 1,000+ in-domain lines; at "
              f"~18 usable lines a page that is ~55 pages transcribed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
