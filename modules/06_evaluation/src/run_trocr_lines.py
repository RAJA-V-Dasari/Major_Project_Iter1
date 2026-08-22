"""
Produce the line-pipeline's transcription of the benchmark pages.

    06_evaluation/bench_pages.json
        -> 06_evaluation/predictions/trocr_lines/<page_id>.md

This is the incumbent, and the thing an OCR-first approach has to
beat: 02_segment finds the lines, each is cropped, and TrOCR reads
them one at a time. Concatenating the readings in reading order is the
best this architecture can produce - there is no stage after it that
could recover a word the line split in half.

Deliberately generous to the incumbent: it reads segment.py's line
boxes directly rather than the crops on disk, so it is not also being
charged for crop_lines' size floor. Any gap the page-level approach
shows over this is therefore a floor on the real gap, not a ceiling.

Run:
    python run_trocr_lines.py
"""

import json
import sys
import time
from pathlib import Path

import cv2

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES = STAGE_DIR.parent

sys.path.insert(0, str(MODULES / "02_segment" / "src"))
sys.path.insert(0, str(MODULES / "07_reconstruct" / "src"))

import segment as S      # noqa: E402
import marker_ocr        # noqa: E402  (owns the offline-load fix)

CLEAN = MODULES / "02_segment" / "input"

PAGES = STAGE_DIR / "bench_pages.json"
OUT_DIR = STAGE_DIR / "predictions" / "trocr_lines"


def page_lines(path):
    """segment.py's line boxes, in reading order."""

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        return None, []

    record, _, _ = S.segment_page(path)

    if record is None:
        return None, []

    boxes = []

    for block in record["blocks"]:
        for line in block["lines"]:
            boxes.append(line["bbox"])

    boxes.sort(key=lambda b: (b[1], b[0]))

    return gray, boxes


def main():

    entries = json.load(open(PAGES, encoding="utf-8"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    truth_dir = STAGE_DIR / "ground_truth"

    # only pages that have ground truth - reading the rest costs
    # minutes and scores nothing
    entries = [e for e in entries
               if (truth_dir / f"s{e['student']:02d}_c{e['cie']}"
                   f"_p{e['page']:02d}.md").exists()]

    if not entries:
        raise SystemExit("no benchmark page has ground truth yet")

    print(f"Reading {len(entries)} page(s) with TrOCR, line by line\n",
          flush=True)

    started = time.time()

    for entry in entries:

        key = f"s{entry['student']:02d}_c{entry['cie']}_p{entry['page']:02d}"

        gray, boxes = page_lines(CLEAN / entry["path"])

        if gray is None:
            print(f"  {key}: unreadable")
            continue

        crops = [gray[b[1]:b[3], b[0]:b[2]] for b in boxes]
        crops = [c for c in crops if c.size]

        page_started = time.time()

        texts = marker_ocr.read_batch(crops, batch_size=8)

        body = "\n".join(t.strip() for t in texts if t.strip())

        (OUT_DIR / f"{key}.md").write_text(body, encoding="utf-8")

        print(f"  {key}: {len(crops)} lines, "
              f"{time.time() - page_started:.0f}s", flush=True)

    print(f"\nTotal {time.time() - started:.0f}s")
    print(f"Predictions: {OUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
