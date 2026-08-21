"""
Measure the marker reader against real crops before trusting it.

Pulls every geometric marker candidate from a range of students, reads
each with TrOCR, applies marker_text's allow-list, and renders a contact
sheet with the reading and the verdict printed under each crop - so the
question "does this work" is answered by looking, not by a number with
no provenance.

    python validate_markers.py --students 10
    python validate_markers.py --students 10 --limit 60    # quick pass

Writes marker_validation.png and marker_validation.csv into the stage's
check/ directory, which is gitignored - the crops are real student work.
"""

import argparse
import csv
import time
import sys
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(STAGE_DIR.parent / "02_segment" / "src"))

import segment as S           # noqa: E402
import reconstruct as R       # noqa: E402
import marker_text            # noqa: E402
import marker_ocr             # noqa: E402

CHECK_DIR = STAGE_DIR / "check"

CELL_W, CELL_H = 150, 132
LABEL_H = 34
PER_ROW = 12

ACCEPT_COLOUR = (40, 140, 40)
REJECT_COLOUR = (40, 40, 200)


def collect(students):
    """Every geometric candidate, with the crop it came from."""

    items = []

    for student in students:
        for cie in (1, 2, 3):

            pages = R.booklet_pages(student, cie)
            if not pages:
                continue

            margins, _ = R.booklet_margin(pages)

            for number, path in pages:

                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue

                record = R.load_geometry(student, cie, number)
                pitch = record["rule_pitch"] if record else S.FALLBACK_PITCH

                clean, _, _, _ = S.split_rules(S.ink_mask(gray))

                for mark in R.find_markers(clean, margins.get(number), pitch):

                    pad = 6
                    y0 = max(0, mark["top"] - pad)
                    y1 = min(gray.shape[0], mark["bottom"] + pad)
                    x0 = max(0, mark["left"] - pad)
                    x1 = min(gray.shape[1], mark["right"] + pad)

                    crop = gray[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue

                    items.append({
                        "student": student, "cie": cie, "page": number,
                        "top": mark["top"], "crop": crop,
                    })

    return items


def sheet(items):
    """Contact sheet: crop above, reading and verdict below."""

    cells = []

    for item in items:

        cell = np.full((CELL_H + LABEL_H, CELL_W, 3), 255, np.uint8)

        crop = item["crop"]
        h, w = crop.shape
        scale = min((CELL_H - 8) / max(h, 1), (CELL_W - 8) / max(w, 1), 3.0)
        shown = cv2.resize(crop, (max(1, int(w * scale)),
                                  max(1, int(h * scale))))
        sh, sw = shown.shape
        y = (CELL_H - sh) // 2
        x = (CELL_W - sw) // 2
        cell[y:y + sh, x:x + sw] = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR)

        colour = ACCEPT_COLOUR if item["accepted"] else REJECT_COLOUR

        reading = (item["text"] or "").strip()[:14] or "(blank)"
        verdict = f"{'OK ' if item['accepted'] else 'NO '}{reading}"

        cv2.putText(cell, verdict, (4, CELL_H + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
        cv2.putText(cell, f"s{item['student']:02d}c{item['cie']}p{item['page']:02d}",
                    (4, CELL_H + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 120, 120), 1,
                    cv2.LINE_AA)

        cells.append(cell)

    blank = np.full((CELL_H + LABEL_H, CELL_W, 3), 255, np.uint8)

    rows = []
    for start in range(0, len(cells), PER_ROW):
        row = cells[start:start + PER_ROW]
        row += [blank] * (PER_ROW - len(row))
        rows.append(np.hstack(row))

    return np.vstack(rows) if rows else blank


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=int, default=10)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    items = collect(range(1, args.students + 1))

    if args.limit:
        items = items[:args.limit]

    if not items:
        raise SystemExit("no candidates found")

    print(f"Loading {marker_ocr.MODEL_NAME} ...", flush=True)

    started = time.time()

    def progress(done, total):
        rate = (time.time() - started) / max(done, 1)
        left = (total - done) * rate
        print(f"  {done}/{total}  ({rate:.1f}s/crop, ~{left / 60:.1f} min left)",
              flush=True)

    texts = marker_ocr.read_batch([i["crop"] for i in items],
                                  progress=progress)

    for item, text in zip(items, texts):
        accepted, cleaned = marker_text.classify(text)
        item["text"] = text
        item["normalised"] = cleaned
        item["accepted"] = accepted

    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(CHECK_DIR / "marker_validation.png"), sheet(items))

    with open(CHECK_DIR / "marker_validation.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["student", "cie", "page", "top", "text",
                         "normalised", "accepted"])
        for i in items:
            writer.writerow([i["student"], i["cie"], i["page"], i["top"],
                             i["text"], i["normalised"], int(i["accepted"])])

    accepted = sum(1 for i in items if i["accepted"])

    print(f"\nCandidates : {len(items)}")
    print(f"Accepted   : {accepted} ({accepted / len(items) * 100:.0f}%)")
    print(f"Rejected   : {len(items) - accepted}")
    print(f"\nSheet : {CHECK_DIR / 'marker_validation.png'}")
    print(f"Table : {CHECK_DIR / 'marker_validation.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
