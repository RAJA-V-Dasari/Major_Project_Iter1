"""
Pull every marker candidate out of the corpus, and label what it can.

    07_reconstruct/input/       (prepared pages)
        -> 07_reconstruct/markers/crops/NNNNN.png
        -> 07_reconstruct/markers/manifest.csv
        -> 07_reconstruct/markers/sheets/*.png   contact sheets to check

WHY POSITION LABELS ITSELF, ON THE NEAT BOOKLETS
------------------------------------------------
Reading a mark is the hard problem; ORDERING marks is free, because we
already have them sorted by page and y. And the paper is fixed:

    1   2a  2b  2c   3a|3b   4a|4b

So on a booklet where the candidate count is plausible and nothing
looks spurious, the k-th candidate IS the k-th question, and no
recogniser is needed to say so. Checked by hand against
student_01/cie_1: six candidates, positionally 1, 2a, 2b, 2c, 3b, 4b -
which is exactly what is written on those pages.

That is the training set. It costs no labelling time, and it is drawn
from precisely the cases a model finds easy - which is the usual
weakness of bootstrapping and is acceptable here, because the messy
cases are what the model is being built to solve and cannot be labelled
cheaply anyway.

WHAT "PLAUSIBLE" MEANS, AND WHY IT IS STRICT
-------------------------------------------
A booklet qualifies only if its candidate count is between MIN_CLEAN
and MAX_CLEAN. Over 10 students the count ranges 3-29, and the high end
is exactly the over-detection that makes labels wrong - student_03/cie_1
has 29 candidates for at most 8 questions, so positional assignment
there would mislabel 21 marks and poison the training set.

Being strict costs coverage and buys correctness, which is the right
trade for training data: a model cannot come out better than what it
was shown.

Run:
    python extract_markers.py                 # whole corpus
    python extract_markers.py --students 10   # a first pass
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(STAGE_DIR.parent / "02_segment" / "src"))

import segment as S           # noqa: E402
import reconstruct as R       # noqa: E402
import question_schema as Q   # noqa: E402

MARKER_DIR = STAGE_DIR / "markers"
CROP_DIR = MARKER_DIR / "crops"
SHEET_DIR = MARKER_DIR / "sheets"
MANIFEST = MARKER_DIR / "manifest.csv"

# A booklet with a candidate count in this range is trusted for
# positional labelling. See the module note.
MIN_CLEAN = 4
MAX_CLEAN = 8

CROP_PAD = 6

CELL_W, CELL_H, LABEL_H = 128, 112, 30
PER_ROW = 14
PER_SHEET = 14 * 12


def booklet_candidates(student, cie):
    """Every geometric candidate in one booklet, in reading order."""

    pages = R.booklet_pages(student, cie)

    if not pages:
        return []

    margins, _ = R.booklet_margin(pages)

    found = []

    for number, path in pages:

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        record = R.load_geometry(student, cie, number)
        pitch = record["rule_pitch"] if record else S.FALLBACK_PITCH

        clean, _, _, _ = S.split_rules(S.ink_mask(gray))

        for mark in R.find_markers(clean, margins.get(number), pitch):

            y0 = max(0, mark["top"] - CROP_PAD)
            y1 = min(gray.shape[0], mark["bottom"] + CROP_PAD)
            x0 = max(0, mark["left"] - CROP_PAD)
            x1 = min(gray.shape[1], mark["right"] + CROP_PAD)

            crop = gray[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            found.append({
                "student": student, "cie": cie, "page": number,
                "top": mark["top"], "left": mark["left"],
                "width": mark["right"] - mark["left"],
                "height": mark["bottom"] - mark["top"],
                "crop": crop,
            })

    return found


def _worker(args):

    student, cie = args

    try:
        found = booklet_candidates(student, cie)
    except Exception as error:                      # noqa: BLE001
        return student, cie, [], str(error)

    return student, cie, found, None


def positional_labels(count):
    """The k-th of `count` candidates is the k-th question - or None if
    the count is not plausible enough to trust."""

    if not (MIN_CLEAN <= count <= MAX_CLEAN):
        return None

    # With fewer candidates than questions, the student answered a
    # subset. Which subset is genuinely ambiguous, so only the
    # unambiguous prefix is claimed: 1, 2a, 2b, 2c are always in the
    # same order, while 3a-vs-3b and 4a-vs-4b are alternatives whose
    # identity position cannot settle.
    labels = []

    for index in range(count):
        if index < 4:
            labels.append(Q.TOP_LEVEL[index])
        else:
            labels.append(None)  # in the 3/4 option zone - unlabelled

    return labels


def sheet(items, start):
    """Contact sheet with the manifest id under each crop."""

    cells = []

    for item in items:

        cell = np.full((CELL_H + LABEL_H, CELL_W, 3), 255, np.uint8)

        crop = item["crop"]
        h, w = crop.shape
        scale = min((CELL_H - 6) / max(h, 1), (CELL_W - 6) / max(w, 1), 3.0)
        shown = cv2.resize(crop, (max(1, int(w * scale)),
                                  max(1, int(h * scale))))
        sh, sw = shown.shape
        cell[(CELL_H - sh) // 2:(CELL_H - sh) // 2 + sh,
             (CELL_W - sw) // 2:(CELL_W - sw) // 2 + sw] = cv2.cvtColor(
                 shown, cv2.COLOR_GRAY2BGR)

        label = item["auto_label"] or "?"
        colour = (40, 130, 40) if item["auto_label"] else (150, 150, 150)

        cv2.putText(cell, f"{item['id']} {label}", (4, CELL_H + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA)
        cv2.putText(cell,
                    f"s{item['student']:02d}c{item['cie']}p{item['page']:02d}",
                    (4, CELL_H + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (130, 130, 130), 1, cv2.LINE_AA)

        cells.append(cell)

    blank = np.full((CELL_H + LABEL_H, CELL_W, 3), 255, np.uint8)

    rows = []
    for at in range(0, len(cells), PER_ROW):
        row = cells[at:at + PER_ROW]
        row += [blank] * (PER_ROW - len(row))
        rows.append(np.hstack(row))

    return np.vstack(rows) if rows else blank


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=int)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    if not R.SOURCE_DIR.exists():
        raise SystemExit(f"source not found: {R.SOURCE_DIR}")

    booklets = R.all_booklets()

    if args.students:
        booklets = [b for b in booklets if b[0] <= args.students]

    print(f"Scanning {len(booklets)} booklet(s) with {args.workers} worker(s)",
          flush=True)

    CROP_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    for stale in CROP_DIR.glob("*.png"):
        stale.unlink()
    for stale in SHEET_DIR.glob("*.png"):
        stale.unlink()

    rows = []
    identifier = 0
    clean_books = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        for done, (student, cie, found, error) in enumerate(
                pool.map(_worker, booklets, chunksize=2), start=1):

            if done % 20 == 0:
                print(f"  {done}/{len(booklets)} booklets, "
                      f"{identifier} candidates", flush=True)

            if error:
                print(f"  student_{student:02d}/cie_{cie}: {error}")
                continue

            labels = positional_labels(len(found))

            if labels:
                clean_books += 1

            for index, item in enumerate(found):
                item["id"] = identifier
                item["auto_label"] = labels[index] if labels else None
                item["booklet_size"] = len(found)
                identifier += 1
                rows.append(item)

    for item in rows:
        cv2.imwrite(str(CROP_DIR / f"{item['id']:05d}.png"), item["crop"])

    with open(MANIFEST, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "student", "cie", "page", "top", "left",
                         "width", "height", "booklet_size", "auto_label",
                         "label"])
        for i in rows:
            writer.writerow([
                f"{i['id']:05d}", i["student"], i["cie"], i["page"],
                i["top"], i["left"], i["width"], i["height"],
                i["booklet_size"], i["auto_label"] or "",
                i["auto_label"] or "",   # `label` starts as the guess,
            ])                           # and is the column a human edits

    for at in range(0, len(rows), PER_SHEET):
        page = at // PER_SHEET
        cv2.imwrite(str(SHEET_DIR / f"sheet_{page:02d}.png"),
                    sheet(rows[at:at + PER_SHEET], at))

    labelled = sum(1 for i in rows if i["auto_label"])

    print(f"\nBooklets      : {len(booklets)}")
    print(f"  trusted     : {clean_books} "
          f"({clean_books / max(len(booklets), 1) * 100:.0f}%, "
          f"{MIN_CLEAN}-{MAX_CLEAN} candidates)")
    print(f"Candidates    : {len(rows)}")
    print(f"  auto-labelled: {labelled} "
          f"({labelled / max(len(rows), 1) * 100:.0f}%)")
    print(f"\nCrops    : {CROP_DIR}")
    print(f"Manifest : {MANIFEST}")
    print(f"Sheets   : {SHEET_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
