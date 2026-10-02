"""
Package the whole corpus for reading on Colab, in resumable batches.

    01_prepare/03_tone/output/  (prepared pages)
        -> 02_read/upload/batch_NN.zip
        -> 02_read/upload/MANIFEST.csv

WHY BATCHES AND NOT ONE ZIP
---------------------------
1,231 content pages is about 330MB, and a free Colab session is
capped at roughly 12 hours and disconnects when idle. At the 7B's
measured rate a full corpus pass does not fit in one sitting, so the
run has to survive being interrupted. Fixed batches make that
concrete: each is a unit of work that either completed or did not,
and the notebook skips pages it has already written.

COVER PAGES NEVER LEAVE THIS MACHINE
------------------------------------
page_01 of every booklet is the identity block - name, USN, signature,
marks. This script refuses to include one, and says so if asked. That
is the whole basis on which sending these pages to a hosted GPU is
acceptable at all, so it is enforced here rather than left to the
caller to remember.

Run:
    python prepare_corpus_batches.py                 # default 250/batch
    python prepare_corpus_batches.py --per-batch 150
    python prepare_corpus_batches.py --students 10   # a trial slice
"""

import argparse
import csv
import re
import zipfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES = STAGE_DIR.parent

CLEAN = MODULES / "01_prepare" / "03_tone" / "output"
OUT_DIR = STAGE_DIR / "upload"

COVER_PAGE = 1

PAGE = re.compile(r"student_(\d+)[\\/]cie_(\d+)[\\/]page_(\d+)\.png$")


def content_pages(limit_students=None):
    """Every page except the covers, as (page_id, path)."""

    found = []

    for path in sorted(CLEAN.glob("student_*/cie_*/page_*.png")):

        match = PAGE.search(str(path))
        if not match:
            continue

        student, cie, page = (int(g) for g in match.groups())

        if page == COVER_PAGE:
            continue                      # identity block - never sent

        if limit_students and student > limit_students:
            continue

        found.append((f"s{student:02d}_c{cie}_p{page:02d}", path,
                      student, cie, page))

    return found


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-batch", type=int, default=250)
    parser.add_argument("--students", type=int)
    args = parser.parse_args()

    if not CLEAN.exists():
        raise SystemExit(f"source not found: {CLEAN}")

    pages = content_pages(args.students)

    if not pages:
        raise SystemExit("no content pages found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.zip"):
        stale.unlink()

    rows = []

    for start in range(0, len(pages), args.per_batch):

        index = start // args.per_batch
        window = pages[start:start + args.per_batch]

        target = OUT_DIR / f"batch_{index:02d}.zip"

        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for page_id, path, *_ in window:
                archive.write(path, f"{page_id}.png")

        size = target.stat().st_size / 1e6

        for page_id, path, student, cie, page in window:
            rows.append({"batch": index, "page_id": page_id,
                         "student": student, "cie": cie, "page": page})

        print(f"  batch_{index:02d}.zip  {len(window):>4} pages  {size:>6.1f} MB")

    with open(OUT_DIR / "MANIFEST.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    students = len({r["student"] for r in rows})
    batches = rows[-1]["batch"] + 1

    print(f"\nPages    : {len(rows)} from {students} student(s)")
    print(f"Batches  : {batches}")
    print(f"Output   : {OUT_DIR}")
    print(f"Manifest : {OUT_DIR / 'MANIFEST.csv'}")
    print("\nNo cover pages included - page_01 is excluded by construction.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
