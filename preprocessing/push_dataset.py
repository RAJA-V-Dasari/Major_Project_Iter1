"""
Replace the Hugging Face dataset with the normalised page images.

DESTRUCTIVE. `--recreate` deletes the repo and makes a fresh one, which
purges git history - the raw scans currently in that repo become
unrecoverable from Hugging Face. They survive only in
preprocessing/.hf_cache on this machine, so back that up before running
this.

Order of operations is deliberate: the local archive is verified BEFORE
anything remote is touched, so the run aborts rather than deleting the
remote copy of files we cannot reproduce.

Run:
    python push_dataset.py --dry-run     # show what would happen
    python push_dataset.py --recreate    # delete repo, recreate, upload
    python push_dataset.py               # upload into existing repo
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "prss-majorproject-37/Handwritten-AnswerScripts-MajorProject"
REPO_TYPE = "dataset"

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / ".hf_cache"

CSV_PATH = BASE_DIR / "students.csv"

# Working files that live under output/ but are not part of the dataset.
IGNORE_PATTERNS = [
    "_identity_sheets/*",
    "_identity_counts.csv",
    ".rotated.json",
]

README = """---
license: other
pretty_name: Handwritten Answer Scripts (Normalised)
tags:
  - document-layout-analysis
  - handwriting
  - ocr
---

# Handwritten Answer Scripts — normalised

Scanned CIE answer booklets, converted to a single consistent layout.

**Private dataset. Contains personal data** — student names, USNs,
signatures and marks appear on the cover sheets. Do not redistribute or
make public.

## Structure

```
student_<NN>/
    cie_<M>/
        page_01.png
        page_02.png
        ...
students.csv
```

- `student_01` … `student_61`
- `cie_1`, `cie_2`, `cie_3` — a folder exists only for an exam the
  student actually sat
- Pages numbered from 1, zero-padded, in reading order

## What normalising involved

The source was inconsistent in four ways, all resolved here:

1. **Mixed formats** — one-PDF-per-page, single multi-page PDFs,
   loose PNGs, and one PDF with no file extension. All are now PNG.
2. **Page order** — scanner output named pages `Scan_x.pdf`,
   `Scan_x (2).pdf` … `(10)`, which sorts wrongly two ways: `(10)`
   before `(2)`, and the unnumbered file (page 1) last. Re-ordered.
3. **Orientation** — every source scan was stored flipped
   (upside-down AND mirrored, i.e. a vertical flip, not a rotation).
   Corrected.
4. **Resolution** — PDFs wrapped the same 1700x2338 scan with differing
   page geometry, so a fixed-DPI render would have upscaled some. The
   embedded image is extracted instead: native pixels, no resampling.

All pages are 1700x2338 (A4 at ~200 DPI); a small number are 1700x2339
from per-page rounding.

## students.csv

`student_id, usn, name, cie_1_pages, cie_2_pages, cie_3_pages,
total_pages, cies_sat, review_flag`

Names and USNs were transcribed by reading the handwritten cover sheets.
`review_flag` marks entries where a digit was ambiguous, or where the
USN does not match the `1BM23CS###` cohort pattern — those are worth
checking before the field is relied on.

## Known gaps

- `student_18` … `student_61` sat only some CIEs; missing folders are
  genuine (the exam was not written), not lost data.
"""


def collect_files():
    """Every page that will be uploaded."""

    return sorted(OUTPUT_DIR.glob("student_*/cie_*/page_*.png"))


def verify_archive(api, token):
    """
    Confirm the local archive still holds every file currently in the
    remote repo, so nothing unreproducible is destroyed.

    Returns (ok, message).
    """

    try:
        info = api.repo_info(
            REPO_ID, repo_type=REPO_TYPE, files_metadata=True, token=token
        )
    except Exception as exc:
        return False, f"could not read remote repo: {exc}"

    missing = []

    for sibling in info.siblings:

        if sibling.rfilename in (".gitattributes", "README.md"):
            continue

        if not (CACHE_DIR / sibling.rfilename).exists():
            missing.append(sibling.rfilename)

    if missing:
        return False, (
            f"{len(missing)} remote file(s) are NOT in the local archive "
            f"at {CACHE_DIR}. Refusing to delete. First few: "
            f"{missing[:5]}"
        )

    return True, f"all {len(info.siblings)} remote files present locally"


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    if not token:
        sys.exit("HF_TOKEN not set. Run:  set -a; . ../.env; set +a")

    if not OUTPUT_DIR.exists():
        sys.exit(f"{OUTPUT_DIR} not found - run convert_dataset.py first")

    api = HfApi(token=token)

    pages = collect_files()

    if not pages:
        sys.exit("no pages found to upload")

    students = len({p.parents[1].name for p in pages})

    size = sum(p.stat().st_size for p in pages) / 1e9

    print(f"Pages    : {len(pages)}")
    print(f"Students : {students}")
    print(f"Size     : {size:.2f} GB")
    print(f"Target   : {REPO_ID}")
    print(f"Mode     : {'RECREATE (destructive)' if args.recreate else 'upload into existing'}")
    print()

    ok, message = verify_archive(api, token)

    print(f"Archive check: {message}")

    if not ok:
        sys.exit("ABORTED - local archive incomplete.")

    if args.dry_run:
        print("\nDry run - nothing was changed.")
        return

    if args.recreate:

        print("\nDeleting repo (purges history) ...")

        api.delete_repo(REPO_ID, repo_type=REPO_TYPE, missing_ok=True)

        print("Creating fresh repo ...")

        api.create_repo(
            REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True
        )

    readme = OUTPUT_DIR / "README.md"

    readme.write_text(README)

    if CSV_PATH.exists():
        import shutil

        shutil.copy(CSV_PATH, OUTPUT_DIR / "students.csv")

    print("\nUploading ...")

    api.upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(OUTPUT_DIR),
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Normalised dataset: consistent naming, orientation and format",
    )

    print("\nDone.")
    print(f"https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
