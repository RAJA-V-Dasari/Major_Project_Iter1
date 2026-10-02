"""
Publish the prepared pages as the cleaned Hugging Face dataset.

    data/pages/  ->  prss-majorproject-37/cleaned-handwritten-answerscripts

One script rather than a copy per stage: the publishable stage has moved
as the preparation grew (crop -> tone -> ...), and per-stage copies would
drift. `--stage` selects which one to publish; the default is the end of
`prepare`, the folder every later stage reads.

    pages      data/pages/                  (default: the toned pages)
    02_crop    data/prepare/02_crop/
    01_deskew  data/prepare/01_deskew/

The raw scans live in a SEPARATE repo
(prss-majorproject-37/Handwritten-AnswerScripts-MajorProject) and are
never touched by this script.

Uploading the same paths replaces the previous stage's files in place,
so republishing is how the dataset advances. Every stage so far emits
exactly the same 1384 page paths, so nothing is orphaned; if a future
stage drops or renames pages, stale files would need deleting
separately - upload_folder only adds and overwrites.

Always PRIVATE - cover pages carry real names, USNs, signatures and
marks.

Run:
    python scripts/publish_dataset.py --dry-run
    python scripts/publish_dataset.py                  # publish data/pages
    python scripts/publish_dataset.py --stage 02_crop  # an earlier one
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "common" / "layout.py").exists())))

from common import layout                                  # noqa: E402

REPO_ID = "prss-majorproject-37/cleaned-handwritten-answerscripts"
REPO_TYPE = "dataset"

STAGES = {
    "pages": layout.PAGES,
    "02_crop": layout.PREPARE / "02_crop",
    "01_deskew": layout.PREPARE / "01_deskew",
}

# The dataset card is tracked in git; the copy inside the published
# folder is generated from it at upload time, so the published card and
# the tracked one cannot drift.
README_PATH = layout.REPO / "reading" / "docs" / "DATASET_CARD.md"

# Working/diagnostic files that live in a stage's folder but are not part
# of the dataset.
IGNORE_PATTERNS = ["measurements.json", "angles.json", ".cache/**"]


def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--stage", default="pages", choices=sorted(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = layout.hf_token()

    if not token:
        sys.exit("HF_TOKEN not set - put it in .env at the repo root")

    output_dir = STAGES[args.stage]

    if not output_dir.exists():
        sys.exit(f"{output_dir} not found - run the {args.stage} stage first")

    if not README_PATH.exists():
        sys.exit(f"{README_PATH} not found - the dataset card is required")

    pages = sorted(output_dir.glob("student_*/cie_*/page_*.png"))

    if not pages:
        sys.exit(f"no pages under {output_dir}")

    students = len({p.parents[1].name for p in pages})
    size = sum(p.stat().st_size for p in pages) / 1e9

    print(f"Stage    : {args.stage}  ({output_dir})")
    print(f"Pages    : {len(pages)}")
    print(f"Students : {students}")
    print(f"Size     : {size:.2f} GB")
    print(f"Target   : {REPO_ID}  (private)")

    if args.dry_run:
        print("\nDry run - nothing was changed.")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print("\nCreating repo if it doesn't exist ...")

    api.create_repo(
        REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True
    )

    (output_dir / "README.md").write_text(
        README_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    print("Uploading ...")

    api.upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(output_dir),
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"Publish {args.stage} output",
    )

    print("\nDone.")
    print(f"https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
