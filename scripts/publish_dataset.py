"""
Publish a pipeline stage's output as the cleaned Hugging Face dataset.

    preprocessing/<stage>/output/  ->  prss-majorproject-37/cleaned-handwritten-answerscripts

One script rather than a copy per stage: the publishable stage moves
as the pipeline grows (crop -> tone -> de-rule -> ...), and per-stage
copies would drift. `--stage` selects which one to publish; the
default is the current end of the pipeline.

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
    python publish_dataset.py --dry-run
    python publish_dataset.py                  # publish default stage
    python publish_dataset.py --stage 02_crop  # publish an earlier one
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "prss-majorproject-37/cleaned-handwritten-answerscripts"
REPO_TYPE = "dataset"

PREP_DIR = Path(__file__).resolve().parent
REPO_ROOT = PREP_DIR.parent.parent   # modules/01_prepare/ -> repo root

# The end of the pipeline as it stands.
DEFAULT_STAGE = "03_tone"

# The dataset card is tracked in git at the repo root; the copy inside
# the stage's output/ is generated from it at upload time, so the
# published card and the tracked one cannot drift.
README_PATH = REPO_ROOT / "DATASET.md"

# Working/diagnostic files that live under a stage's output/ but are
# not part of the dataset.
IGNORE_PATTERNS = ["measurements.json", "angles.json"]


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    if not token:
        sys.exit("HF_TOKEN not set. Run:  set -a; . ./.env; set +a")

    output_dir = PREP_DIR / args.stage / "output"

    if not output_dir.exists():
        sys.exit(f"{output_dir} not found - run the {args.stage} stage first")

    if not README_PATH.exists():
        sys.exit(f"{README_PATH} not found - the dataset card is required")

    pages = sorted(output_dir.glob("student_*/cie_*/page_*.png"))

    if not pages:
        sys.exit(f"no pages under {output_dir}")

    students = len({p.parents[1].name for p in pages})
    size = sum(p.stat().st_size for p in pages) / 1e9

    print(f"Stage    : {args.stage}")
    print(f"Pages    : {len(pages)}")
    print(f"Students : {students}")
    print(f"Size     : {size:.2f} GB")
    print(f"Target   : {REPO_ID}  (private)")

    if args.dry_run:
        print("\nDry run - nothing was changed.")
        return

    api = HfApi(token=token)

    print("\nCreating repo if it doesn't exist ...")

    api.create_repo(
        REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True
    )

    (output_dir / "README.md").write_text(README_PATH.read_text())

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
