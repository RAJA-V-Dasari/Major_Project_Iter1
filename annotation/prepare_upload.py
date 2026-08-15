"""
Package the sampled pages for upload to CVAT.

`images/` holds symlinks into preprocessing/output, and an upload that
does not follow links would send 120 broken files. This dereferences
them into a single zip, which is also the shape CVAT's task creation
form prefers.

Stored without compression on purpose: PNG is already deflate-encoded,
so re-compressing costs minutes of CPU and saves almost nothing.

Run:
    python prepare_upload.py
"""

import csv
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
MANIFEST_PATH = BASE_DIR / "manifest.csv"

OUTPUT_DIR = BASE_DIR / "preannotations"
ZIP_PATH = OUTPUT_DIR / "cvat_images.zip"


def main():

    with open(MANIFEST_PATH) as handle:
        rows = list(csv.DictReader(handle))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = []

    total = 0

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as archive:

        for row in rows:

            source = IMAGE_DIR / row["file_name"]

            if not source.exists():
                missing.append(row["file_name"])
                continue

            # resolve() so the real page is written, not the link
            real = source.resolve()

            archive.write(real, arcname=row["file_name"])

            total += real.stat().st_size

    print(f"Pages   : {len(rows) - len(missing)}")
    print(f"Size    : {total / 1e6:.0f} MB")
    print(f"Archive : {ZIP_PATH}")

    if missing:
        print(f"\n{len(missing)} page(s) missing from images/ - "
              f"re-run sample_pages.py: {missing[:5]}")

    print(
        "\nUpload this in CVAT's task form. Image ORDER inside the task "
        "does not matter: the COCO pre-annotations bind to file_name, "
        "not to position."
    )


if __name__ == "__main__":
    main()
