"""
Build a PII-reduced upload set for a cloud annotation tool.

WHAT IS EXCLUDED AND WHY
------------------------
Every booklet's identity block - name, USN, department, date, the
student's signature, the invigilator's signature and the marks grid -
is on `page_01`. Those 153 cover pages are dropped entirely. What
remains is 1232 answer pages carrying handwriting but no names, no
USNs and no marks.

That is a large reduction, NOT a guarantee. Handwriting is itself
identifying, and a student may have written their USN on a later page.
Spot-check before uploading, and treat the result as "much lower risk",
not "anonymous".

Cover pages still need labelling; do those locally. They are a fixed
printed form, so a handful is enough (the labelling guide caps them at
15) and they are the least informative pages in the corpus.

WHY IT DOWNSCALES
-----------------
1232 pages at full resolution is ~6 GB of PNG, which is slow to upload
and over most free-tier limits. Layout annotation does not need
lossless pixels - it needs to see structure. JPEG at 1400px wide is
~250 MB total and still perfectly legible for deciding prose vs maths
vs diagram.

Coordinates therefore come back in DOWNSCALED space. `scales.json`
records the factor per page, and import_cloud_labels.py uses it to map
boxes back to full resolution. Do not skip that step - boxes that are
30% too small look plausible and will quietly poison training.

Run:
    python export_for_cloud.py
    python export_for_cloud.py --width 1200 --quality 80
"""

import argparse
import json
import re
import zipfile
from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent

CORPUS_DIR = BASE_DIR.parent / "preprocessing" / "output"

OUT_DIR = BASE_DIR / "cloud_upload"

WIDTH = 1400
QUALITY = 85

# Pages per zip. Cloud tools reject very large single uploads, and a
# failed 6 GB upload wastes far more time than four 250 MB ones.
CHUNK = 400

COVER_PAGE = 1


def content_pages():
    """Every page except the identity-bearing cover of each booklet."""

    pages = []

    covers = 0

    for path in sorted(CORPUS_DIR.glob("student_*/cie_*/page_*.png")):

        student = int(re.search(r"student_(\d+)", path.parts[-3]).group(1))
        cie = int(re.search(r"cie_(\d+)", path.parts[-2]).group(1))
        number = int(re.search(r"(\d+)", path.stem).group(1))

        if number == COVER_PAGE:
            covers += 1
            continue

        pages.append((f"s{student:02d}_c{cie}_p{number:02d}.jpg", path))

    return pages, covers


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--quality", type=int, default=QUALITY)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--limit", type=int)

    args = parser.parse_args()

    pages, covers = content_pages()

    if args.limit:
        pages = pages[:args.limit]

    if not pages:
        raise SystemExit(f"no pages under {CORPUS_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for stale in OUT_DIR.glob("*.zip"):
        stale.unlink()

    scales = {}

    staging = OUT_DIR / "_staging"
    staging.mkdir(exist_ok=True)

    print(f"Excluded : {covers} cover page(s) - identity block lives there")
    print(f"Exporting: {len(pages)} content page(s) at {args.width}px\n")

    written = []

    for index, (name, source) in enumerate(pages, start=1):

        if index % 200 == 0:
            print(f"  {index}/{len(pages)}", flush=True)

        with Image.open(source) as image:

            image = image.convert("RGB")

            full_width, full_height = image.size

            scale = args.width / full_width

            small = image.resize(
                (args.width, int(full_height * scale)), Image.LANCZOS
            )

        target = staging / name

        small.save(target, "JPEG", quality=args.quality, optimize=True)

        scales[name] = {
            "source": str(source.relative_to(CORPUS_DIR)),
            "full_width": full_width,
            "full_height": full_height,
            "scaled_width": small.width,
            "scaled_height": small.height,
            # multiply an annotated coordinate by this to get back to
            # full resolution
            "to_full": round(full_width / small.width, 6),
        }

        written.append(target)

    chunks = 0

    total_bytes = 0

    for start in range(0, len(written), args.chunk):

        chunks += 1

        batch = written[start:start + args.chunk]

        archive_path = OUT_DIR / f"content_pages_{chunks:02d}.zip"

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
            for item in batch:
                archive.write(item, arcname=item.name)

        total_bytes += archive_path.stat().st_size

    for item in written:
        item.unlink()

    staging.rmdir()

    with open(OUT_DIR / "scales.json", "w") as handle:
        json.dump(scales, handle, indent=1)

    print(f"\nPages   : {len(written)}")
    print(f"Archives: {chunks} x up to {args.chunk} pages")
    print(f"Size    : {total_bytes / 1e6:.0f} MB total")
    print(f"Output  : {OUT_DIR}")

    print(
        "\nBefore uploading, spot-check a few pages: a student may have "
        "written their USN on a later page. This removes the identity "
        "block, not every possible identifier."
    )

    print(
        "\nWhen labels come back, run import_cloud_labels.py - the boxes "
        "are in downscaled coordinates and must be mapped back."
    )


if __name__ == "__main__":
    main()
