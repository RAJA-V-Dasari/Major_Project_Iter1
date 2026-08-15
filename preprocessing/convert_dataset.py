"""
Normalise the Hugging Face answer-script dataset into a predictable tree:

    output/student_01/cie_1/page_01.png

Student and page numbers are zero-padded to two digits so the names
sort correctly in a shell, a file browser, and any glob - without
padding, page_10 sorts before page_2. CIE numbers are single-digit by
definition (1-3) and are left unpadded.

Source repo (private):
    prss-majorproject-37/Handwritten-AnswerScripts-MajorProject

The source is inconsistent in three ways this script resolves:

  1. Mixed formats. Some booklets are one-PDF-per-page, some are a
     single multi-page PDF, some are already PNG, and one file has no
     extension at all (it is a PDF).

  2. Scanner page ordering. Files land as `Scan_20260803.pdf`,
     `Scan_20260803 (2).pdf`, ... `(10)`, so plain lexicographic
     sorting puts page 10 before page 2, and the un-numbered file
     (which is page 1) last. See sort_key().

  3. Resolution. PDFs wrap the scan with differing page geometry -
     Student 4's pages are 1700x2338 pt (1 pt per px) while Student
     20's are 612x842 pt for the same 1700x2338 px image. A fixed DPI
     would upscale one and not the other, so each page is rendered at
     the zoom that reproduces its own embedded image's resolution.

Run:
    python convert_dataset.py --check      # report only, write nothing
    python convert_dataset.py              # convert students 1..31
    python convert_dataset.py --students 4,20,25
"""

import argparse
import io
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


REPO_ID = "prss-majorproject-37/Handwritten-AnswerScripts-MajorProject"
REPO_TYPE = "dataset"

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / ".hf_cache"

CIE_COUNT = 3

# Source folders map onto output student numbers in three groups. The
# source numbering is not continuous - "Student 1001" is not student
# 1001 - so the mapping is built explicitly rather than parsed.
#
#   Student 1..31    -> student_01..student_31   (19 is absent)
#   Student 1001..18 -> student_32..student_49
#   t 1..12          -> student_50..student_61
#
# Everyone from student_32 onward sat only TWO of the three CIEs, and
# WHICH two varies: some have CIE 1+2, others 2+3, others 1+3. The
# converter therefore emits only the CIE folders that exist in the
# source and does not treat a missing third as an error for them.
SOURCE_GROUPS = [
    ("Student {n}", range(1, 32), 1, True),
    ("Student {n}", range(1001, 1019), 32, False),
    ("t {n}", range(1, 13), 50, False),
]

# Fallback DPI, used only when a PDF page is not a single embedded
# image and must be rasterised instead.
FALLBACK_DPI = 200

# DO NOT "fix" page orientation here.
#
# An earlier version of this script pulled pages out of PDFs with
# extract_image(), which returns the raw stored image bytes and IGNORES
# the page's transformation matrix. Scanner PDFs routinely store the
# image flipped and carry a negative-Y matrix that turns it back the
# right way at render time. Skipping the matrix therefore produced
# mirrored pages from PDFs while PNG sources came through fine - and a
# blanket flip "correcting" that then broke the PNGs instead.
#
# Rendering the page applies the matrix, so pages come out correctly
# oriented with no post-hoc flip, rotate, or per-page orientation
# guessing. If pages ever look flipped again, suspect the extraction
# path before adding a transform.

# Files are fetched in one parallel snapshot rather than one request per
# page. Sequentially this ran at ~4 pages per 45s (~2.5 hours for the
# set); the transfer is latency-bound, not bandwidth-bound, so
# concurrency is what fixes it.
DOWNLOAD_WORKERS = 16

# Student 4 / CIE - 1 holds both PDFs and PNGs. They were checked by
# eye: the two 2026-08-03 PDFs are re-scans of the same first two pages
# that the eight 2026-08-04 PNGs already cover (same booklet number
# 241713, same cover sheet, same "1 b. a host." page). The PNG set is
# the complete booklet, so the PDFs are dropped rather than appended -
# appending them would duplicate pages 1 and 2.
FORMAT_OVERRIDES = {
    ("Student 4", "CIE - 1"): ".png",
}


def sort_key(path):
    """
    Order scanner output the way the pages actually run.

        Scan_x.pdf       -> page 1   (no suffix)
        Scan_x (2).pdf   -> page 2
        Scan_x (10).pdf  -> page 10  (not before page 2)
        document-0000    -> by its number

    Returns a tuple of (stem-without-number, index) so files from
    different scan sessions in one folder stay grouped and in order.
    """

    name = Path(path).name

    stem = Path(name).stem

    # "Scan_20260803 (7)" -> group "Scan_20260803", index 7
    match = re.fullmatch(r"(.*?)\s*\((\d+)\)", stem)

    if match:
        return (match.group(1).strip(), int(match.group(2)))

    # "document-0003" -> group "document", index 3
    match = re.fullmatch(r"(.*?)-(\d+)", stem)

    if match:
        return (match.group(1).strip(), int(match.group(2)))

    # bare "Scan_20260803" is page 1 of its group
    return (stem, 1)


def page_images_from_pdf(pdf_path):
    """
    Yield one PIL image per PDF page, RENDERED at native scan
    resolution.

    Rendering (not extracting) is deliberate - see the orientation note
    above. The zoom is chosen so the output matches the resolution of
    the image actually embedded in the page, which avoids both
    upscaling and throwing detail away:

        Student 4  page 1700x2338 pt, image 1700x2338 px -> zoom 1.0
        Student 20 page  612x842  pt, image 1700x2338 px -> zoom 2.78

    A fixed DPI cannot serve both. Falls back to FALLBACK_DPI when the
    page is not a single full-page image and there is no native size to
    match.
    """

    document = pymupdf.open(pdf_path)

    try:
        for index in range(len(document)):

            page = document.load_page(index)

            images = page.get_images(full=True)

            zoom = None

            if len(images) == 1 and page.rect.width > 0:

                try:
                    native_width = document.extract_image(images[0][0])["width"]
                    zoom = native_width / page.rect.width
                except Exception:
                    zoom = None

            if zoom and zoom > 0:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            else:
                pixmap = page.get_pixmap(dpi=FALLBACK_DPI)

            yield Image.frombytes(
                "RGB", (pixmap.width, pixmap.height), pixmap.samples
            )

    finally:
        document.close()


def is_pdf(path):
    """Detect PDFs by content, since one source file has no extension."""

    with open(path, "rb") as handle:
        return handle.read(5) == b"%PDF-"


def source_map():
    """
    Build {source folder name -> (output student number, expect_all_cies)}
    from SOURCE_GROUPS.
    """

    mapping = {}

    for template, source_range, start, expect_all in SOURCE_GROUPS:

        for offset, number in enumerate(source_range):

            mapping[template.format(n=number)] = (start + offset, expect_all)

    return mapping


def build_index(api):
    """
    Map output student number -> cie folder name -> list of repo paths.
    """

    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)

    mapping = source_map()

    index = defaultdict(lambda: defaultdict(list))

    for path in files:

        parts = path.split("/")

        if len(parts) < 2:
            continue

        entry = mapping.get(parts[0])

        if entry is None:
            continue

        student, _ = entry

        # len(parts) == 2 is a CIE stored as a bare file rather than a
        # folder, e.g. "Student 20/CIE - 1" - the file IS the whole CIE
        index[student][parts[1]].append(path)

    return index


def cie_number(folder_name):
    """'CIE - 2' -> 2."""

    match = re.search(r"(\d+)", folder_name)

    return int(match.group(1)) if match else None


def prefetch(token, repo_paths):
    """
    Download every needed file in one parallel pass.

    Returns the snapshot root; a repo path joined onto it is the local
    file. Already-cached files are skipped, so re-runs are cheap and an
    interrupted run resumes rather than starting over.
    """

    patterns = sorted({p for p in repo_paths})

    root = snapshot_download(
        REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=patterns,
        local_dir=CACHE_DIR,
        token=token,
        max_workers=DOWNLOAD_WORKERS,
    )

    return Path(root)


def convert_cie(api, token, student, folder, repo_paths, dry_run, root=None):
    """
    Convert one CIE folder. Returns (pages_written, notes).
    """

    number = cie_number(folder)

    if number is None or not 1 <= number <= CIE_COUNT:
        return 0, [f"unexpected CIE folder name {folder!r}, skipped"]

    notes = []

    # keyed on the SOURCE folder, taken from the paths themselves, so it
    # cannot drift if output numbering changes
    source_folder = repo_paths[0].split("/")[0] if repo_paths else ""

    override = FORMAT_OVERRIDES.get((source_folder, folder))

    if override:
        kept = [p for p in repo_paths if p.lower().endswith(override)]
        dropped = len(repo_paths) - len(kept)
        if dropped:
            notes.append(
                f"kept {len(kept)} {override} file(s), dropped {dropped} "
                "duplicate(s) per FORMAT_OVERRIDES"
            )
        repo_paths = kept

    ordered = sorted(repo_paths, key=sort_key)

    destination = OUTPUT_DIR / f"student_{student:02d}" / f"cie_{number}"

    if dry_run:
        return len(ordered), notes

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)

    page = 0

    for repo_path in ordered:

        if root is not None:
            local = root / repo_path
        else:
            local = Path(
                hf_hub_download(
                    REPO_ID,
                    repo_path,
                    repo_type=REPO_TYPE,
                    local_dir=CACHE_DIR,
                    token=token,
                )
            )

        if is_pdf(local):

            for image in page_images_from_pdf(local):
                page += 1
                image.save(destination / f"page_{page:02d}.png")

        else:

            with Image.open(local) as image:
                page += 1
                image.convert("RGB").save(
                    destination / f"page_{page:02d}.png"
                )

    return page, notes


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would be done, write nothing",
    )

    parser.add_argument(
        "--students",
        help="comma-separated subset, e.g. 4,20,25",
    )

    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    if not token:
        sys.exit(
            "HF_TOKEN not set. Run:  set -a; . ./.env; set +a"
        )

    api = HfApi(token=token)

    index = build_index(api)

    # which output students are expected to have all three CIEs
    expects_all = {}

    for template, source_range, start, expect_all in SOURCE_GROUPS:
        for offset, _ in enumerate(source_range):
            expects_all[start + offset] = expect_all

    wanted = sorted(expects_all)

    if args.students:
        wanted = [int(s) for s in args.students.split(",") if s.strip()]

    problems = []

    total_pages = 0

    root = None

    if not args.check:

        needed = [
            path
            for student in wanted
            for paths in index.get(student, {}).values()
            for path in paths
        ]

        print(f"Fetching {len(needed)} source file(s) ...")

        root = prefetch(token, needed)

        print(f"Cached at {root}\n")

    for student in wanted:

        folders = index.get(student)

        if not folders:
            problems.append(f"student_{student:02d}: ENTIRE STUDENT MISSING")
            print(f"student_{student:02d}: *** MISSING ***")
            continue

        found = sorted(
            (cie_number(f), f) for f in folders if cie_number(f) is not None
        )

        have = {n for n, _ in found}

        missing = [n for n in range(1, CIE_COUNT + 1) if n not in have]

        # Students from student_32 on sat only two CIEs, so an absent
        # third is expected, not a gap in the data. No folder is created
        # for an exam that was never written.
        if missing and expects_all.get(student, True):
            problems.append(
                f"student_{student:02d}: missing CIE {missing}"
            )

        summary = []

        for number, folder in found:

            pages, notes = convert_cie(
                api, token, student, folder, folders[folder], args.check, root
            )

            total_pages += pages

            summary.append(f"cie_{number}={pages}p")

            for note in notes:
                problems.append(f"student_{student:02d}/cie_{number}: {note}")

        if not missing:
            flag = ""
        elif expects_all.get(student, True):
            flag = f"  MISSING CIE {missing}"
        else:
            flag = f"  (sat {len(have)} of 3)"

        print(f"student_{student:02d}: {'  '.join(summary)}{flag}")

    print()
    print(f"Total pages: {total_pages}")

    if problems:
        print("\n--- ATTENTION ---")
        for problem in problems:
            print(" *", problem)


if __name__ == "__main__":
    main()
