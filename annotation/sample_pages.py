"""
Choose which pages to annotate by hand.

Annotating all 1385 pages is not realistic, and it is not necessary -
what a layout model needs is coverage of the *variation*, not volume.
This picks a stratified sample and records the choice in a manifest so
the set is reproducible and auditable.

Three things drive the design:

1. HANDWRITING IS THE DOMINANT VARIABLE. Layout differs far more
   between two students than between two pages by the same student, so
   every student is represented before any student is represented
   twice.

2. THE SPLIT IS BY STUDENT, NOT BY PAGE. Pages from one booklet share
   handwriting, ruling, scan quality and subject matter. Splitting at
   page level would put near-identical pages either side of the
   train/test line and report a score the model has not earned. No
   student appears in more than one split.

3. COVER PAGES ARE CAPPED. Page 1 of every booklet is the same printed
   form (identity block, marks table). It is ~11% of the corpus but
   contributes almost no layout variety, so it is sampled deliberately
   at a fixed small count rather than proportionally.

Run:
    python sample_pages.py                 # write manifest + link images
    python sample_pages.py --size 150
    python sample_pages.py --dry-run
"""

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR.parent / "preprocessing" / "output"

IMAGE_DIR = BASE_DIR / "images"
MANIFEST_PATH = BASE_DIR / "manifest.csv"

# Fixed so the sample is reproducible. Changing it reshuffles the whole
# set, which invalidates any annotation already done against it.
SEED = 20260814

DEFAULT_SIZE = 120

# Cover pages are near-identical across booklets; a handful is enough
# to teach the printed form, more is wasted annotation effort.
COVER_QUOTA = 15

# Roughly 70/15/15, applied to STUDENTS.
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def booklets():
    """(student, cie, [pages]) for every booklet, in stable order."""

    found = []

    for student_dir in sorted(SOURCE_DIR.glob("student_*")):

        match = re.fullmatch(r"student_(\d+)", student_dir.name)

        if not match:
            continue

        student = int(match.group(1))

        for cie_dir in sorted(student_dir.glob("cie_*")):

            cie = int(cie_dir.name.split("_")[1])

            pages = sorted(
                cie_dir.glob("page_*.png"),
                key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
            )

            if pages:
                found.append((student, cie, pages))

    return found


def assign_splits(students, rng):
    """
    Partition students - not pages - between train/val/test.

    Shuffled first so the split does not follow student id, which
    correlates with the source cohort (1-31 sat three CIEs, 32+ sat
    two) and would otherwise put a whole cohort in one split.
    """

    shuffled = list(students)

    rng.shuffle(shuffled)

    total = len(shuffled)

    n_train = round(total * SPLIT_RATIOS["train"])
    n_val = round(total * SPLIT_RATIOS["val"])

    split = {}

    for index, student in enumerate(shuffled):

        if index < n_train:
            split[student] = "train"
        elif index < n_train + n_val:
            split[student] = "val"
        else:
            split[student] = "test"

    return split


def choose_pages(all_booklets, size, rng):
    """
    Pick `size` pages: covers up to quota, then content pages spread
    over students in rounds so coverage comes before depth.
    """

    covers = []
    content = defaultdict(list)

    for student, cie, pages in all_booklets:

        covers.append((student, cie, pages[0]))

        for page in pages[1:]:
            content[student].append((student, cie, page))

    # spread the cover quota over distinct students rather than taking
    # three booklets from the same one
    rng.shuffle(covers)

    picked_covers = []
    seen_students = set()

    for item in covers:

        if len(picked_covers) >= COVER_QUOTA:
            break

        if item[0] in seen_students:
            continue

        picked_covers.append(item)
        seen_students.add(item[0])

    remaining = size - len(picked_covers)

    for pages in content.values():
        rng.shuffle(pages)

    # round-robin: every student contributes their 1st page before any
    # contributes a 2nd
    picked_content = []

    students = sorted(content)

    depth = 0

    while len(picked_content) < remaining:

        added = False

        for student in students:

            if len(picked_content) >= remaining:
                break

            if depth < len(content[student]):
                picked_content.append(content[student][depth])
                added = True

        if not added:
            break

        depth += 1

    return picked_covers, picked_content


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        raise SystemExit(f"{SOURCE_DIR} not found - run convert_dataset.py")

    rng = random.Random(SEED)

    all_booklets = booklets()

    if not all_booklets:
        raise SystemExit(f"no booklets under {SOURCE_DIR}")

    students = sorted({s for s, _, _ in all_booklets})

    split = assign_splits(students, rng)

    picked_covers, picked_content = choose_pages(all_booklets, args.size, rng)

    rows = []

    for kind, picked in (("cover", picked_covers), ("content", picked_content)):

        for student, cie, page in picked:

            page_number = int(re.search(r"(\d+)", page.stem).group(1))

            name = f"s{student:02d}_c{cie}_p{page_number:02d}.png"

            rows.append(
                {
                    "file_name": name,
                    "source": str(page.relative_to(SOURCE_DIR)),
                    "student_id": f"student_{student:02d}",
                    "cie": cie,
                    "page": page_number,
                    "page_kind": kind,
                    "split": split[student],
                }
            )

    rows.sort(key=lambda r: r["file_name"])

    total_pages = sum(len(p) for _, _, p in all_booklets)

    print(f"Corpus    : {total_pages} pages, {len(all_booklets)} booklets, "
          f"{len(students)} students")
    print(f"Sample    : {len(rows)} pages "
          f"({len(picked_covers)} cover, {len(picked_content)} content)")
    print(f"Students  : {len({r['student_id'] for r in rows})} represented")

    print("\nSplit (by student, so no handwriting crosses the line):")

    for name in ("train", "val", "test"):
        pages = [r for r in rows if r["split"] == name]
        studs = {r["student_id"] for r in pages}
        print(f"  {name:<6} {len(studs):>3} students   {len(pages):>3} pages")

    if args.dry_run:
        print("\nDry run - nothing written.")
        return

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for stale in IMAGE_DIR.glob("*.png"):
        stale.unlink()

    for row in rows:

        target = IMAGE_DIR / row["file_name"]

        # symlink, not copy: the images are large and the originals are
        # the single source of truth
        target.symlink_to((SOURCE_DIR / row["source"]).resolve())

    with open(MANIFEST_PATH, "w", newline="") as handle:

        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nImages    : {IMAGE_DIR}")
    print(f"Manifest  : {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
