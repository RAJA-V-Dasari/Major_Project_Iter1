"""
Build the student roster CSV.

Page counts are read from the converted tree (mechanical, exact). Names
and USNs come from transcriptions.py, which was filled in by reading the
handwritten cover sheets.

Run:
    python build_csv.py
"""

import csv
import re
from pathlib import Path

from transcriptions import TRANSCRIPTIONS


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"

CSV_PATH = BASE_DIR / "students.csv"

CIES = (1, 2, 3)


def main():

    if not OUTPUT_DIR.exists():
        raise SystemExit(f"{OUTPUT_DIR} not found - run convert_dataset.py")

    rows = []

    seen = set()

    for path in sorted(OUTPUT_DIR.glob("student_*")):

        match = re.fullmatch(r"student_(\d+)", path.name)

        if not match:
            continue

        number = int(match.group(1))

        seen.add(number)

        counts = {}

        for cie in CIES:

            cie_dir = path / f"cie_{cie}"

            counts[cie] = (
                len(list(cie_dir.glob("page_*.png"))) if cie_dir.exists() else 0
            )

        usn, name, flag = TRANSCRIPTIONS.get(number, ("", "", "NOT TRANSCRIBED"))

        rows.append(
            {
                "student_id": path.name,
                "usn": usn,
                "name": name,
                "cie_1_pages": counts[1],
                "cie_2_pages": counts[2],
                "cie_3_pages": counts[3],
                "total_pages": sum(counts.values()),
                "cies_sat": sum(1 for c in counts.values() if c > 0),
                "review_flag": flag,
            }
        )

    with open(CSV_PATH, "w", newline="") as handle:

        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))

        writer.writeheader()

        writer.writerows(rows)

    missing = [n for n in TRANSCRIPTIONS if n not in seen]

    flagged = [r for r in rows if r["review_flag"]]

    print(f"Students : {len(rows)}")
    print(f"Pages    : {sum(r['total_pages'] for r in rows)}")
    print(f"Saved    : {CSV_PATH}")

    if missing:
        print(f"\nTranscribed but no folder: {missing}")

    if flagged:
        print(f"\n{len(flagged)} row(s) flagged for review:")
        for row in flagged:
            print(
                f"  {row['student_id']}  {row['usn']:<12} "
                f"{row['name']:<26} {row['review_flag']}"
            )


if __name__ == "__main__":
    main()
