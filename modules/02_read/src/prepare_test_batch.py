"""
Package a small diagnostic batch: every page that broke, plus controls.

    01_prepare/03_tone/output/  (prepared pages)
        -> 02_read/upload/batch_test.zip
        -> 02_read/upload/TEST_BATCH.csv

WHY A HAND-PICKED BATCH AND NOT A RANDOM SAMPLE
-----------------------------------------------
A random 30 pages of this corpus is ~28 pages that already worked. It
would confirm the prompt still reads ordinary prose and say nothing
about the failures it was rewritten for, most of which appear on one or
two pages each.

So every page here is one whose behaviour is already known and
recorded, and the run either changes it or does not. See
`modules/02_read/BATCH00_REVIEW.md` for what each did.

WHY THE CONTROLS MATTER AS MUCH AS THE FAILURES
-----------------------------------------------
The new prompt tells the model to decline readily. A prompt that
declines everything fixes every failure below and is worse than the one
it replaced, because a page of [?] is unmarkable. The controls are
pages that came back correct and must still come back correct - if
`s08_c2_p04` returns `![table: ...]` instead of its seven readable
rows, the rewrite has overshot.

Run:
    python prepare_test_batch.py
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

# page_id -> (what it did on batch00, what this run has to show)
CASES = {
    # --- invented content where it could not read -------------------
    "s08_c2_p03": ("a graph, prose and 4 legible routing tables came "
                   "back as ![](https://example.com/image.png)",
                   "the 28 rows read, or ![table: ...] - not a URL"),
    "s08_c2_p07": ("Dijkstra working {A,-,0}{B,A,2} became "
                   "| A | A2 | B |, a shape not on the page",
                   "read it, or decline it; do not invent a schema"),
    "s04_c2_p03": ("one line ran to 2447 chars of [?][?][?]...",
                   "no run of repeated [?] inside a line"),

    # --- empty-cell loops that ran to the token ceiling -------------
    "s05_c2_p04": ("| | | repeated 508 times", "no empty rows"),
    "s10_c2_p06": ("| | | repeated 509 times", "no empty rows"),
    "s03_c2_p02": ("| | | repeated 503 times", "no empty rows"),
    "s02_c3_p09": ("|  |  | repeated 305 times", "no empty rows"),
    "s03_c2_p04": ("| ? | ? | repeated 216 times, 5256 chars",
                   "no empty rows"),
    "s05_c3_p07": ("8453 chars of empty wide rows", "no empty rows"),
    "s09_c3_p06": ("| 00000 | v | | | repeated 48 times", "no empty rows"),

    # --- figures the pipeline could not place -----------------------
    "s03_c1_p07": ("two substantial diagrams flattened into label soup "
                   "- web client / Server 1 / Request - with no marker",
                   "one ![diagram: ...] per figure, no label runs"),
    "s01_c1_p04": ("marker sat correctly after '* Eg:' but its box "
                   "pointed at prose at the top of the page",
                   "marker still between '* Eg:' and the page end"),
    "s02_c3_p06": ("![](x1,y1,x2,y2) emitted 114 times",
                   "no literal template text"),
    "s07_c3_p07": ("![](x1,y1,x2,y2)", "no literal template text"),
    "s10_c2_p03": ("![](x1,y1,x2,y2)", "no literal template text"),
    "s03_c3_p08": ("![fig:LAN](0,0,840,600) - invented label and coords",
                   "![diagram: ...] only"),
    "s09_c2_p07": ("one box swallowed a table and clipped the second "
                   "graph", "a marker per figure, in reading order"),

    # --- structure that would corrupt assembly ----------------------
    "s04_c1_p08": ("headings 0,1,2,3,4 - a numbered list inside one "
                   "answer, filed as five questions",
                   "list items, not headings"),
    "s08_c2_p11": ("headings 3,4,5 from a numbered list",
                   "list items, not headings"),
    "s10_c2_p09": ("17 plain lines of subnet arithmetic each wrapped "
                   "in | |", "plain lines, no table"),
    "s07_c2_p07": ("transcribed THE ONE RULE THAT MATTERS from the "
                   "prompt itself", "no prompt text in the output"),

    # --- controls: these were right and must stay right -------------
    "s08_c2_p04": ("CONTROL - a 7-row routing table, read exactly",
                   "the same 7 rows; NOT a ![table: ...] marker"),
    "s10_c2_p09_ctl": ("", ""),          # placeholder, removed below
    "s06_c1_p04": ("CONTROL - clean sequence-number maths in LaTeX",
                   "the same maths, still in $ ... $"),
    "s09_c2_p05": ("CONTROL - hex/binary IP header working",
                   "the same values"),
    "s07_c2_p08": ("CONTROL - Dijkstra working inside bare ``` fences",
                   "the fenced block SURVIVES; it is the student's own"),
    "s07_c2_p09": ("CONTROL - the page really is four graphs and "
                   "nothing else", "markers, no invented prose"),
    "s06_c3_p06": ("CONTROL - timing diagram above 7 lines of prose",
                   "marker for the diagram, prose still transcribed"),
    "s04_c3_p03": ("CONTROL - three network diagrams down the page",
                   "a marker each, prose between them kept"),
    "s02_c1_p02": ("CONTROL - ordinary prose, no figures, no tables",
                   "unchanged; a [?] storm here means over-declining"),
    "s06_c1_p03": ("CONTROL - ordinary prose with a short label run",
                   "prose kept as prose"),
}

CASES.pop("s10_c2_p09_ctl")

PAGE_ID = re.compile(r"^s(\d+)_c(\d+)_p(\d+)$")


def page_path(page_id):
    m = PAGE_ID.match(page_id)
    if not m:
        return None
    s, c, p = (int(g) for g in m.groups())
    if p == COVER_PAGE:
        return None                       # identity block - never sent
    return CLEAN / f"student_{s:02d}" / f"cie_{c}" / f"page_{p:02d}.png"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    if not CLEAN.exists():
        raise SystemExit(f"source not found: {CLEAN}")

    chosen, missing, covers = [], [], []

    for page_id in sorted(CASES):
        path = page_path(page_id)
        if path is None:
            covers.append(page_id)
            continue
        if not path.exists():
            missing.append(page_id)
            continue
        chosen.append((page_id, path))

    if covers:
        raise SystemExit(f"refusing to include cover page(s): {covers}")
    if missing:
        raise SystemExit(f"no such page(s): {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / "batch_test.zip"

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for page_id, path in chosen:
            archive.write(path, f"{page_id}.png")

    csv_path = OUT_DIR / "TEST_BATCH.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["page_id", "was", "must_now"])
        for page_id, _ in chosen:
            writer.writerow([page_id, *CASES[page_id]])

    controls = [p for p, _ in chosen if CASES[p][0].startswith("CONTROL")]

    print(f"batch_test.zip  {len(chosen)} pages  "
          f"{target.stat().st_size / 1e6:.1f} MB")
    print(f"  {len(chosen) - len(controls)} known failures, "
          f"{len(controls)} controls")
    print(f"  {target}")
    print(f"  {csv_path}")
    print("\nNo cover pages included - page_01 is excluded by construction.")

    print("\nWhat each page has to show:")
    for page_id, _ in chosen:
        was, must = CASES[page_id]
        tag = "CTRL" if was.startswith("CONTROL") else "FAIL"
        print(f"  [{tag}] {page_id}  {must}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
