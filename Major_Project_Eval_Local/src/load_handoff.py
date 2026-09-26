"""
Read part 1's booklets into the shape the grader marks.

    handoff/data/<booklet>/booklet.json  ->  Booklet / Part records

THE CONTRACT, AND THE THREE RULES THAT MATTER
---------------------------------------------
Part 1's README is explicit about how its output must be read, and each
of these rules exists because ignoring it produces a wrong mark rather
than an error:

1. **`answer[]` is the source of truth, not `part.text`.** The `text`
   field is a convenience join that the handoff itself calls
   deliberately redundant. Marking from it would mean marking a string
   whose construction we do not control.

2. **Skip anything with `excluded: true`.** A `crossed` item is work the
   student struck out. It ships so a marker can see what was cancelled;
   scoring it would award marks for work the student explicitly
   withdrew. Use `.get("excluded")`, never `item["excluded"]` - a `gap`
   item has no such key at all.

3. **A `gap` means incomplete, not short.** One page of the corpus was
   lost before it was ever read (`s19_c2_p14`). A part containing a gap
   is missing content that existed; scoring it as a thin answer would
   penalise a student for our lost page.

WHY DIAGRAMS ARE CARRIED, NOT FLATTENED
---------------------------------------
455 of the answer items are image crops - tables, routing diagrams,
waveforms - that part 1 deliberately never transcribed, because
linearising a grid invents a reading order the page does not have. No
tier in this grader reads pixels either. So a diagram is carried as a
path and a page reference, and any rubric item whose evidence is a
drawing goes to the human tier holding the actual crop.

Run:
    python src/load_handoff.py --report
    python src/load_handoff.py --booklet student_01_cie_2 --verbose
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths

BOOKLET_NAME = re.compile(r"^student_(\d+)_cie_(\d+)$")


@dataclass
class Part:
    """One labelled (or unlabelled) part of one question."""

    booklet_id: str
    cie: int
    label: str | None          # "2a" - matches the rubric's question id
    raw_marker: str | None     # what was actually written in the margin
    number: str | None
    part: str | None
    prose: str                 # the answer, crossed-out work removed
    diagrams: list = field(default_factory=list)   # paths to real crops
    pages: list = field(default_factory=list)
    gaps: int = 0
    crossed: int = 0
    crosses_page_break: bool = False

    @property
    def attempted(self):
        return bool(self.prose.strip()) or bool(self.diagrams)

    @property
    def incomplete(self):
        """A lost page means content is missing, not that it is short."""
        return self.gaps > 0


@dataclass
class Booklet:
    booklet_id: str
    student: int
    cie: int
    parts: list = field(default_factory=list)

    @property
    def labelled(self):
        return [p for p in self.parts if p.label]

    @property
    def unlabelled(self):
        """Parts with no question label - these need paper alignment."""
        return [p for p in self.parts if not p.label]

    def by_label(self):
        """{question id: Part}, merging a label that appears twice.

        A student can restate a label mid-answer, and part 1 preserves
        the order they wrote in rather than sorting. Merging keeps all
        of the work under one question instead of letting the second
        fragment silently replace the first.
        """

        merged = {}
        for part in self.labelled:
            existing = merged.get(part.label)
            if existing is None:
                merged[part.label] = part
                continue
            existing.prose = f"{existing.prose}\n{part.prose}".strip()
            existing.diagrams.extend(part.diagrams)
            existing.pages.extend(part.pages)
            existing.gaps += part.gaps
            existing.crossed += part.crossed
        return merged


def load_part(row_id, cie, booklet_dir, question, part):
    prose, diagrams, gaps, crossed = [], [], 0, 0

    for item in part["answer"]:
        kind = item.get("type")

        if kind == "gap":
            # No `excluded` key on a gap - check the type first.
            gaps += 1
            continue

        if item.get("excluded"):
            crossed += 1
            continue

        if kind == "text":
            text = (item.get("text") or "").strip()
            if text:
                prose.append(text)
        elif kind == "diagram":
            image = item.get("image")
            if image:
                diagrams.append({
                    "path": booklet_dir / image,
                    "page_id": item.get("page_id"),
                    "bbox": item.get("bbox"),
                })

    return Part(
        booklet_id=row_id,
        cie=cie,
        label=part.get("label"),
        raw_marker=part.get("raw_marker"),
        number=part.get("number"),
        part=part.get("part"),
        prose=" ".join(prose),
        diagrams=diagrams,
        pages=list(part.get("pages") or []),
        gaps=gaps,
        crossed=crossed,
        crosses_page_break=bool(part.get("crosses_page_break")),
    )


def load_booklet(booklet_id):
    booklet_dir = paths.HANDOFF_DATA / booklet_id
    with open(booklet_dir / "booklet.json", encoding="utf-8") as handle:
        data = json.load(handle)

    match = BOOKLET_NAME.match(booklet_id)
    student, cie = int(match.group(1)), int(match.group(2))

    parts = [
        load_part(booklet_id, cie, booklet_dir, question, part)
        for question in data["questions"]
        for part in question["parts"]
    ]

    return Booklet(booklet_id=booklet_id, student=student, cie=cie,
                   parts=parts)


def load_all():
    paths.require(paths.HANDOFF_INDEX, "handoff index.json")
    with open(paths.HANDOFF_INDEX, encoding="utf-8") as handle:
        index = json.load(handle)
    return index, [load_booklet(r["booklet_id"]) for r in index["booklets"]]


def report(index, booklets):
    """Reproduce the handoff's own totals.

    If our numbers differ from the ones part 1 published, we are reading
    the contract wrong - and every mark downstream inherits that. This
    is the cheapest possible check that the two halves of the project
    agree about what the data says.
    """

    claimed = index["totals"]

    parts = sum(len(b.parts) for b in booklets)
    labelled = sum(len(b.labelled) for b in booklets)
    diagrams = sum(len(p.diagrams) for b in booklets for p in b.parts)
    gaps = sum(p.gaps for b in booklets for p in b.parts)
    crossed = sum(p.crossed for b in booklets for p in b.parts)

    checks = [
        ("booklets", len(booklets), index["booklets_total"]),
        ("parts", parts, claimed["parts"]),
        ("parts labelled", labelled, claimed["parts_labelled"]),
        ("diagrams", diagrams, claimed["diagrams"]),
    ]

    print(f"{'':22} {'ours':>8} {'handoff':>8}")
    ok = True
    for name, ours, theirs in checks:
        flag = "" if ours == theirs else "   <- MISMATCH"
        if ours != theirs:
            ok = False
        print(f"{name:22} {ours:>8} {theirs:>8}{flag}")

    # Not published as a total by part 1, but worth stating: these are
    # the items we deliberately drop and the ones we flag.
    print(f"\n{'crossed-out items skipped':22} {crossed:>8}")
    print(f"{'gap items (lost pages)':22} {gaps:>8}")

    unlabelled = sum(len(b.unlabelled) for b in booklets)
    print(f"{'unlabelled parts':22} {unlabelled:>8}   "
          f"({unlabelled / parts:.0%} - these need paper alignment)")

    attempted = sum(1 for b in booklets for p in b.parts if p.attempted)
    print(f"{'parts with content':22} {attempted:>8}")

    print("\nOK - we read the contract the same way part 1 wrote it."
          if ok else "\nMISMATCH - do not grade until this is understood.")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--booklet")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.booklet:
        booklet = load_booklet(args.booklet)
        print(f"{booklet.booklet_id}  cie {booklet.cie}  "
              f"{len(booklet.parts)} parts "
              f"({len(booklet.unlabelled)} unlabelled)\n")
        for label, part in sorted(booklet.by_label().items()):
            marks = []
            if part.diagrams:
                marks.append(f"{len(part.diagrams)} fig")
            if part.crossed:
                marks.append(f"{part.crossed} crossed")
            if part.gaps:
                marks.append(f"{part.gaps} GAP")
            suffix = f"  [{', '.join(marks)}]" if marks else ""
            print(f"  {label:4} {len(part.prose):>5} chars{suffix}")
            if args.verbose:
                print(f"       {part.prose[:200]}...\n")
        for part in booklet.unlabelled:
            print(f"  {'?':4} {len(part.prose):>5} chars  "
                  f"raw_marker={part.raw_marker!r}")
        return

    index, booklets = load_all()
    if args.report:
        raise SystemExit(0 if report(index, booklets) else 1)

    print(f"loaded {len(booklets)} booklets, "
          f"{sum(len(b.parts) for b in booklets)} parts")


if __name__ == "__main__":
    main()
