"""
Reassemble per-page Markdown into one answer per question per student.

    06_evaluation/predictions/<engine>/<page_id>.md
        -> 08_assemble/output/answers.json     {student: {cie: {question: text}}}
        -> 08_assemble/output/answers.csv      flat, one row per answer
        -> 08_assemble/output/coverage.csv     what is missing, per booklet

WHY THIS REPLACES THE MARKER DETECTOR
-------------------------------------
07_reconstruct finds question boundaries geometrically and then has to
work out WHICH question each mark is, which is where it stalled: TrOCR
read 2 of 8 marks correctly, and a CNN over 576 hand-labelled crops
reached 78.4% against a 71.2% majority baseline. Both were reading a
40x40 crop with no context.

The recogniser reads the same mark as part of the page and gets it
right, because "4b)" at the start of a line after "Part - C" is not an
ambiguous glyph. So the question number arrives already attached to its
text, and grouping is string handling rather than pattern recognition.

WHAT A HEADING MEANS
--------------------
The prompt asks for `### 2a)` per question and `#### i)` per sub-part.
A page can open mid-answer, so text before the first heading on a page
continues whatever question was open at the end of the previous page -
pages are processed in order per booklet for exactly that reason.

QUESTION NUMBERS ARE NORMALISED, NOT TRUSTED VERBATIM
-----------------------------------------------------
The model writes the same question a dozen ways: "2a)", "2 a", "Q2a",
"(2a)", "2a.". They are the same question and must land in the same
bucket or the grouping is useless. `question_schema.TOP_LEVEL` is the
authority on what a valid question is; anything that does not
normalise onto it is kept under its raw name and reported, rather than
being dropped or forced onto the nearest match.

Run:
    python assemble.py --engine qwen3b
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES = STAGE_DIR.parent

sys.path.insert(0, str(MODULES / "07_reconstruct" / "src"))

import question_schema as Q   # noqa: E402

PRED_ROOT = MODULES / "06_evaluation" / "predictions"
OUT_DIR = STAGE_DIR / "output"

PAGE_ID = re.compile(r"^s(\d+)_c(\d+)_p(\d+)$")

# "### 2a)" / "## Q2 a." / "#### i)" - the hash count separates a
# question from its sub-part, but the text is what actually decides.
HEADING = re.compile(r"^(#{1,6})\s*(.+?)\s*$")

# Strip the decoration students and models put around a number, so
# "Q2a)", "(2a)", "2 a." and "2a" all reduce to the same thing.
DECORATION = re.compile(r"^[\(\[\s]*(?:q(?:uestion)?\s*)?(.*?)[\)\]\.\:\s]*$",
                        re.I)

ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


def normalise_question(text):
    """A heading -> a canonical question id, or None if it is not one."""

    if not text:
        return None

    core = DECORATION.match(text.strip()).group(1)
    core = re.sub(r"\s+", "", core).lower()

    if not core or len(core) > 4:
        return None

    # "2a" / "2" / "4b"
    match = re.fullmatch(r"([1-9])([a-e])?", core)

    if match:
        number, letter = match.group(1), match.group(2)
        candidate = f"{number}{letter or ''}"
        if candidate in Q.INDEX:
            return candidate
        if number in Q.INDEX:
            return number
        return candidate          # kept, but will be reported as unknown

    return None


def normalise_subpart(text):
    """A heading -> a sub-part id (i, ii, a, b ...), or None."""

    if not text:
        return None

    core = DECORATION.match(text.strip()).group(1)
    core = re.sub(r"\s+", "", core).lower()

    if core in ROMAN or re.fullmatch(r"[a-h]", core):
        return core

    return None


def parse_page(markdown):
    """[(question_or_None, subpart_or_None, text)] in page order.

    A leading section with no heading is returned with question None,
    meaning "whatever was open before this page".
    """

    sections = []

    question = None
    subpart = None
    buffer = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            sections.append((question, subpart, text))
        buffer.clear()

    for line in markdown.splitlines():

        match = HEADING.match(line)

        if match and line.lstrip().startswith("#"):

            label = match.group(2)

            as_question = normalise_question(label)
            as_subpart = normalise_subpart(label)

            if as_question:
                flush()
                question, subpart = as_question, None
                continue

            if as_subpart:
                flush()
                subpart = as_subpart
                continue

            # a heading that is neither - "Part - C", a stray title.
            # Not a boundary; keep it as content so nothing is lost.

        buffer.append(line)

    flush()

    return sections


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()

    engine_dir = PRED_ROOT / args.engine

    if not engine_dir.exists():
        raise SystemExit(f"{engine_dir} not found")

    pages = []

    for path in sorted(engine_dir.glob("*.md")):
        match = PAGE_ID.match(path.stem)
        if not match:
            print(f"  skipping oddly-named {path.name}")
            continue
        student, cie, page = (int(g) for g in match.groups())
        pages.append((student, cie, page, path))

    if not pages:
        raise SystemExit(f"no page markdown in {engine_dir}")

    pages.sort()

    # {(student, cie): {question: [text, ...]}}
    answers = defaultdict(lambda: defaultdict(list))
    unknown = defaultdict(int)
    orphan_pages = []

    open_question = {}

    for student, cie, page, path in pages:

        booklet = (student, cie)
        markdown = path.read_text(encoding="utf-8")

        for question, subpart, text in parse_page(markdown):

            if question is None:
                question = open_question.get(booklet)

                if question is None:
                    # nothing open and no heading - a page whose first
                    # answer the recogniser did not label. Kept under a
                    # sentinel rather than silently discarded.
                    orphan_pages.append(f"s{student:02d}_c{cie}_p{page:02d}")
                    question = "_unlabelled"

            if question not in Q.INDEX and question != "_unlabelled":
                unknown[question] += 1

            if subpart:
                text = f"({subpart}) {text}"

            answers[booklet][question].append(text)
            open_question[booklet] = question

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nested = defaultdict(dict)
    rows = []

    for (student, cie), questions in sorted(answers.items()):
        merged = {q: "\n".join(parts).strip()
                  for q, parts in questions.items()}
        nested[f"student_{student:02d}"][f"cie_{cie}"] = merged
        for question, text in merged.items():
            rows.append({"student": student, "cie": cie,
                         "question": question, "chars": len(text),
                         "text": text})

    with open(OUT_DIR / "answers.json", "w", encoding="utf-8") as handle:
        json.dump(nested, handle, indent=1, ensure_ascii=False)

    with open(OUT_DIR / "answers.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT_DIR / "coverage.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["student", "cie", "found", "missing"])
        for (student, cie), questions in sorted(answers.items()):
            found = [q for q in Q.TOP_LEVEL if q in questions]
            missing = [q for q in Q.TOP_LEVEL if q not in questions]
            writer.writerow([student, cie, " ".join(found),
                             " ".join(missing)])

    booklets = len(answers)
    per_booklet = [len([q for q in v if q in Q.INDEX])
                   for v in answers.values()]

    print(f"Pages      : {len(pages)}")
    print(f"Booklets   : {booklets}")
    print(f"Answers    : {len(rows)}")
    print(f"Recognised questions per booklet: "
          f"{sum(per_booklet) / max(booklets, 1):.1f} of {len(Q.TOP_LEVEL)}")

    if unknown:
        print(f"\nHeadings that are not questions in the schema "
              f"({sum(unknown.values())}):")
        for name, count in sorted(unknown.items(), key=lambda x: -x[1])[:10]:
            print(f"    {name!r}: {count}")

    if orphan_pages:
        print(f"\nPages opening with no question heading "
              f"({len(orphan_pages)}): {' '.join(orphan_pages[:8])}")

    print(f"\nOutput     : {OUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
