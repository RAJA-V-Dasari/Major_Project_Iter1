"""
Check the examiner's marks against the arithmetic printed on the covers.

    gold/gold_marks.csv  ->  pass / fail, per row, with the reason

WHAT THIS FILE IS CHECKING, AND WHAT IT IS NOT
----------------------------------------------
Two different questions get confused here easily, so they are kept
apart:

  * **Did we transcribe the cover correctly?** Answered by the
    arithmetic: parts against row totals, row totals against the grand
    total, grand total against the box at the foot of the page. A
    failure here is very likely OUR error, and is reported as a problem.

  * **Did the examiner mark according to the paper?** Answered by
    comparing their Part C totals against the choice policy. A failure
    here is THEIR deviation, and is reported as an observation.

The examiner's grid is a reference, not a ground truth - see
gold/gold_notes.md, where six of the fifty covers carry a demonstrable
error. Treating every disagreement as our fault would mean tuning the
grader to reproduce those mistakes.

WHY THE COVER CHECKS ITSELF
---------------------------
The grid is not just four rows of numbers. Each row carries its own
TOTAL, the grid carries a grand total, and the bottom of the page
repeats that grand total in a separate box written at a different time.
So every cell is covered by two independent sums that the examiner wrote
by hand, and a misread digit almost always breaks one of them.

That is what makes reading fifty of these affordable. Without it, being
sure of ~400 cells means a person checking ~400 cells. With it, the
arithmetic finds the handful that disagree and a person looks only at
those. The ones that reconcile are not "probably right" - they are
right unless a reading error happens to preserve two separate sums.

WHAT IT CANNOT CATCH
--------------------
Two compensating errors in one row (reading 3,5 as 5,3) preserve the
total and pass. So does a row the examiner left blank. This is a filter,
not a proof, and `confidence` on each row records whether a human has
actually looked.

THE CHOICE ROWS ARE NOT SUM-CHECKED
-----------------------------------
Questions 3 and 4 are answered by internal choice, so the pair is worth
the BETTER half - max(3a, 3b), max(4a, 4b) - never the sum. That is the
paper's instruction and what the grader implements.

Where a student answered both halves and the examiner totalled them
anyway, no arithmetic on this page can separate a transcription error
from the examiner's deviation. So those rows are not sum-checked: the
reading is validated by the grand total instead, and the deviation is
reported as an observation about the examiner.

Run:
    python marking/src/gold_check.py
    python marking/src/gold_check.py --explain
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths

QUESTIONS = ["q1", "q2", "q3", "q4"]
COLUMNS = ["a", "b", "c", "d"]

# Questions 3 and 4 are the Part C choice pairs on all three papers.
CHOICE = {"q3", "q4"}


def number(value):
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def question_caps(keys):
    """{cie: {q1: 5, q2: 15, q3: 10, q4: 10}} straight from the rubric."""

    caps = {}
    for cie, key in keys.items():
        per_question = {}
        for spec in key["questions"]:
            # "2a" -> "q2"; the cover grid has one row per question
            # number, with the parts as columns.
            row = f"q{spec['id'][0]}"
            marks = spec["marks"]
            if spec.get("choice_with"):
                # A choice pair contributes its marks once.
                per_question[row] = max(per_question.get(row, 0), marks)
            else:
                per_question[row] = per_question.get(row, 0) + marks
        caps[cie] = per_question
    return caps


def check_row(row, caps):
    """Returns (problems, notes) for one booklet."""

    problems, notes = [], []
    cie = int(row["cie"])
    cap = caps.get(cie, {})

    row_totals = []

    for question in QUESTIONS:
        parts = [number(row[f"{question}{c}"]) for c in COLUMNS]
        written = number(row[f"{question}t"])
        present = [p for p in parts if p is not None]

        if written is None:
            if present:
                problems.append(
                    f"{question}: parts {'+'.join(str(p) for p in present)} "
                    "written but the TOTAL column is blank"
                )
            # No parts and no total is a question the student did not
            # attempt. That is data, not an error.
            row_totals.append(None)
            continue

        row_totals.append(written)
        total = sum(present)
        limit = cap.get(question)

        if question in CHOICE and len(present) > 1:
            # Part C is answered by CHOICE, so the marks for the pair are
            # the better half - not the sum. Where the examiner has
            # totalled both halves anyway, their row total will not equal
            # anything we can derive from the parts, and no arithmetic
            # here can tell a transcription error apart from the
            # examiner's own deviation.
            #
            # So this row is not sum-checked. My reading of it is
            # validated by the grand total instead (which uses the
            # examiner's written row totals), and the deviation from the
            # paper's policy is reported separately, as an observation
            # about the examiner rather than a fault in the data.
            best = max(present)
            if abs(best - written) > 1e-9:
                notes.append(
                    f"{question}: choice pair marked "
                    f"{'+'.join(f'{p:g}' for p in present)}; policy says "
                    f"the better half ({best:g}) but the examiner "
                    f"recorded {written:g}"
                )
        elif abs(total - written) > 1e-9:
            problems.append(
                f"{question}: parts sum to {total:g} but the row total "
                f"reads {written:g}"
            )

        if limit is not None and written - limit > 1e-9:
            problems.append(
                f"{question}: row total {written:g} exceeds the "
                f"{limit:g} marks the question is worth"
            )

    grid = number(row["total_grid"])
    box = number(row["total_box"])
    maximum = number(row["max"])
    counted = [t for t in row_totals if t is not None]

    if grid is None:
        if counted:
            problems.append("the grand total is blank but rows are marked")
    else:
        if abs(sum(counted) - grid) > 1e-9:
            problems.append(
                f"row totals sum to {sum(counted):g} but the grand total "
                f"reads {grid:g}"
            )
        if box is not None and abs(box - grid) > 1e-9:
            problems.append(
                f"grand total {grid:g} disagrees with the Marks Obtained "
                f"box ({box:g})"
            )
        if maximum is not None and grid - maximum > 1e-9:
            problems.append(f"grand total {grid:g} exceeds {maximum:g}")

    return problems, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explain", action="store_true",
                        help="also print the choice-pair observations")
    args = parser.parse_args()

    keys = {}
    for path in sorted(paths.KEYS_DIR.glob("cie*.json")):
        with open(path, encoding="utf-8") as handle:
            key = json.load(handle)
        keys[int(key["cie"])] = key
    caps = question_caps(keys)

    paths.require(paths.GOLD_CSV, "gold/gold_marks.csv")
    with open(paths.GOLD_CSV, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    clean, flagged, accepted, excluded, all_notes = [], [], [], [], []

    for row in rows:
        problems, notes = check_row(row, caps)
        booklet = row["booklet_id"]
        confidence = row.get("confidence", "")
        all_notes.extend((booklet, n) for n in notes)

        # An empty row raises no complaints, which would otherwise let a
        # booklet with no gold marks at all be counted as one that
        # reconciles. It has nothing to reconcile.
        if confidence == "no_gold":
            excluded.append(booklet)
        elif not problems:
            clean.append(booklet)
        elif confidence.startswith("accepted") or confidence == "no_gold":
            # A human has looked at this one, established what the cover
            # actually says, and recorded it in gold_notes.md. The
            # arithmetic still does not close - because the EXAMINER's
            # arithmetic does not close - and that is the finding, not a
            # fault to keep re-reporting.
            accepted.append((booklet, confidence, problems))
        else:
            flagged.append((booklet, confidence, problems))

    for booklet, confidence, problems in flagged:
        marker = "" if confidence == "check" else "   <- NOT yet flagged in the csv"
        print(f"\n{booklet}{marker}")
        for problem in problems:
            print(f"    {problem}")

    if accepted:
        print(f"\nKnown and documented ({len(accepted)}) - see gold_notes.md:")
        for booklet, confidence, problems in accepted:
            print(f"  {booklet}  [{confidence}]")
            for problem in problems:
                print(f"      {problem}")

    if all_notes:
        print(f"\n\nExaminer deviations from the choice policy "
              f"({len(all_notes)}):")
        print("Part C is answered by choice, so the pair is worth the")
        print("better half. Where the examiner totalled both halves")
        print("instead, our grader will disagree with them BY DESIGN.")
        print("These are the examiner's deviations, not our errors, and")
        print("agreement is reported with and without them.")
        for booklet, note in all_notes:
            print(f"  {booklet}  {note}")

    if excluded:
        print(f"\nNo gold marks, excluded from agreement ({len(excluded)}):")
        for booklet in excluded:
            print(f"  {booklet}")

    print(
        f"\n\n{len(rows)} booklets: {len(clean)} reconcile, "
        f"{len(accepted)} known and documented, "
        f"{len(excluded)} excluded, "
        f"{len(flagged)} need a human."
    )
    print(f"gold set usable for agreement: {len(clean) + len(accepted)} "
          f"of {len(rows)}")

    # A row marked "check" that now reconciles has been resolved; say so,
    # because a stale flag is how a resolved problem gets re-litigated.
    stale = [r["booklet_id"] for r in rows
             if r.get("confidence") == "check"
             and r["booklet_id"] in clean]
    if stale:
        print(f"marked 'check' but reconciles now: {', '.join(stale)}")

    raise SystemExit(1 if flagged else 0)


if __name__ == "__main__":
    main()
