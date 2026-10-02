"""
Check the hand-authored rubric against itself and against the scheme.

    keys/cie*.json  ->  pass / fail, with the reason

WHY THIS RUNS BEFORE ANYTHING ELSE
----------------------------------
The rubric is the ceiling on every number this project produces. It is
transcribed by eye from skewed phone scans, and a slip there is silent:
a rubric whose items sum to 9 when the question is worth 10 does not
crash anything, it just quietly loses a mark on every one of the fifty
booklets and lands as a systematic disagreement with the examiner that
looks like a grading problem.

So the arithmetic is checked mechanically and the judgement is checked
by a person. This file is the mechanical half. It refuses to pass on:

  * rubric items that do not sum to the question's marks
  * questions that do not sum to the paper's total, once Part C's
    internal choice is resolved
  * a `choice_with` that is not reciprocated, or points at nothing
  * duplicate question ids
  * a `scheme_page` naming a file that was never rendered
  * an `exact` value too short or too common to be safe (see below)

WHY `exact` IS AUDITED
----------------------
Tier 1 awards a mark whenever an `exact` value appears in the answer,
with no model involved. That is only safe when the value could not
appear by accident. "57088" is safe. "2", "88" and "ok" are not - they
turn up in prose about anything, and an item that fires on them hands
out marks for coincidence. Short numeric values are therefore reported
for a human to confirm rather than silently trusted.

Run:
    python marking/src/validate_keys.py
    python marking/src/validate_keys.py --cie 2 --verbose
    python marking/src/validate_keys.py --no-renders   # no scheme PDFs here

--no-renders skips only the scheme_page check, for a machine that does
not hold the department's PDFs. Everything else still has to pass.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths

TOLERANCE = 1e-6

# An `exact` value shorter than this, and made only of digits, is
# reported. Not an error - "69" is a real answer on CIE-1 4b - but it
# needs a human to have thought about it.
SHORT_NUMERIC = 3


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, where, message):
        self.errors.append(f"{where}: {message}")

    def warn(self, where, message):
        self.warnings.append(f"{where}: {message}")

    @property
    def ok(self):
        return not self.errors


def check_key(key, report, verbose=False, renders=True):
    cie = key["cie"]
    where_key = f"cie{cie}"

    questions = key["questions"]
    by_id = {}

    for spec in questions:
        qid = spec["id"]
        where = f"{where_key} q{qid}"

        if qid in by_id:
            report.error(where, "duplicate question id")
        by_id[qid] = spec

        # --- rubric items sum to the question ---------------------
        total = sum(item["marks"] for item in spec["rubric"])
        if abs(total - spec["marks"]) > TOLERANCE:
            report.error(
                where,
                f"rubric sums to {total} but the question is worth "
                f"{spec['marks']}",
            )

        if not spec["rubric"]:
            report.error(where, "no rubric items")

        # --- the scheme page must exist ---------------------------
        page = paths.SCHEME_PAGES / f"{spec['scheme_page']}.png"
        if renders and not page.exists():
            report.error(
                where, f"scheme_page {spec['scheme_page']} not rendered"
            )

        # --- breakdown provenance ---------------------------------
        source = spec.get("breakdown_source")
        if source not in ("printed", "inferred"):
            report.error(where, f"breakdown_source is {source!r}")

        # --- exact values worth a second look ---------------------
        for item in spec["rubric"]:
            for value in item.get("exact", []):
                stripped = value.strip()
                if not stripped:
                    report.error(where, "empty exact value")
                elif stripped.isdigit() and len(stripped) < SHORT_NUMERIC:
                    report.warn(
                        where,
                        f"exact {value!r} is short and numeric - confirm it "
                        "cannot appear by accident",
                    )

            # A single alphabetic character matches essentially any
            # text. These arrived honestly - "via D", "accumulator R",
            # "node A" - and each one silently inflated keyword coverage
            # on every answer in the corpus. They are an error, not a
            # warning.
            for group in item.get("keywords") or []:
                for alternate in group:
                    stripped = alternate.strip()
                    if len(stripped) == 1 and stripped.isalpha():
                        report.error(
                            where,
                            f"keyword {alternate!r} is a single letter - it "
                            "matches noise, not evidence",
                        )
                    elif stripped.lower() in {"no", "a", "an", "is", "of"}:
                        report.error(
                            where,
                            f"keyword {alternate!r} is a stopword - it "
                            "matches almost any answer",
                        )

            if not item.get("exact") and not item.get("keywords"):
                report.error(where, f"item has neither exact nor keywords: "
                                    f"{item['point'][:50]}")

    # --- choice pairs are reciprocal ------------------------------
    for spec in questions:
        partner_id = spec.get("choice_with")
        if partner_id is None:
            continue
        where = f"{where_key} q{spec['id']}"

        partner = by_id.get(partner_id)
        if partner is None:
            report.error(where, f"choice_with {partner_id!r} does not exist")
            continue
        if partner.get("choice_with") != spec["id"]:
            report.error(
                where,
                f"choice_with {partner_id} is not reciprocated "
                f"(it points at {partner.get('choice_with')!r})",
            )
        if abs(partner["marks"] - spec["marks"]) > TOLERANCE:
            report.error(
                where,
                f"choice pair is uneven: {spec['marks']} vs "
                f"{partner['marks']}",
            )

    # --- the paper adds up ----------------------------------------
    # A choice pair contributes its marks once, not twice: the student
    # answers one of them. This is also what the covers show - the
    # examiner totals the better half, never the sum.
    counted, seen = 0.0, set()
    for spec in questions:
        if spec["id"] in seen:
            continue
        counted += spec["marks"]
        seen.add(spec["id"])
        partner_id = spec.get("choice_with")
        if partner_id:
            seen.add(partner_id)

    if abs(counted - key["max_marks"]) > TOLERANCE:
        report.error(
            where_key,
            f"questions total {counted} once choices are resolved, but the "
            f"paper is {key['max_marks']} marks",
        )

    # --- part totals, where the scheme prints them ----------------
    for part, expected in (key.get("parts") or {}).items():
        actual, seen = 0.0, set()
        for spec in questions:
            if spec["part"] != part or spec["id"] in seen:
                continue
            actual += spec["marks"]
            seen.add(spec["id"])
            if spec.get("choice_with"):
                seen.add(spec["choice_with"])
        if abs(actual - expected) > TOLERANCE:
            report.error(
                where_key,
                f"Part {part} totals {actual}, scheme prints {expected}",
            )

    if verbose:
        print(f"\n{where_key}: {len(questions)} questions, "
              f"{sum(len(q['rubric']) for q in questions)} rubric items")
        for spec in questions:
            total = sum(i["marks"] for i in spec["rubric"])
            choice = f" (choice with {spec['choice_with']})" if spec.get(
                "choice_with") else ""
            printed = "printed" if spec["breakdown_source"] == "printed" \
                else "INFERRED"
            print(f"  {spec['id']:>3}  part {spec['part']}  "
                  f"{spec['marks']:>5} marks  {len(spec['rubric'])} items "
                  f"(sum {total}){choice}  [{printed}]")
            for note in spec.get("notes", []):
                if note.startswith("DEFECT"):
                    print(f"        !! {note[:100]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cie", type=int, choices=[1, 2, 3])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-renders", action="store_true",
                        help="skip the check that each scheme_page is "
                             "rendered (no scheme PDFs on this machine)")
    args = parser.parse_args()

    pattern = f"cie{args.cie}.json" if args.cie else "cie*.json"
    key_files = sorted(paths.KEYS_DIR.glob(pattern))

    if not key_files:
        raise SystemExit(f"no keys matching {pattern} in {paths.KEYS_DIR}")

    report = Report()
    items = 0

    for path in key_files:
        try:
            with open(path, encoding="utf-8") as handle:
                key = json.load(handle)
        except json.JSONDecodeError as exc:
            report.error(path.name, f"invalid JSON - {exc}")
            continue

        items += sum(len(q["rubric"]) for q in key["questions"])
        check_key(key, report, verbose=args.verbose,
                  renders=not args.no_renders)

    print()
    for warning in report.warnings:
        print(f"  warn   {warning}")
    for error in report.errors:
        print(f"  ERROR  {error}")

    print(
        f"\n{len(key_files)} key(s), {items} rubric items, "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    )

    # Say what is still missing, so a partial corpus reads as partial
    # rather than as a pass.
    missing = [c for c in (1, 2, 3)
               if not (paths.KEYS_DIR / f"cie{c}.json").exists()]
    if missing and not args.cie:
        print(f"not yet authored: {', '.join(f'cie{c}' for c in missing)}")

    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
