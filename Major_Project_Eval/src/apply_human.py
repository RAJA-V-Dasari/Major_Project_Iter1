"""
Replay the human tier's decisions onto the marks.

    output/human_marks.jsonl + output/marks/*.json  ->  updated marks

WHY THIS FILE HAS TO EXIST
--------------------------
`grade.py --all` rewrites every `output/marks/*.json` from scratch. That
is correct - it is how the ladder stays reproducible from the rubric and
the handoff alone - but it means every decision made after it ran is
gone the next time it runs.

The model's verdicts survive that, because `verdicts.jsonl` is on disk
and `apply_verdicts.py` puts them back. Until this file existed, human
decisions did not: `serve.py` wrote them to `output/human_marks.jsonl`
and nothing in the repo ever read that log. A person's judgement, made
with the crops in front of them, is the most expensive artifact this
project produces and was the only one with no way back.

So the full re-run is:

    python src/grade.py --all
    python src/apply_verdicts.py output/grade_llm_verdicts.jsonl
    python src/apply_human.py

in that order, cheapest tier to most authoritative.

THE HUMAN WINS
--------------
Unlike `apply_verdicts.py`, this does not skip an item some other tier
has already settled. That asymmetry is the point of the ladder: the
model is checked because it is cheap and fallible, while a person
holding the drawing is the most reliable reading available and the only
tier allowed to overrule another. `serve.py` records what each decision
displaced, and the audit below reports it.

Where the same item was decided more than once, the last line wins - the
log is append-only, so a later decision is a correction of an earlier
one.

Run:
    python src/apply_human.py
    python src/apply_human.py --dry-run
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
from grade import write_summary

HUMAN_LOG = paths.OUT_DIR / "human_marks.jsonl"


def load_decisions(path):
    """{(booklet, question, item): row}, last line winning."""

    decisions, superseded = {}, 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["booklet_id"], row["question"], row["item_index"])
            if key in decisions:
                superseded += 1
            decisions[key] = row
    return decisions, superseded


def apply(decisions, dry_run=False):
    applied, missing = [], []
    displaced = Counter()
    touched = {}

    for path in sorted(paths.MARKS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)

        changed = False
        for question in report["questions"]:
            for index, item in enumerate(question["items"]):
                key = (report["booklet_id"], question["id"], index)
                row = decisions.pop(key, None)
                if row is None:
                    continue

                marks = max(0.0, min(float(row["marks"]),
                                     item["marks_available"]))
                if item["awarded"] is not None and item["tier"] != "human":
                    displaced[item["tier"]] += 1

                why = row.get("note") or "marked by a human, with the drawings"
                if row.get("overrode"):
                    why = f"{why} (overrides the {row['overrode']} tier)"
                item.update(awarded=marks, tier="human", why=why,
                            evidence=[])
                item.pop("quote", None)
                item.pop("llm_declined", None)
                applied.append((key, marks))
                changed = True

        if changed:
            for question in report["questions"]:
                question["marks_settled"] = round(sum(
                    i["awarded"] for i in question["items"]
                    if i["awarded"] is not None), 2)
                question["marks_pending"] = round(sum(
                    i["marks_available"] for i in question["items"]
                    if i["awarded"] is None), 2)
            report["marks_settled"] = round(sum(
                q["marks_settled"] for q in report["questions"]
                if q["counted"]), 2)
            report["marks_pending"] = round(sum(
                q["marks_pending"] for q in report["questions"]
                if q["counted"]), 2)
            touched[path] = report

    # Anything still in `decisions` names an item no marks file has.
    missing = sorted(decisions)

    if not dry_run:
        for path, report in touched.items():
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        if touched:
            write_summary()          # grade.py wrote it before we decided

    return applied, missing, displaced, len(touched)


def render(applied, missing, displaced, superseded):
    out = ["# Human tier audit", ""]
    out.append("Generated by `src/apply_human.py`.")
    out.append("")
    out.append(f"- **{len(applied)} decision(s) applied**")
    if superseded:
        out.append(f"- {superseded} earlier decision(s) superseded by a "
                   "later line for the same item")
    if displaced:
        out.append(f"- {sum(displaced.values())} overruled another tier:")
        for tier, count in displaced.most_common():
            out.append(f"  - `{tier}`: {count}")
    if missing:
        out.append(f"- **{len(missing)} decision(s) matched no item** - see "
                   "below")
        out.append("")
        out.append("## Decisions with nowhere to land")
        out.append("")
        out.append("An item the log names that the current marks do not "
                   "have. This means the rubric changed under a recorded "
                   "judgement, and the mark is not lost but is no longer "
                   "applicable - resolve it by hand.")
        out.append("")
        out.append("| booklet | question | item |")
        out.append("|---|---|---|")
        for booklet, question, index in missing:
            out.append(f"| `{booklet}` | `{question}` | {index} |")
    out.append("")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=str(HUMAN_LOG))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"no human decisions recorded yet: {path}")

    decisions, superseded = load_decisions(path)
    total = len(decisions)
    applied, missing, displaced, files = apply(
        dict(decisions), dry_run=args.dry_run)

    if not args.dry_run:
        out = paths.OUT_DIR / "human_audit.md"
        out.write_text(render(applied, missing, displaced, superseded),
                       encoding="utf-8")

    print(f"{total} decision(s) in the log")
    if superseded:
        print(f"  {superseded} superseded by a later line")
    print(f"  applied  {len(applied)}")
    if displaced:
        print(f"  overruled {sum(displaced.values())} "
              f"({dict(displaced)})")
    if missing:
        print(f"  UNPLACED {len(missing)} - the rubric no longer has "
              "these items")
    if args.dry_run:
        print("\n(dry run - nothing written)")
    else:
        print(f"\n{files} booklet(s) updated")
        print("\nthen re-run: python src/agreement.py")


if __name__ == "__main__":
    main()
