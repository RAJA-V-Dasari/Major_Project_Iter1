"""
Apply the model tier's verdicts, and reject the ones it cannot justify.

    verdicts.jsonl + output/marks/*.json  ->  updated marks
                                          ->  output/verdict_audit.md

THE CHECK THIS FILE EXISTS FOR
------------------------------
The notebook's prompt tells the model it may not award a mark it cannot
quote the student's own words to support. A prompt is a request. This is
the enforcement: every awarded quote is looked for in the answer, and an
award whose quote is not actually there is **discarded** and the item
returned to the queue.

The check is deliberately downstream of the model and does not trust it.
A generous judge's failure mode is not obvious nonsense, it is plausible
nonsense at scale - hundreds of items awarded on reasons that read well
and cite nothing. Requiring a verbatim span makes that mechanically
detectable, and the audit below reports the rejection rate as a property
of the model rather than hiding it.

MATCHING IS FORGIVING ABOUT FORM, STRICT ABOUT CONTENT
------------------------------------------------------
A quote is accepted if it appears in the answer once whitespace, case,
the dash zoo and the model's own wrapping quotation marks are
normalised away. It is not accepted on a fuzzy or partial match: the
point is to confirm the model read something that is actually there,
and "nearly there" is how a paraphrase passes.

WHAT ELSE IS REFUSED
--------------------
  * a verdict for an item that is not pending (already settled, or
    belongs to a human) - applying it would silently overwrite a
    decision a cheaper, more reliable tier already made
  * marks above what the item is worth
  * a negative award
  * `decline` - which is not a failure. It sends the item to the human
    queue holding the crops, which is what declining is for.
  * a `zero` on an answer whose evidence is a drawing the model was never
    shown, or a part that lost a page - see `zero_blocked`. Refused into
    the human queue rather than settled at zero.

Run:
    python src/apply_verdicts.py verdicts.jsonl
    python src/apply_verdicts.py verdicts.jsonl --dry-run
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
from grade import write_summary
from tiers.exact import normalise

MIN_QUOTE = 8

# The model usually hands its quote back wrapped in quotation marks. Those
# are its punctuation, not the student's, and stripping them weakens nothing:
# what is left still has to appear verbatim. On the first real run this alone
# accounted for 22 of 47 rejections - form, not fabrication.
WRAPPERS = "\"'\u201c\u201d\u2018\u2019 \t\n"


def quote_supports(quote, answer):
    """Is this quote really in the student's answer?"""

    quote = normalise(quote.strip().strip(WRAPPERS))
    if len(quote) < MIN_QUOTE:
        return False, "quote too short to be evidence"
    if normalise(answer).find(quote) < 0:
        return False, "quote does not appear in the answer"
    return True, ""


def chain_questions():
    """(cie, question id) for every question the scheme marks as a chain."""

    out = set()
    for cie in (1, 2, 3):
        key = paths.KEYS_DIR / f"cie{cie}.json"
        if not key.exists():
            continue
        with open(key, encoding="utf-8") as handle:
            for spec in json.load(handle)["questions"]:
                if spec.get("chain"):
                    out.add((cie, spec["id"]))
    return out


CHAINS = chain_questions()


def zero_blocked(question, cie=None):
    """Was a zero on this question ever the model's to give?

    Rule 1 of this project: a cheap tier may award on evidence it finds,
    but is restrained from concluding absence. The model is a cheap tier
    relative to a human, and on a drawing-backed answer it is not merely
    cheap, it is blind - the crops are never sent to it.

    The prompt already says so (rule 4: if the evidence would be in a
    diagram you cannot see, decline). On the first real run the model
    obeyed that on 55 of 438 drawing-backed items and zeroed 295 of them,
    which is what a prompt alone is worth. This is the enforcement.

    An award is unaffected: finding evidence in the text is proof, and it
    still has to survive the quote check. Only the zero is refused, and it
    is refused into the human queue rather than thrown away.

    The chain carve-out is the same judgement `grade.py` already makes for
    the cheap tiers: on a sequential question a wrong block size at step
    one moves every later value, so marking later steps against the
    scheme's absolute answers charges one slip five times. The model is
    sent the scheme's own carry-forward note and told to mark against the
    student's preceding boundary. On CIE-2 3b it was sent exactly that and
    zeroed all five organizations citing the absolute addresses, for two
    students the examiner gave 10/10 and 6/10. Hence this is enforced and
    not requested.
    """

    if question.get("figures"):
        count = question["figures"]
        return (f"the answer carries {count} drawing(s) the model "
                "was never shown")
    if question.get("gaps"):
        return "this part contains a page that was lost before it was read"
    if (cie, question.get("id")) in CHAINS:
        return ("this question is a chain, and a zero here charges one "
                "early slip at every later step")
    return None


def load_verdicts(path):
    verdicts = {}
    duplicates = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["booklet_id"], row["question"], row["item_index"])
            if key in verdicts:
                duplicates += 1
            # A re-run appends rather than overwrites, so the last
            # record for an item is the current one.
            verdicts[key] = row
    return verdicts, duplicates


def apply(verdicts, dry_run=False):
    applied, rejected, skipped, refused = [], [], [], []
    counts = Counter()
    touched = {}

    for path in sorted(paths.MARKS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)

        changed = False
        for question in report["questions"]:
            for index, item in enumerate(question["items"]):
                key = (report["booklet_id"], question["id"], index)
                verdict = verdicts.get(key)
                if verdict is None:
                    continue

                counts[verdict.get("verdict", "?")] += 1

                if item["awarded"] is not None:
                    skipped.append((key, "item was already settled by "
                                         f"the {item['tier']} tier"))
                    continue
                if item["tier"] == "human":
                    skipped.append((key, "item belongs to the human tier"))
                    continue

                kind = verdict.get("verdict")
                if kind == "decline":
                    item["llm_declined"] = verdict.get("reason", "")
                    changed = True
                    continue

                if kind == "zero":
                    blocked = zero_blocked(question, report["cie"])
                    if blocked:
                        item["llm_declined"] = (
                            "the model said zero, but " + blocked)
                        refused.append((key, blocked))
                        changed = True
                        continue
                    item.update(awarded=0.0, tier="llm",
                                why=verdict.get("reason", "model: zero"),
                                evidence=[])
                    applied.append((key, 0.0))
                    changed = True
                    continue

                if kind != "award":
                    rejected.append((key, f"unknown verdict {kind!r}"))
                    continue

                marks = verdict.get("marks")
                available = item["marks_available"]
                if not isinstance(marks, (int, float)):
                    rejected.append((key, "marks is not a number"))
                    continue
                if marks < 0 or marks > available + 1e-9:
                    rejected.append(
                        (key, f"awarded {marks} of {available} available"))
                    continue

                ok, why = quote_supports(verdict.get("quote", ""),
                                         question["answer"])
                if not ok:
                    rejected.append((key, why))
                    continue

                item.update(
                    awarded=float(marks), tier="llm",
                    why=verdict.get("reason", "model: award"),
                    evidence=[], quote=verdict.get("quote", ""),
                )
                applied.append((key, float(marks)))
                changed = True

        if changed:
            settled = sum(
                i["awarded"] for q in report["questions"] if q["counted"]
                for i in q["items"] if i["awarded"] is not None)
            pending = sum(
                i["marks_available"] for q in report["questions"]
                if q["counted"] for i in q["items"] if i["awarded"] is None)
            report["marks_settled"] = round(settled, 2)
            report["marks_pending"] = round(pending, 2)
            for question in report["questions"]:
                question["marks_settled"] = round(sum(
                    i["awarded"] for i in question["items"]
                    if i["awarded"] is not None), 2)
                question["marks_pending"] = round(sum(
                    i["marks_available"] for i in question["items"]
                    if i["awarded"] is None), 2)
            touched[path] = report

    if not dry_run:
        for path, report in touched.items():
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        if touched:
            write_summary()          # grade.py wrote it before we decided

    return applied, rejected, skipped, refused, counts, len(touched)


def render(applied, rejected, skipped, refused, counts, verdicts):
    out = ["# Model tier audit", ""]
    out.append("Generated by `src/apply_verdicts.py`.")
    out.append("")
    out.append(f"- {len(verdicts)} verdicts read")
    for kind, count in counts.most_common():
        out.append(f"  - `{kind}`: {count}")
    out.append(f"- **{len(applied)} applied**")
    out.append(f"- **{len(rejected)} rejected**")
    out.append(f"- **{len(refused)} zeros refused** and sent to a human")
    out.append(f"- {len(skipped)} skipped (not the model's to decide)")
    out.append("")

    awards = counts.get("award", 0)
    if awards:
        out.append(f"Rejection rate on awards: "
                   f"**{len(rejected) / awards:.1%}**.")
        out.append("")
        out.append("This is a property of the model, not of the corpus. An")
        out.append("award is rejected when the quote it offered is not")
        out.append("actually in the student's answer - which is exactly the")
        out.append("failure a prompt alone cannot prevent.")
        out.append("")

    if rejected:
        out.append("## Rejected awards")
        out.append("")
        out.append("| booklet | question | item | why |")
        out.append("|---|---|---|---|")
        for (booklet, question, index), why in rejected:
            out.append(f"| `{booklet}` | `{question}` | {index} | {why} |")
        out.append("")

    if refused:
        out.append("## Zeros refused")
        out.append("")
        out.append("A zero the model was not in a position to give. See")
        out.append("`zero_blocked` - the evidence sits in a drawing it was")
        out.append("never shown, so its silence is not the student's. These")
        out.append("items go to the human queue, not to zero.")
        out.append("")
        reasons = Counter(why for _, why in refused)
        for why, count in reasons.most_common():
            out.append(f"- {count} x {why}")
        out.append("")

    if skipped:
        out.append("## Skipped")
        out.append("")
        reasons = Counter(why for _, why in skipped)
        for why, count in reasons.most_common():
            out.append(f"- {count} x {why}")
        out.append("")

    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verdicts", help="verdicts.jsonl from the notebook")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.verdicts)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")

    verdicts, duplicates = load_verdicts(path)
    if duplicates:
        print(f"{duplicates} item(s) had more than one verdict; "
              "the last was kept")

    applied, rejected, skipped, refused, counts, files = apply(
        verdicts, dry_run=args.dry_run)

    report = render(applied, rejected, skipped, refused, counts, verdicts)
    if not args.dry_run:
        out = paths.OUT_DIR / "verdict_audit.md"
        out.write_text(report, encoding="utf-8")

    print(f"\n{len(verdicts)} verdicts: {dict(counts)}")
    print(f"  applied  {len(applied)}")
    print(f"  rejected {len(rejected)}")
    print(f"  refused  {len(refused)} zero(s) the model could not see to give")
    print(f"  skipped  {len(skipped)}")
    awards = counts.get("award", 0)
    if awards:
        print(f"\nrejection rate on awards: {len(rejected) / awards:.1%}")
    if args.dry_run:
        print("\n(dry run - nothing written)")
    else:
        print(f"\n{files} booklet(s) updated")
        print(f"{paths.OUT_DIR / 'verdict_audit.md'}")
        print("\nthen re-run: python src/agreement.py")


if __name__ == "__main__":
    main()
