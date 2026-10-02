"""
Apply the model tier's verdicts, and reject the ones it cannot justify.

    verdicts.jsonl + data/marking/marks/*.json  ->  updated marks
                                          ->  data/marking/verdict_audit.md

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
    the human queue rather than settled at zero. A verdict whose `shown`
    count covers every crop was not blind, so the drawing clause (and
    only that clause) does not apply to it - see `shown_all`.

Run:
    python marking/src/apply_verdicts.py verdicts.jsonl
    python marking/src/apply_verdicts.py verdicts.jsonl --dry-run
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


def shown_all(question, verdict):
    """Was this verdict given by a reader that saw every drawing?

    A verdict row may carry `shown`: how many of the answer's crops the
    reader was actually given (`claude_tier.py --vision` sends them; the
    text-only readers never do). Only a count covering all of them lifts
    the blindness that `zero_blocked` and the figure routing guard
    against - one crop of four is still three unseen.
    """

    figures = question.get("figures") or 0
    if figures == 0:
        # Nothing was cropped, so the only way to see a drawing part 1
        # missed is the page itself - `pages_seen` records that it was
        # looked at, rather than taken on the crop count's word.
        return bool(verdict.get("pages_seen"))
    return int(verdict.get("shown") or 0) >= figures


def zero_blocked(question, cie=None, saw_drawings=False):
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

    `saw_drawings` lifts the first clause and only the first: the reason
    for it is blindness, and a reader that was shown every crop is not
    blind. The lost page and the chain are unchanged by seeing pixels -
    the page is still lost, and the chain still needs a person to judge
    carry-forward against the model's own tendency to cite the key.
    """

    if question.get("figures") and not saw_drawings:
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
    applied, rejected, skipped, refused, on_drawing = [], [], [], [], []
    counts, models = Counter(), Counter()
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
                models[verdict.get("model") or "?"] += 1

                if item["awarded"] is not None:
                    skipped.append((key, "item was already settled by "
                                         f"the {item['tier']} tier"))
                    continue
                saw = shown_all(question, verdict)
                # grade.py routes an item to a person when its rubric
                # point IS a drawing, because no text reader can judge
                # it. A reader that was shown every crop can; one that
                # was not still cannot.
                if item["tier"] == "human" and not saw:
                    skipped.append((key, "item belongs to the human tier"))
                    continue

                # Which reader produced this verdict. The model tier has had
                # more than one (Qwen on Colab, Qwen via Ollama, Claude in
                # session), and a mark that cannot say which one gave it
                # cannot be compared against a re-run by another. Recorded
                # wherever the verdict is, and nowhere it was rejected.
                model = verdict.get("model") or "unrecorded"

                kind = verdict.get("verdict")
                if kind == "decline":
                    item["llm_declined"] = verdict.get("reason", "")
                    item["llm_model"] = model
                    changed = True
                    continue

                if kind == "zero":
                    blocked = zero_blocked(question, report["cie"],
                                           saw_drawings=saw)
                    if blocked:
                        item["llm_declined"] = (
                            "the model said zero, but " + blocked
                            + ". Its reason: " + verdict.get("reason", ""))
                        item["llm_model"] = model
                        refused.append((key, blocked))
                        changed = True
                        continue
                    item.update(awarded=0.0, tier="llm", llm_model=model,
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
                crop = verdict.get("crop")
                page = verdict.get("page")
                reads = (verdict.get("crop_reads") or "").strip()
                on_crop = (isinstance(crop, int)
                           and 0 <= crop < question["figures"])
                on_page = bool(page) and page in (verdict.get("pages_seen")
                                                  or [])
                if not ok and not (saw and reads and (on_crop or on_page)):
                    rejected.append((key, why))
                    continue

                item.update(
                    awarded=float(marks), tier="llm", llm_model=model,
                    why=verdict.get("reason", "model: award"),
                    evidence=[], quote=verdict.get("quote", "") if ok else "",
                )
                if not ok:
                    # The evidence is a drawing. There is no text to find
                    # a quote in, so the check above cannot run - what is
                    # kept instead is WHICH crop and what the reader says
                    # it shows, so a person can hold one against the other
                    # in serve.py. Counted separately in the audit, never
                    # passed off as quote-checked.
                    item.update(crop=crop if on_crop else None,
                                page=None if on_crop else page,
                                crop_reads=reads)
                    on_drawing.append(key)
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

    return (applied, rejected, skipped, refused, on_drawing, counts, models,
            len(touched))


def render(applied, rejected, skipped, refused, on_drawing, counts, models,
           verdicts):
    out = ["# Model tier audit", ""]
    out.append("Generated by `marking/src/apply_verdicts.py`.")
    out.append("")
    out.append(f"- {len(verdicts)} verdicts read")
    for kind, count in counts.most_common():
        out.append(f"  - `{kind}`: {count}")
    out.append("- read by: " + ", ".join(
        f"`{model}` ({count})" for model, count in models.most_common()))
    out.append(f"- **{len(applied)} applied**")
    if on_drawing:
        out.append(f"  - of which **{len(on_drawing)} awards rest on a "
                   "drawing** - no text to quote, so the quote check could "
                   "not run; each names its crop and what it shows, for a "
                   "person to spot-check in `serve.py`")
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

    (applied, rejected, skipped, refused, on_drawing, counts, models,
     files) = apply(verdicts, dry_run=args.dry_run)

    report = render(applied, rejected, skipped, refused, on_drawing, counts,
                    models, verdicts)
    if not args.dry_run:
        out = paths.OUT_DIR / "verdict_audit.md"
        out.write_text(report, encoding="utf-8")

    print(f"\n{len(verdicts)} verdicts: {dict(counts)}")
    print(f"  read by  {dict(models)}")
    print(f"  applied  {len(applied)}"
          + (f"  ({len(on_drawing)} on a drawing, not quote-checkable)"
             if on_drawing else ""))
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
        print("\nthen re-run: python marking/src/agreement.py")


if __name__ == "__main__":
    main()
