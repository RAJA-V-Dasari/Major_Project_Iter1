"""
Mark the booklets against the answer schemes.

    keys/cie*.json + handoff  ->  output/marks/<booklet>.json
                              ->  output/summary.csv
                              ->  output/queue_llm.jsonl
                              ->  output/queue_human.jsonl

THE LADDER
----------
Each rubric item is decided by the cheapest tier that can honestly
decide it, and every award records which tier decided and on what
evidence:

  exact     the item names concrete things - 57088, /26, 14.24.74.126.
            Present with its keywords -> awarded. Absent with no
            keywords -> zero. Deterministic in both directions, and it
            can explain itself completely.
  keyword   the same logic where the item has no exact values: named
            essentially everything -> awarded; named almost none of
            it -> zero.
  semantic  the student named some of the terms but not enough to be
            sure. An embedding model breaks the tie - and may only
            CONFIRM keyword evidence, never supply it.
  llm       everything still ambiguous, queued rather than guessed.
  human     anything whose evidence is a drawing, anything touching the
            lost page, and anything the model declined.

The tiers are ordered by cost, but that is not the point of the order.
Each tier is only trusted where it is actually reliable, and the
carve-outs below are where that ends.

PART C IS A CHOICE, AND THE CHOICE IS THE BETTER HALF
-----------------------------------------------------
3a or 3b, 4a or 4b - max(), never the sum. A student who answered both
has not earned twice the marks. The examiner's own covers deviate from
this on four booklets (see gold/gold_notes.md); the paper's instruction
is what is implemented here, and the disagreement is reported rather
than reproduced.

WHERE A LOW SCORE IS NOT A ZERO
-------------------------------
Four carve-outs, each against a real way of being wrong:

  * **Chain questions are never zeroed by a cheap tier.** Subnetting,
    fragmentation, CRC and the delay cascade all hang off a sequence. A
    student who takes a wrong block size at step one produces values
    that are correct relative to their own error and absent from the
    key. Zeroing those charges one slip five times.
  * **A part containing a `gap` is incomplete, not wrong.** One page was
    lost before it was ever read.
  * **An answer whose evidence is a drawing cannot be zeroed on prose.**
    This one was measured, not imagined. On student_01_cie_2, question
    2b's entire routing table sits in four crops behind 73 characters of
    prose, and half of question 1 was written onto the handshake diagram
    itself. The examiner gave 5/5 for both; marking the text alone
    scored them 0 and 2. Nothing below the human tier reads pixels, so
    the absence of a claim in the text says nothing about the answer.
  * **A question the examiner marked but we cannot find** is reported,
    not scored zero. That combination means the content exists and we
    have failed to locate it - which is a bug in us, not a zero for the
    student.

Note the asymmetry running through all four: a cheap tier may always
AWARD on evidence it finds, and is only ever restrained from concluding
absence. Finding the evidence is proof; failing to find it is not.

Run:
    python src/grade.py --all
    python src/grade.py --booklet student_01_cie_2 --verbose
    python src/grade.py --all --no-semantic     # no torch needed
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_handoff as lh
import paths
from tiers import DEFAULT, exact as tier_exact
from tiers import semantic as tier_semantic

CHOICE_ROWS = {"3", "4"}


# ----------------------------------------------------------------- data


def load_keys():
    keys = {}
    for path in sorted(paths.KEYS_DIR.glob("cie*.json")):
        with open(path, encoding="utf-8") as handle:
            key = json.load(handle)
        keys[int(key["cie"])] = key
    if not keys:
        raise SystemExit(f"no keys in {paths.KEYS_DIR}")
    return keys


def load_alignment():
    """{(booklet_id, label): question_id} for the resolved odd labels."""

    path = paths.OUT_DIR / "alignment.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        (row["booklet_id"], row["label"]): row["proposed"]
        for row in data["proposals"]
        if row["confident"] and row["proposed"]
    }


def load_gold():
    if not paths.GOLD_CSV.exists():
        return {}
    with open(paths.GOLD_CSV, encoding="utf-8", newline="") as handle:
        return {r["booklet_id"]: r for r in csv.DictReader(handle)}


def examiner_marked(gold_row, qid):
    """Did the faculty award anything against this question? True/False/None.

    Part C is read at the row level - the examiner uses the grid's
    columns loosely there. See gold/gold_notes.md.
    """

    if not gold_row or gold_row.get("confidence") == "no_gold":
        return None
    number = qid[0]
    if number in CHOICE_ROWS:
        value = gold_row.get(f"q{number}t", "")
        if not value.strip():
            value = "".join(gold_row.get(f"q{number}{c}", "") for c in "abcd")
        return bool(value.strip())
    letter = qid[1:] or "a"
    return bool((gold_row.get(f"q{number}{letter}") or "").strip())


def build_sections(booklet, alignment, valid_ids):
    """{question id: merged content} for one booklet."""

    sections = {}
    for part in booklet.parts:
        label = part.label
        if label not in valid_ids:
            label = alignment.get((booklet.booklet_id, label))
            if label is None:
                continue          # a structural header, or unresolved

        section = sections.setdefault(label, {
            "prose": [], "diagrams": [], "pages": [], "gaps": 0,
            "crossed": 0,
        })
        if part.prose.strip():
            section["prose"].append(part.prose.strip())
        section["diagrams"].extend(part.diagrams)
        section["pages"].extend(part.pages)
        section["gaps"] += part.gaps
        section["crossed"] += part.crossed

    for section in sections.values():
        section["text"] = "\n".join(section["prose"])
    return sections


# ------------------------------------------------------------- the ladder


def grade_item(item, section, semantic, *, in_chain, th=DEFAULT, sim=None):
    """Run one rubric item down the ladder."""

    result = {
        "point": item["point"],
        "marks_available": item["marks"],
        "figure": bool(item.get("figure")),
        "awarded": None,
        "tier": None,
        "why": None,
        "evidence": [],
    }

    answer = section["text"] if section else ""

    # When may a cheap tier record a zero? Only when failing to find the
    # evidence in the text actually means the student did not give it.
    # Three cases where it does not, each measured rather than imagined:
    no_zero = None
    if in_chain:
        no_zero = ("this question is a chain, so a wrong earlier step makes "
                   "later values legitimately differ")
    elif section and section["gaps"]:
        no_zero = "this part contains a page that was lost before it was read"
    elif section and section["diagrams"]:
        no_zero = (f"this answer carries {len(section['diagrams'])} drawing(s) "
                   "that no tier below a human can read")

    # A drawing is the evidence for this item, and nothing below the
    # human tier reads pixels. Part 1 deliberately never transcribed
    # these; guessing from the surrounding prose would be inventing.
    # But with no text AND no crop filed under the question there is no
    # drawing to wait for: the rest of the question is scored unattempted
    # below, and a person would be sent to look at nothing.
    empty = not answer.strip() and not (section and section["diagrams"])
    if item.get("figure") and not empty:
        result.update(
            tier="human",
            why="the evidence for this item is a drawing",
        )
        return result

    if not answer.strip():
        result.update(
            awarded=0.0, tier="unattempted",
            why="no content filed under this question",
        )
        return result

    text = tier_exact.normalise(answer)

    decision = tier_exact.decide(item, text, no_zero=no_zero, th=th)
    if decision and decision["awarded"] is not None:
        result.update(decision)
        return result

    escalation = decision["why"] if decision else None

    if semantic is not None:
        similarity, quote = sim if sim else (None, None)
        decision = semantic.decide(item, text, answer, no_zero=no_zero,
                                   th=th, similarity=similarity,
                                   quote=quote)
        if decision:
            result.update(decision)
            return result

    result.update(
        tier="llm",
        why=escalation or "keyword evidence is partial - neither tier can "
                          "settle this honestly",
    )
    return result


def grade_question(spec, section, semantic, gold_row, booklet_id,
                   th=DEFAULT, sims=None):
    qid = spec["id"]
    in_chain = bool(spec.get("chain"))

    items = [grade_item(item, section, semantic, in_chain=in_chain, th=th,
                        sim=(sims[i] if sims else None))
             for i, item in enumerate(spec["rubric"])]

    settled = sum(i["awarded"] for i in items if i["awarded"] is not None)
    pending = sum(i["marks_available"] for i in items
                  if i["awarded"] is None)

    attempted = bool(section and section["text"].strip())
    marked = examiner_marked(gold_row, qid)

    # The examiner found an answer here and we did not. That is our
    # failure to locate content, not the student's failure to write it.
    missing_but_marked = marked is True and not attempted

    return {
        "id": qid,
        "part": spec["part"],
        "marks_available": spec["marks"],
        "choice_with": spec.get("choice_with"),
        "breakdown_source": spec["breakdown_source"],
        "attempted": attempted,
        "answer_chars": len(section["text"]) if section else 0,
        # The queues have to be self-contained: whatever runs the model
        # tier gets the answer, the rubric and the marks in one record
        # and needs no access to this repo or the corpus.
        "answer": section["text"] if section else "",
        "figures": len(section["diagrams"]) if section else 0,
        "gaps": section["gaps"] if section else 0,
        "examiner_marked": marked,
        "missing_but_marked": missing_but_marked,
        "marks_settled": round(settled, 2),
        "marks_pending": round(pending, 2),
        "items": items,
    }


def resolve_choices(questions):
    """Count the better half of each Part C pair, never the sum."""

    by_id = {q["id"]: q for q in questions}
    counted, seen, notes = [], set(), []

    for question in questions:
        qid = question["id"]
        if qid in seen:
            continue
        partner_id = question.get("choice_with")
        if not partner_id or partner_id not in by_id:
            counted.append(qid)
            seen.add(qid)
            continue

        partner = by_id[partner_id]
        seen.update({qid, partner_id})

        # Rank on settled marks; break ties on how much is still
        # pending, then on how much the student actually wrote.
        winner = max(
            (question, partner),
            key=lambda q: (q["marks_settled"],
                           q["marks_settled"] + q["marks_pending"],
                           q["answer_chars"]),
        )
        loser = partner if winner is question else question
        counted.append(winner["id"])

        winner["counted"] = True
        loser["counted"] = False

        if loser["attempted"]:
            notes.append(
                f"{qid}/{partner_id}: both attempted; counting "
                f"{winner['id']} ({winner['marks_settled']:g} settled) over "
                f"{loser['id']} ({loser['marks_settled']:g})"
            )
        # If the loser still has enough pending to overtake, the choice
        # is not final until those items are decided.
        if (loser["marks_settled"] + loser["marks_pending"]
                > winner["marks_settled"] + 1e-9):
            winner["choice_provisional"] = True
            loser["choice_provisional"] = True
            notes.append(
                f"{qid}/{partner_id}: PROVISIONAL - {loser['id']} has "
                f"{loser['marks_pending']:g} marks pending and could "
                f"overtake {winner['id']}"
            )

    for question in questions:
        question.setdefault("counted", True)
        question.setdefault("choice_provisional", False)

    return counted, notes


def grade_booklet(booklet, key, alignment, semantic, gold_row,
                  th=DEFAULT, sim_cache=None):
    valid_ids = [q["id"] for q in key["questions"]]
    sections = build_sections(booklet, alignment, valid_ids)

    questions = [
        grade_question(spec, sections.get(spec["id"]), semantic, gold_row,
                       booklet.booklet_id, th=th,
                       sims=(sim_cache or {}).get(spec["id"]))
        for spec in key["questions"]
    ]
    counted, notes = resolve_choices(questions)

    settled = sum(q["marks_settled"] for q in questions if q["counted"])
    pending = sum(q["marks_pending"] for q in questions if q["counted"])

    return {
        "booklet_id": booklet.booklet_id,
        "student": booklet.student,
        "cie": booklet.cie,
        "max_marks": key["max_marks"],
        "counted": counted,
        "marks_settled": round(settled, 2),
        "marks_pending": round(pending, 2),
        "choice_notes": notes,
        "questions": questions,
    }


# ------------------------------------------------------------------ output


def queue_rows(report, key_by_cie):
    """The two review queues, each record self-contained."""

    llm, human = [], []
    key = key_by_cie[report["cie"]]
    specs = {q["id"]: q for q in key["questions"]}

    for question in report["questions"]:
        if not question["counted"]:
            continue
        spec = specs[question["id"]]

        if question["missing_but_marked"]:
            human.append({
                "kind": "missing_but_marked",
                "booklet_id": report["booklet_id"],
                "question": question["id"],
                "marks_available": question["marks_available"],
                "note": "the examiner awarded marks for this question but no "
                        "content is filed under it - locate the answer before "
                        "recording a zero",
            })

        for index, item in enumerate(question["items"]):
            if item["awarded"] is not None:
                continue
            record = {
                "booklet_id": report["booklet_id"],
                "cie": report["cie"],
                "question": question["id"],
                "item_index": index,
                "point": item["point"],
                "marks_available": item["marks_available"],
                "why": item["why"],
                # Self-contained: whatever runs the tier gets the answer,
                # the rubric and the marks in one record and needs no
                # access to this repo.
                "question_text": spec["question"],
                "model_solution": spec["solution"],
                "chain": spec.get("chain"),
                "answer": question["answer"],
            }
            # Only an item whose RUBRIC says the evidence is a drawing
            # goes straight to a person. An item that merely sits in an
            # answer containing a drawing still goes to the model: it
            # can read the prose, and where the evidence really is in
            # the figure it is told to decline rather than guess. That
            # keeps the human tier for what actually needs eyes instead
            # of for everything near a picture.
            record["answer_has_figures"] = question["figures"]
            if item["tier"] == "human":
                record["kind"] = "figure"
                human.append(record)
            else:
                llm.append(record)

    return llm, human


def write_outputs(reports, key_by_cie):
    paths.MARKS_DIR.mkdir(parents=True, exist_ok=True)

    all_llm, all_human = [], []
    for report in reports:
        out = paths.MARKS_DIR / f"{report['booklet_id']}.json"
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        llm, human = queue_rows(report, key_by_cie)
        all_llm.extend(llm)
        all_human.extend(human)

    for name, rows in (("queue_llm.jsonl", all_llm),
                       ("queue_human.jsonl", all_human)):
        with open(paths.OUT_DIR / name, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    write_summary(reports)
    return all_llm, all_human


SUMMARY_FIELDS = ["booklet_id", "student", "cie", "counted", "marks_settled",
                  "marks_pending", "max_marks"]


def write_summary(reports=None):
    """Rewrite output/summary.csv from the marks on disk.

    This is importable, and the tiers that run after grade.py call it,
    because grade.py writes this file before either of them has decided
    anything. Left to grade.py alone it always describes the ladder's own
    verdicts and nothing since - which is how summary.csv came to report
    21.0 for a booklet whose marks file said 40.0 and whose agreement row
    said 31. All three were true when written, which is the problem.
    """

    if reports is None:
        reports = [json.load(open(p, encoding="utf-8"))
                   for p in sorted(paths.MARKS_DIR.glob("*.json"))]
    with open(paths.OUT_DIR / "summary.csv", "w", encoding="utf-8",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for report in reports:
            writer.writerow({
                "booklet_id": report["booklet_id"],
                "student": report["student"],
                "cie": report["cie"],
                "counted": " ".join(report["counted"]),
                "marks_settled": report["marks_settled"],
                "marks_pending": report["marks_pending"],
                "max_marks": report["max_marks"],
            })


def summarise(reports, llm, human):
    items = [i for r in reports for q in r["questions"] if q["counted"]
             for i in q["items"]]
    by_tier = {}
    for item in items:
        by_tier[item["tier"]] = by_tier.get(item["tier"], 0) + 1

    decided = sum(1 for i in items if i["awarded"] is not None)
    settled = sum(r["marks_settled"] for r in reports)
    pending = sum(r["marks_pending"] for r in reports)

    print(f"\n{len(reports)} booklets, {len(items)} rubric items counted")
    print(f"\ndecided without a human or a model: {decided}/{len(items)} "
          f"({decided / len(items):.0%})")
    for tier, count in sorted(by_tier.items(), key=lambda kv: -kv[1]):
        print(f"  {tier:12} {count:>5}  ({count / len(items):.0%})")

    print(f"\nmarks settled {settled:.0f}, pending {pending:.0f} "
          f"({settled / (settled + pending):.0%} settled)")
    print(f"queues: {len(llm)} for the model, {len(human)} for a human")

    missing = [q for r in reports for q in r["questions"]
               if q["missing_but_marked"] and q["counted"]]
    if missing:
        print(f"\n{len(missing)} question(s) the examiner marked but we "
              f"could not locate - see queue_human.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--booklet")
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--no-alignment", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    keys = load_keys()
    alignment = {} if args.no_alignment else load_alignment()
    gold = load_gold()
    _, booklets = lh.load_all()

    if args.booklet:
        booklets = [b for b in booklets if b.booklet_id == args.booklet]
        if not booklets:
            raise SystemExit(f"no booklet {args.booklet}")
    elif not args.all:
        raise SystemExit("pass --all or --booklet")

    semantic = None
    if not args.no_semantic:
        if tier_semantic.available():
            print("loading the semantic tier...", flush=True)
            semantic = tier_semantic.Semantic()
        else:
            print("sentence-transformers not installed - the semantic tier "
                  "is skipped and its items go to the model queue.\n"
                  "  install with: uv pip install -e .[semantic]")

    reports = []
    for booklet in booklets:
        report = grade_booklet(booklet, keys[booklet.cie], alignment,
                               semantic, gold.get(booklet.booklet_id))
        reports.append(report)
        if args.verbose:
            print(f"\n{report['booklet_id']}  "
                  f"{report['marks_settled']}/{report['max_marks']} settled, "
                  f"{report['marks_pending']} pending")
            for question in report["questions"]:
                flag = "" if question["counted"] else "   (not counted)"
                print(f"  {question['id']:3} "
                      f"{question['marks_settled']:>5}/"
                      f"{question['marks_available']:<4} "
                      f"pending {question['marks_pending']:<5}{flag}")
                for item in question["items"]:
                    award = ("pending" if item["awarded"] is None
                             else f"{item['awarded']:g}")
                    print(f"       [{item['tier'] or '-':10}] {award:>7}  "
                          f"{item['point'][:60]}")
            for note in report["choice_notes"]:
                print(f"  note: {note}")

    llm, human = write_outputs(reports, keys)
    summarise(reports, llm, human)


if __name__ == "__main__":
    main()
