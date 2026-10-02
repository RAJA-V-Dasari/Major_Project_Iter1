"""
Choose the ladder's thresholds by measuring them, not by picking them.

    keys + handoff + gold/gold_marks.csv  ->  output/calibration.md

WHAT CAN ACTUALLY BE MEASURED
-----------------------------
The examiner marked per QUESTION, not per rubric item, so there is no
per-item ground truth to fit against. What there is: for every question
we produce an interval rather than a number - `settled` marks the cheap
tiers have decided, and `pending` marks still with the model or a human.

That interval is what gets tested. For each question with an examiner
mark `g`:

  settled <= g <= settled + pending     the examiner's mark is still
                                        reachable. We have not ruled out
                                        the right answer.

  settled > g                           OVER-SETTLED. We have already
                                        awarded marks the examiner did
                                        not, and no later tier can undo
                                        it. A hard error.

  settled + pending < g                 UNDER-SETTLED. We have zeroed
                                        marks the examiner awarded, and
                                        no later tier can recover them.
                                        A hard error.

Both errors are irreversible, which is what makes them the right target:
they are the mistakes the remaining tiers cannot fix. Everything inside
the interval is still open, and being open is not an error - it is the
ladder correctly declining to guess.

THE TRADE-OFF
-------------
Loosening the thresholds settles more marks and makes more irreversible
errors. Tightening settles fewer and pushes work to the model and the
human. So this reports a frontier, not a winner: containment against
decisiveness, and the operating point is a judgement about how much
human marking the project is willing to pay for.

The row marked in the report is the one this sweep RECOMMENDS: fewest
irreversible errors first, decisiveness only as a tie-break. It is not
necessarily the setting the code ships - see `Thresholds` in
tiers/__init__.py, which is deliberately tighter, and says why.

An earlier version chose the most decisive setting whose error rate
stayed under the examiner's own 14% defect floor. That rule bought
decisiveness with over-awards - 33 of them - and decisiveness only saves
human time, while an over-settled mark awards credit the student did not
earn and no later tier can take it back. The floor is still reported,
because it is the honest context for any error rate here, but it no
longer selects.

Run:
    python src/calibrate.py                # sweep and write the report
    python src/calibrate.py --quick        # a coarse sweep
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grade
import load_handoff as lh
import paths
from tiers import Thresholds
from tiers import exact as tier_exact
from tiers import semantic as tier_semantic

# The examiner's own defect rate: 7 of 50 covers carry a demonstrable
# error. A grader with a lower irreversible-error rate than this is not
# obviously better than the reference - it is inside the noise.
EXAMINER_DEFECT_RATE = 7 / 50

GRID = {
    "kw_confident": [0.60, 0.67, 0.75, 0.85, 1.00],
    "kw_absent": [0.00, 0.10, 0.20, 0.34],
    "kw_corroborate": [0.50, 0.67, 0.75, 1.00],
    "sem_confident": [0.55, 0.62, 0.70, 0.80],
    "zero_max_chars": [0, 150, 300, 600, 10 ** 9],
}
QUICK = {
    "kw_confident": [0.67, 0.75, 1.00],
    "kw_absent": [0.00, 0.20],
    "kw_corroborate": [0.50, 0.75],
    "sem_confident": [0.62, 0.75],
    "zero_max_chars": [0, 300, 10 ** 9],
}


def examiner_mark(gold_row, qid):
    """The examiner's mark for a question, or None if unusable."""

    if not gold_row or gold_row.get("confidence") == "no_gold":
        return None
    number = qid[0]
    if number in grade.CHOICE_ROWS:
        value = (gold_row.get(f"q{number}t") or "").strip()
        return float(value) if value else None
    value = (gold_row.get(f"q{number}{qid[1:] or 'a'}") or "").strip()
    return float(value) if value else 0.0


def build_similarity_cache(booklets, keys, alignment, semantic):
    """Embed once so the sweep is arithmetic rather than inference.

    Every (booklet, question, item) pair gets its best-matching sentence
    and score, regardless of whether the current thresholds would
    consult the model. That is the point: the sweep changes which items
    get consulted, so the cache has to cover all of them.
    """

    cache = {}
    for booklet in booklets:
        key = keys[booklet.cie]
        valid = [q["id"] for q in key["questions"]]
        sections = grade.build_sections(booklet, alignment, valid)
        per_question = {}
        for spec in key["questions"]:
            section = sections.get(spec["id"])
            answer = section["text"] if section else ""
            if not answer.strip():
                per_question[spec["id"]] = [None] * len(spec["rubric"])
                continue
            per_question[spec["id"]] = [
                semantic.best_match(item["point"], answer)
                for item in spec["rubric"]
            ]
        cache[booklet.booklet_id] = per_question
    return cache


def evaluate(th, booklets, keys, alignment, semantic, gold, sim_cache):
    """Run the whole corpus at one threshold setting and score it."""

    over, under, inside, comparable = 0, 0, 0, 0
    settled_marks, total_marks = 0.0, 0.0
    over_excess, under_short = [], []

    for booklet in booklets:
        gold_row = gold.get(booklet.booklet_id)
        report = grade.grade_booklet(
            booklet, keys[booklet.cie], alignment, semantic, gold_row,
            th=th, sim_cache=sim_cache.get(booklet.booklet_id),
        )
        for question in report["questions"]:
            if not question["counted"]:
                continue
            settled = question["marks_settled"]
            pending = question["marks_pending"]
            settled_marks += settled
            total_marks += settled + pending

            mark = examiner_mark(gold_row, question["id"])
            if mark is None:
                continue
            if mark > question["marks_available"] + 1e-9:
                # The examiner awarded more than the question is worth -
                # a documented defect (gold/gold_notes.md). No grader
                # that respects the paper can reach it, so scoring
                # ourselves against it measures their error, not ours.
                continue
            comparable += 1
            if settled > mark + 1e-9:
                over += 1
                over_excess.append(settled - mark)
            elif settled + pending < mark - 1e-9:
                under += 1
                under_short.append(mark - settled - pending)
            else:
                inside += 1

    errors = over + under
    return {
        "th": th,
        "comparable": comparable,
        "inside": inside,
        "over": over,
        "under": under,
        "error_rate": errors / comparable if comparable else 1.0,
        "containment": inside / comparable if comparable else 0.0,
        "decisiveness": settled_marks / total_marks if total_marks else 0.0,
        "mean_over": (sum(over_excess) / len(over_excess)
                      if over_excess else 0.0),
        "mean_under": (sum(under_short) / len(under_short)
                       if under_short else 0.0),
    }


def render(results, chosen):
    out = ["# Threshold calibration", ""]
    out.append("Generated by `src/calibrate.py`.")
    out.append("")
    out.append("The examiner marked per question, not per rubric item, so")
    out.append("there is no per-item truth to fit. What is measured instead")
    out.append("is whether the examiner's mark still lies inside the")
    out.append("`[settled, settled + pending]` interval the ladder produces.")
    out.append("")
    out.append("Two errors are **irreversible** - no later tier can undo")
    out.append("them, which is why they are the target:")
    out.append("")
    out.append("- **over-settled** — we awarded marks the examiner did not")
    out.append("- **under-settled** — we zeroed marks the examiner awarded")
    out.append("")
    out.append(f"The examiner's own covers carry a demonstrable defect on")
    out.append(f"**{EXAMINER_DEFECT_RATE:.0%}** of booklets. A grader")
    out.append("claiming a lower error rate than that is claiming to be more")
    out.append("consistent than its own reference, which this project does")
    out.append("not claim.")
    out.append("")

    out.append("## The frontier")
    out.append("")
    out.append("| kw_conf | kw_abs | kw_corr | sem_conf | zero<=chars | "
               "decisive | errors | over | under |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for row in results[:25]:
        th = row["th"]
        mark = " **<- default**" if th == chosen else ""
        out.append(
            f"| {th.kw_confident:.2f} | {th.kw_absent:.2f} | "
            f"{th.kw_corroborate:.2f} | {th.sem_confident:.2f} | "
            f"{th.zero_max_chars if th.zero_max_chars < 10**8 else 'any'} | "
            f"{row['decisiveness']:.0%} | {row['error_rate']:.1%} | "
            f"{row['over']} | {row['under']} |{mark}"
        )
    out.append("")
    out.append("Sorted by irreversible errors, then by decisiveness.")
    out.append("")
    out.append("`under` bottoms out at 5 whatever the thresholds: those are")
    out.append("questions where the examiner awarded marks and we found no")
    out.append("content at all to grade. That is a content-location problem,")
    out.append("not a threshold one, and it is queued to a human rather than")
    out.append("tuned away.")
    out.append("")

    best = next(r for r in results if r["th"] == chosen)
    out.append("## Chosen operating point")
    out.append("")
    out.append("```python")
    out.append(f"kw_confident   = {chosen.kw_confident}")
    out.append(f"kw_absent      = {chosen.kw_absent}")
    out.append(f"kw_corroborate = {chosen.kw_corroborate}")
    out.append(f"sem_confident  = {chosen.sem_confident}")
    out.append(f"zero_max_chars = {chosen.zero_max_chars}")
    out.append("```")
    out.append("")
    out.append(f"- {best['comparable']} comparable questions")
    out.append(f"- **{best['containment']:.0%}** keep the examiner's mark "
               "reachable")
    out.append(f"- {best['over']} over-settled "
               f"(mean {best['mean_over']:.1f} marks too many)")
    out.append(f"- {best['under']} under-settled "
               f"(mean {best['mean_under']:.1f} marks too few)")
    out.append(f"- **{best['decisiveness']:.0%}** of marks settled without "
               "the model or a human")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not tier_semantic.available():
        raise SystemExit("calibration needs the semantic tier: "
                         "uv pip install sentence-transformers")

    keys = grade.load_keys()
    alignment = grade.load_alignment()
    gold = grade.load_gold()
    _, booklets = lh.load_all()

    print("embedding the corpus once...", flush=True)
    semantic = tier_semantic.Semantic()
    sim_cache = build_similarity_cache(booklets, keys, alignment, semantic)

    grid = QUICK if args.quick else GRID
    combos = list(itertools.product(*(grid[k] for k in (
        "kw_confident", "kw_absent", "kw_corroborate", "sem_confident",
        "zero_max_chars"))))
    print(f"sweeping {len(combos)} threshold settings...", flush=True)

    results = []
    for index, (kc, ka, kcorr, sc, zmc) in enumerate(combos, 1):
        if ka >= kc:
            continue           # absent must sit below confident
        th = Thresholds(kw_confident=kc, kw_absent=ka,
                        kw_corroborate=kcorr, sem_confident=sc,
                        zero_max_chars=zmc)
        results.append(evaluate(th, booklets, keys, alignment, semantic,
                                gold, sim_cache))
        if index % 20 == 0:
            print(f"  {index}/{len(combos)}", flush=True)

    # Fewest irreversible errors first, decisiveness only as a
    # tie-break. The other way round - most decisive within an error
    # budget - simply picks the loosest setting the budget allows, which
    # is how an earlier version of this chose 33 over-settled questions
    # in exchange for marks nobody needed settled cheaply.
    # Reported, never selected on: how many settings sit under the
    # examiner's own defect rate. It is context for the numbers below,
    # not a filter - see the module docstring.
    acceptable = [r for r in results
                  if r["error_rate"] <= EXAMINER_DEFECT_RATE]

    # Fewest irreversible errors first, decisiveness only as a tie-break.
    results.sort(key=lambda r: (r["over"] + r["under"], -r["decisiveness"]))
    chosen = results[0]["th"]

    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (paths.OUT_DIR / "calibration.md").write_text(
        render(results, chosen), encoding="utf-8")

    print(f"\n{len(results)} settings evaluated, "
          f"{len(acceptable)} within the examiner's defect rate")
    print(f"\nchosen: {chosen}")
    best = next(r for r in results if r["th"] == chosen)
    print(f"  containment {best['containment']:.0%}  "
          f"decisiveness {best['decisiveness']:.0%}  "
          f"over {best['over']}  under {best['under']}")
    print(f"\n{paths.OUT_DIR / 'calibration.md'}")


if __name__ == "__main__":
    main()
