"""
Mark assembled booklets against the CIE answer keys.

    04_evaluate/keys/cie{1,2,3}.json      what earns marks, and how many
    03_assemble/output/booklets/*/        one booklet.md per student per CIE
        -> output/marks/<booklet>.json    per-rubric-item award and evidence
        -> output/marks/summary.csv       one row per booklet
        -> output/marks/review_llm.jsonl  items the cheap tiers could not settle
        -> output/marks/review_human.jsonl  figures, and pages the reader failed

WHY A LADDER AND NOT ONE MODEL
------------------------------
The keys break every question into rubric items that each carry their own
marks - "[ 4 marks]" for the Go-Back-N half of 3a, "1 x 5" for the five
WWW components. That is the unit a human marker works in, so it is the
unit this works in too. Each item is then decided by the cheapest tier
that can honestly decide it:

  keyword   the item names concrete things - "8001", "NAT64", "/26".
            If the student wrote nearly all of them, they said it; if
            they wrote almost none, they did not. Both ends are safe.
  semantic  the student named some of the terms but not enough to be
            sure they made the claim rather than merely mentioned the
            topic. An embedding model breaks that tie.
  llm       everything still ambiguous, queued rather than guessed.
  human     anything whose evidence is a drawing, plus every page the
            reader is known to have mangled.

The tiers are ordered by cost, but that is not the point of the order.
The point is that each tier is only trusted in the region where it is
actually reliable, and the thresholds below are where that region ends.

WHY SIMILARITY NEVER DECIDES ON ITS OWN
---------------------------------------
The semantic tier corroborates keyword evidence; it cannot supply it.
This is not caution, it is a measured failure. Marking student_07's
CIE-2 3a with similarity as a decider awarded 10 out of 10 for
subnetting that is wrong in almost every value - first host 14.24.74.16,
broadcast 14.24.74.255 - because one sentence, "1st subblock - 10
addresses, 10 is not possible so we take 14", scored 0.63 to 0.75
against all five rubric items at once while keyword coverage sat at 0.0
to 0.4. Sentence embeddings measure what an answer is about, and every
wrong subnet is about subnetting. Worse, nothing stops a single sentence
being the evidence for every item in a question.

So similarity may only confirm an item that already has real keyword
support (KW_CORROBORATE). An item whose keywords are absent goes to the
LLM no matter how topical the prose reads. The cost is a bigger LLM
queue; the alternative is a grader that hands out marks for being on
subject, which is the one failure mode a marking system may not have.

WHY A LOW SCORE IS NOT A ZERO
-----------------------------
An empty section can mean the student skipped the question or that
`build_booklet` filed their answer under the wrong heading - 13 pages
are known to be unplaced, and continuation pages attach to whichever
label came last. So an item scoring near zero inside its own section is
re-checked against the whole booklet before any zero is recorded, and
the mismatch is reported. A grader that silently marks a
misfiled answer as unattempted is worse than one that asks.

Run:
    python grade.py --all
    python grade.py --booklet student_07_cie_2 --verbose
    python grade.py --all --no-semantic      # keyword tier only, no torch
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
ROOT = STAGE_DIR.parent.parent

KEYS_DIR = STAGE_DIR / "keys"
BOOKLETS = ROOT / "modules" / "03_assemble" / "output" / "booklets"
OUT_DIR = STAGE_DIR / "output" / "marks"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Keyword coverage above this means the student named essentially
# everything the item asks for; below the low mark means they named
# almost none of it. Between the two the count is not evidence either
# way and the item moves up the ladder.
KW_CONFIDENT = 0.75
KW_ABSENT = 0.20

# Similarity may only confirm an item that already has this much keyword
# support. See "why similarity never decides on its own" above.
KW_CORROBORATE = 0.50

# Cosine similarity of the rubric point against the best-matching
# sentence of the answer. Above the high mark the claim is being made in
# other words; below the low mark nothing in the answer is even on the
# subject.
SEM_CONFIDENT = 0.62
SEM_ABSENT = 0.28

# A section this short cannot hold an answer; treat it as unattempted
# rather than scoring noise.
MIN_ANSWER_CHARS = 12

BOOKLET_NAME = re.compile(r"^student_(\d+)_cie_(\d+)$")
HEADING = re.compile(r"^##\s+(.+?)\s*$", re.M)
PAGE_MARK = re.compile(r"^<!--\s*(\S+?)\s*-->\s*$", re.M)
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
FAILED_PAGE = re.compile(r"Reading failed partway down this page", re.I)

# Heading text -> question id. build_booklet writes "## 2a)" and friends;
# be liberal, because the whole point of that stage was that students
# write the label a dozen different ways.
LABEL_IN_HEADING = re.compile(r"([1-9])\s*[)\].]?\s*[(\[]?\s*([a-d])?", re.I)


def normalise(text):
    """Lowercased, de-marked-up text for substring matching."""

    text = re.sub(r"^<!--.*?-->\s*$", "", text, flags=re.M)
    text = re.sub(r"^```.*$", "", text, flags=re.M)
    text = IMAGE.sub(" ", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_`~]+", "", text)
    text = text.replace("|", " ")
    text = text.replace("→", "->").replace("←", "<-")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("∞", "inf").replace("×", "x")

    return re.sub(r"\s+", " ", text.lower()).strip()


def sentences(text):
    """Answer split into claim-sized pieces for the semantic tier.

    Handwritten answers are line-broken far more often than they are
    punctuated, so a newline counts as a break just as a full stop does.
    """
    parts = re.split(r"(?<=[.;:?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 15]


def hits(alternates, text):
    """Does any spelling of this keyword group appear in `text`?

    Short and numeric alternates match on word boundaries - without that
    the "3" in rubric item 3b matches the 3 inside 1368, and every
    student scores every numeric item. Longer alternates like an IP
    address or "bandwidth-delay product" are specific enough to match as
    plain substrings, which also lets them survive the punctuation the
    reader inevitably gets wrong.
    """
    for alt in alternates:
        alt = alt.lower().strip()
        if not alt:
            continue
        if len(alt) <= 5 and re.fullmatch(r"[\w.^/-]+", alt):
            if re.search(r"(?<![\w.])" + re.escape(alt) + r"(?![\w.])", text):
                return alt
        elif alt in text:
            return alt
    return None


def keyword_coverage(item, text):
    """(fraction of groups present, the alternates that matched)."""

    groups = item.get("keywords") or []
    if not groups:
        return None, []

    found = [hits(g, text) for g in groups]
    matched = [f for f in found if f]

    return len(matched) / len(groups), matched


class Semantic:
    """all-MiniLM-L6-v2, loaded once, or a null object if unavailable.

    The laptop path used to forbid torch outright. That rule was lifted
    for evaluation only - reading still happens on a hosted GPU - but the
    grader must still run without it, because a missing model should
    degrade the tier, not stop the marking.
    """

    def __init__(self, enabled=True):
        self.model = None
        self.reason = "disabled by --no-semantic"

        # Every rubric item in a question is matched against the same
        # answer sentences, and every booklet is matched against the same
        # rubric points. Encoding is the whole cost of this tier, so both
        # are cached rather than recomputed a hundred times over.
        self._vectors = {}

        if not enabled:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME)
            self.reason = None
        except Exception as exc:                      # noqa: BLE001
            self.reason = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self):
        return self.model is not None

    def encode(self, texts):
        """Normalised embeddings for `texts`, reusing anything seen before."""

        missing = [t for t in texts if t not in self._vectors]

        if missing:
            fresh = self.model.encode(missing,
                                      convert_to_tensor=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
            for text, vector in zip(missing, fresh):
                self._vectors[text] = vector

        import torch
        return torch.stack([self._vectors[t] for t in texts])

    def best_match(self, claim, candidates):
        """Highest cosine of `claim` against any candidate sentence."""

        if not self.ready or not candidates:
            return None, None

        from sentence_transformers import util

        vectors = self.encode([claim] + candidates)

        scores = util.cos_sim(vectors[0], vectors[1:])[0]
        best = int(scores.argmax())

        return float(scores[best]), candidates[best]


def load_keys():
    keys = {}
    for path in sorted(KEYS_DIR.glob("cie*.json")):
        with open(path, encoding="utf-8") as handle:
            key = json.load(handle)
        keys[int(key["cie"])] = key
    if not keys:
        raise SystemExit(f"no keys in {KEYS_DIR}")
    return keys


def heading_to_id(heading):
    """"2a)" or "Q3 b" -> "2a"; None if the heading names no question."""

    match = LABEL_IN_HEADING.search(heading)
    if not match:
        return None
    number, sub = match.group(1), (match.group(2) or "").lower()
    return f"{number}{sub}"


def parse_booklet(path):
    """booklet.md -> {question id: section text}, plus figures and flags."""

    text = path.read_text(encoding="utf-8")

    sections, figures, failed = {}, {}, {}
    bounds = [(m.start(), m.end(), m.group(1)) for m in HEADING.finditer(text)]

    for index, (_, end, heading) in enumerate(bounds):
        stop = bounds[index + 1][0] if index + 1 < len(bounds) else len(text)
        body = text[end:stop]

        qid = heading_to_id(heading)
        if qid is None:
            continue

        # A student can restate a label mid-answer, so a question can own
        # more than one section; keep all of it.
        sections[qid] = (sections.get(qid, "") + "\n" + body).strip()
        figures.setdefault(qid, []).extend(IMAGE.findall(body))
        failed[qid] = failed.get(qid, 0) + len(FAILED_PAGE.findall(body))

    return sections, figures, failed, text


def decide(item, section_text, whole_text, semantic):
    """Award marks for one rubric item, and say which tier decided it."""

    marks = item["marks"]
    point = item["point"]

    result = {
        "point": point,
        "marks_available": marks,
        "figure": bool(item.get("figure")),
    }

    # Evidence that is a drawing cannot be read off the transcription.
    # `build_booklet` crops the figure out of the page, so the marker has
    # a picture to look at - but the marker has to be a person.
    if item.get("figure"):
        result.update(awarded=None, tier="human",
                      why="the evidence for this item is a drawing")
        return result

    # An empty section is the commonest way to lose a student marks they
    # earned: Part C is answered by choice, so most empty sections are
    # genuine - but a booklet whose label the reader never recovered puts
    # the whole answer under the previous heading, and the section for
    # the real question is empty too. The two look identical from here.
    # Only the rest of the booklet can tell them apart.
    if len(section_text) < MIN_ANSWER_CHARS:
        elsewhere, found = keyword_coverage(item, whole_text)

        if elsewhere is not None and elsewhere >= KW_CONFIDENT:
            result.update(awarded=None, tier="human",
                          why="nothing filed under this question, but the "
                              "booklet answers it somewhere else - the "
                              "label was probably never recovered",
                          coverage_elsewhere=round(elsewhere, 3),
                          keywords_matched=found)
            return result

        result.update(awarded=0.0, tier="unattempted",
                      why="no answer text filed under this question, and "
                          "nothing elsewhere in the booklet answers it")
        return result

    coverage, matched = keyword_coverage(item, section_text)
    result["keyword_coverage"] = (None if coverage is None
                                  else round(coverage, 3))
    result["keywords_matched"] = matched

    if coverage is not None and coverage >= KW_CONFIDENT:
        result.update(awarded=marks, tier="keyword",
                      why=f"named {len(matched)} of "
                          f"{len(item['keywords'])} required terms")
        return result

    similarity, evidence = semantic.best_match(point, sentences(section_text))

    if similarity is not None:
        result["similarity"] = round(similarity, 3)
        result["similar_to"] = evidence

    # Both cheap tiers agree there is nothing here. Before recording a
    # zero, check the rest of the booklet - a strong match elsewhere
    # means the answer exists and was filed under the wrong heading,
    # which is a reconstruction fault and not a student's.
    if (coverage is not None and coverage <= KW_ABSENT
            and (similarity is None or similarity < SEM_ABSENT)):

        elsewhere, _ = keyword_coverage(item, whole_text)

        if elsewhere is not None and elsewhere >= KW_CONFIDENT:
            result.update(awarded=None, tier="human",
                          why="absent from this question but present "
                              "elsewhere in the booklet - likely filed "
                              "under the wrong heading")
            result["coverage_elsewhere"] = round(elsewhere, 3)
            return result

        result.update(awarded=0.0, tier="keyword",
                      why="none of the required terms appear, and no "
                          "sentence is on the subject")
        return result

    if (similarity is not None and similarity >= SEM_CONFIDENT
            and coverage is not None and coverage >= KW_CORROBORATE):
        result.update(awarded=marks, tier="semantic",
                      why=f"named {len(matched)} of "
                          f"{len(item['keywords'])} required terms, and a "
                          f"sentence makes the claim in other words")
        return result

    # Still ambiguous. Record what the cheap tiers thought so the LLM
    # tier has something to start from, but do not bank it.
    blend = (coverage or 0.0) * 0.5 + ((similarity or 0.0) * 0.5)
    result.update(awarded=None, tier="llm",
                  provisional=round(marks * min(blend, 1.0), 2),
                  why="keyword coverage and similarity both inconclusive")
    return result


def grade_booklet(path, keys, semantic):

    name = path.name
    match = BOOKLET_NAME.match(name)
    if not match:
        return None

    student, cie = int(match.group(1)), int(match.group(2))
    key = keys.get(cie)
    if key is None:
        return None

    md = path / "booklet.md"
    if not md.exists():
        return None

    sections, figures, failed, whole = parse_booklet(md)
    whole_norm = normalise(whole)

    questions = []

    for spec in key["questions"]:
        qid = spec["id"]
        raw = sections.get(qid, "")
        section = normalise(raw)

        items = [decide(item, section, whole_norm, semantic)
                 for item in spec["rubric"]]

        settled = sum(i["awarded"] for i in items if i["awarded"] is not None)
        pending = sum(i["marks_available"] for i in items
                      if i["awarded"] is None)

        questions.append({
            "id": qid,
            "part": spec["part"],
            "marks_available": spec["marks"],
            "choice_with": spec.get("choice_with"),
            "attempted": len(section) >= MIN_ANSWER_CHARS,
            "answer_chars": len(section),
            # Kept so the LLM batch is self-contained: whatever runs the
            # third tier gets the answer, the rubric and the marks in one
            # record and needs no access to this repo. Output is
            # gitignored, and this is a student's answer.
            "answer": raw.strip(),
            "question_text": spec["question"],
            "model_solution": spec["solution"],
            # Present only on the multi-step questions, where the marks
            # attach to a chain of values rather than to independent
            # facts. See add_method_marks.py for which and why.
            "method": spec.get("method"),
            "figures": figures.get(qid, []),
            "failed_pages": failed.get(qid, 0),
            "marks_settled": round(settled, 2),
            "marks_pending": round(pending, 2),
            "items": items,
        })

    return {
        "booklet": name,
        "student": student,
        "cie": cie,
        "max_marks": key["max_marks"],
        "questions": questions,
    }


def resolve_choice(report):
    """Part C is answered by choice; count the better of each pair.

    A student who answers both 3a and 3b has not earned twice the marks,
    and the convention is that the better attempt stands. `counted` says
    which one, so a marker can see the discarded attempt was considered
    rather than lost.
    """
    by_id = {q["id"]: q for q in report["questions"]}
    counted, dropped = [], []
    seen = set()

    for question in report["questions"]:
        qid = question["id"]
        if qid in seen:
            continue

        other = question.get("choice_with")

        if not other or other not in by_id:
            counted.append(qid)
            seen.add(qid)
            continue

        pair = [question, by_id[other]]
        seen.update({qid, other})

        attempted = [q for q in pair if q["attempted"]]

        if not attempted:
            counted.append(qid)
            dropped.append(other)
            continue

        # Rank on what is settled plus what is still out, so a question
        # waiting on a human review is not beaten by a weaker one that
        # merely finished scoring first.
        best = max(attempted,
                   key=lambda q: (q["marks_settled"] + q["marks_pending"],
                                  q["answer_chars"]))

        counted.append(best["id"])
        dropped.extend(q["id"] for q in pair if q["id"] != best["id"])

    report["counted"] = counted
    report["dropped"] = dropped

    settled = sum(by_id[q]["marks_settled"] for q in counted)
    pending = sum(by_id[q]["marks_pending"] for q in counted)

    report["marks_settled"] = round(settled, 2)
    report["marks_pending"] = round(pending, 2)
    report["marks_max"] = round(
        sum(by_id[q]["marks_available"] for q in counted), 2)

    return report


def write_summary(reports):
    """summary.csv, one row per booklet, sorted the way a marker reads it."""

    rows = [{
        "booklet": r["booklet"],
        "student": r["student"],
        "cie": r["cie"],
        "counted": " ".join(r["counted"]),
        "marks_settled": r["marks_settled"],
        "marks_pending": r["marks_pending"],
        "marks_max": r["marks_max"],
    } for r in sorted(reports, key=lambda r: (r["student"], r["cie"]))]

    with open(OUT_DIR / "summary.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def recount(report):
    """Refresh each question's settled/pending after items changed."""

    for question in report["questions"]:
        items = question["items"]
        question["marks_settled"] = round(
            sum(i["awarded"] for i in items if i["awarded"] is not None), 2)
        question["marks_pending"] = round(
            sum(i["marks_available"] for i in items
                if i["awarded"] is None), 2)

    return resolve_choice(report)


def apply_verdicts(path):
    """Fold the LLM tier's verdicts back into the reports.

    The quote rule is enforced here as well as in the notebook, because
    this is the side that decides a student's mark and it must not depend
    on a prompt having been obeyed. A verdict that awards marks without
    quoting the student's own words is not a low-confidence award, it is
    an unsupported one, and it goes to a person instead.
    """
    verdicts = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]

    grouped = {}
    for verdict in verdicts:
        grouped.setdefault(verdict["booklet"], []).append(verdict)

    applied = unsupported = abstained = missing = 0
    reports = []

    for report_path in sorted(OUT_DIR.glob("student_*.json")):

        with open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)

        by_id = {q["id"]: q for q in report["questions"]}

        for verdict in grouped.get(report["booklet"], []):

            question = by_id.get(verdict["question"])
            index = verdict.get("item")

            if question is None or index is None or index >= len(
                    question["items"]):
                missing += 1
                continue

            item = question["items"][index]

            # Only items the cheap tiers left open may be overwritten. A
            # verdict arriving for a settled item means the queue and the
            # reports have drifted apart, and silently applying it would
            # hide that.
            if item["awarded"] is not None:
                missing += 1
                continue

            awarded = verdict.get("awarded")
            quote = (verdict.get("quote") or "").strip()
            note = (verdict.get("note") or "").strip()

            item["llm_note"] = note
            item["llm_quote"] = quote

            if awarded is None:
                item.update(tier="human",
                            why="the LLM tier could not tell either")
                abstained += 1
                continue

            try:
                awarded = float(awarded)
            except (TypeError, ValueError):
                item.update(tier="human",
                            why="the LLM tier returned an unreadable mark")
                abstained += 1
                continue

            if awarded > 0 and not quote:
                item.update(tier="human",
                            why="the LLM awarded marks without quoting the "
                                "student's answer, so the award is "
                                "unsupported")
                unsupported += 1
                continue

            item.update(awarded=max(0.0, min(awarded,
                                             item["marks_available"])),
                        tier="llm",
                        why=note or "decided by the LLM tier")
            applied += 1

        recount(report)

        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

        reports.append(report)

    write_summary(reports)

    print(f"Verdicts read       : {len(verdicts)}")
    print(f"Applied             : {applied}")
    print(f"Abstained -> human  : {abstained}")
    print(f"Unsupported -> human: {unsupported}"
          + ("   <-- marks claimed with no quote" if unsupported else ""))
    if missing:
        print(f"Unmatched           : {missing}   "
              f"(already settled, or no such item)")

    settled = sum(r["marks_settled"] for r in reports)
    pending = sum(r["marks_pending"] for r in reports)
    print(f"\nAcross {len(reports)} booklets: {settled:.1f} marks settled, "
          f"{pending:.1f} still pending")
    print(f"Rewrote {OUT_DIR / 'summary.csv'}")

    return 0


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", type=Path, metavar="VERDICTS",
                        help="fold an LLM tier verdicts.jsonl back into the "
                             "reports written by an earlier run")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--booklet")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-semantic", action="store_true",
                        help="keyword tier only; everything the keywords "
                             "cannot settle goes to the LLM queue")
    args = parser.parse_args()

    if args.apply:
        if not args.apply.exists():
            raise SystemExit(f"no such file: {args.apply}")
        if not OUT_DIR.is_dir():
            raise SystemExit(f"{OUT_DIR} not found - run --all first")
        return apply_verdicts(args.apply)

    if not args.all and not args.booklet:
        raise SystemExit("--all, --booklet or --apply is required")

    if not BOOKLETS.is_dir():
        raise SystemExit(f"{BOOKLETS} not found - run build_booklet.py first")

    keys = load_keys()

    semantic = Semantic(enabled=not args.no_semantic)
    if semantic.ready:
        print(f"semantic tier : {MODEL_NAME}")
    else:
        print(f"semantic tier : OFF ({semantic.reason})")
    print()

    targets = ([BOOKLETS / args.booklet] if args.booklet
               else sorted(p for p in BOOKLETS.iterdir() if p.is_dir()))

    if args.limit:
        targets = targets[:args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, llm_queue, human_queue = [], [], []
    tier_counts = {}

    for path in targets:

        report = grade_booklet(path, keys, semantic)
        if report is None:
            print(f"skipped {path.name}")
            continue

        resolve_choice(report)

        with open(OUT_DIR / f"{report['booklet']}.json", "w",
                  encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)

        for question in report["questions"]:

            # The LLM tier is grouped by question, not by item. Every
            # pending item in a question is judged against the same
            # answer, so sending them together lets one call read the
            # answer once and mark the whole rubric - and lets the model
            # see that the marks it awards have to add up.
            pending_items = []

            for index, item in enumerate(question["items"]):
                tier_counts[item["tier"]] = tier_counts.get(item["tier"], 0) + 1

                if item["awarded"] is not None:
                    continue

                if item["tier"] == "human":
                    human_queue.append({
                        "booklet": report["booklet"],
                        "question": question["id"],
                        "item": index,
                        "point": item["point"],
                        "marks_available": item["marks_available"],
                        "why": item["why"],
                        "counted": question["id"] in report["counted"],
                        "figures": question["figures"],
                        "failed_pages": question["failed_pages"],
                        # Carried so a marker chasing a misfiled answer
                        # knows what to search the booklet for.
                        "coverage_elsewhere": item.get("coverage_elsewhere"),
                        "keywords_matched": item.get("keywords_matched"),
                    })
                else:
                    pending_items.append({
                        "item": index,
                        "point": item["point"],
                        "marks_available": item["marks_available"],
                        "keyword_coverage": item.get("keyword_coverage"),
                        "similarity": item.get("similarity"),
                        "provisional": item.get("provisional"),
                    })

            if pending_items:
                llm_queue.append({
                    "booklet": report["booklet"],
                    "cie": report["cie"],
                    "question": question["id"],
                    "counted": question["id"] in report["counted"],
                    "marks_available": question["marks_available"],
                    "marks_pending": round(
                        sum(i["marks_available"] for i in pending_items), 2),
                    "question_text": question["question_text"],
                    "model_solution": question["model_solution"],
                    "method": question["method"],
                    "answer": question["answer"],
                    "items": pending_items,
                })

        rows.append({
            "booklet": report["booklet"],
            "student": report["student"],
            "cie": report["cie"],
            "counted": " ".join(report["counted"]),
            "marks_settled": report["marks_settled"],
            "marks_pending": report["marks_pending"],
            "marks_max": report["marks_max"],
        })

        if args.verbose:
            print(f"\n=== {report['booklet']} "
                  f"({report['marks_settled']} settled, "
                  f"{report['marks_pending']} pending of "
                  f"{report['marks_max']}) ===")
            for question in report["questions"]:
                mark = "*" if question["id"] in report["counted"] else " "
                print(f" {mark}{question['id']:<4}"
                      f"{question['marks_settled']:>6} settled"
                      f"{question['marks_pending']:>6} pending"
                      f"   of {question['marks_available']}"
                      + ("" if question["attempted"] else "   (unattempted)"))

    with open(OUT_DIR / "summary.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for name, queue in (("review_llm.jsonl", llm_queue),
                        ("review_human.jsonl", human_queue)):
        with open(OUT_DIR / name, "w", encoding="utf-8") as handle:
            for entry in queue:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_items = sum(tier_counts.values())

    print(f"\nBooklets graded : {len(rows)}")
    print(f"Rubric items    : {total_items}")
    print(f"\n{'tier':<14}{'items':>7}{'share':>9}")
    for tier in ("keyword", "semantic", "unattempted", "llm", "human"):
        count = tier_counts.get(tier, 0)
        if not count:
            continue
        print(f"{tier:<14}{count:>7}{count / total_items:>8.1%}")

    decided = sum(tier_counts.get(t, 0)
                  for t in ("keyword", "semantic", "unattempted"))
    print(f"\nDecided without review : {decided / total_items:.1%}")
    llm_items = sum(len(g["items"]) for g in llm_queue)
    print(f"Queued for the LLM     : {llm_items} items in {len(llm_queue)} calls")
    print(f"Queued for a human     : {len(human_queue)}")
    print(f"\nWritten to {OUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
