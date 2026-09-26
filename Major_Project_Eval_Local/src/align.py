"""
Resolve the parts that carry no usable question label.

    handoff + keys  ->  output/alignment.md      what to confirm
                    ->  output/alignment.json    machine-readable decisions

WHAT THIS TURNED OUT TO BE
--------------------------
Part 1's README flags 35 of 325 parts as unlabelled and recommends we
align them against the question paper, since we hold the paper and they
do not. That framing assumed the unlabelled parts were answers whose
question number could not be read.

They are not. All 35 are the section heading the student wrote at the
top of a page - "PART - A", "Part - B", "INTERNALS - II PART - A:-" -
captured by the reader as a leading fragment before the first question
marker on the page. Every one of them is 25 characters or shorter, and
not one carries a figure or a gap.

So there is no alignment problem for them and no content at risk: the
11% of parts that are unlabelled is **0% of the marks**. They are
classified as structural headers and dropped, and the evidence for that
is printed rather than asserted - see `--report`.

What does need resolving is smaller and different: five parts carry a
label that is not a question id on the paper ("2", "3", "1a"). Those
hold real answers, and they are resolved here.

HOW A LABEL IS RESOLVED
-----------------------
Three independent signals, and a proposal is only called confident when
they agree:

  structure  a bare "2" must be a part of question 2; "1a" on a paper
             whose question 1 has no parts must be question 1. This
             narrows the candidates before any content is read.
  content    overlap between the answer and each candidate question's
             distinctive vocabulary, weighted by how rare each term is
             across the eight questions of that paper.
  examiner   did the faculty award marks against that question on the
             cover? An answer we place at 3a is more believable if the
             examiner marked 3a. This signal is INDEPENDENT of anything
             in the handoff, which is what makes it worth having.

Nothing is assigned automatically. Every proposal is written out for a
person to confirm, because a confident wrong question number attaches a
student's work to the wrong rubric - exactly the failure part 1 refused
to risk when it left these unlabelled.

Run:
    python src/align.py --report
    python src/align.py            # write the proposals
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_handoff as lh
import paths

# A part is a structural header if it is this short and says so. Both
# conditions are required: "PART - A" is a heading, but a 900-character
# answer that happens to begin with the word "part" is not.
HEADER_MAX_CHARS = 40
HEADER_WORDS = re.compile(r"\b(part|cie|internals?)\b", re.I)
HEADER_SHAPE = re.compile(
    r"^[\s\-.:]*(internals?\s*[-–]?\s*[ivx0-9]*)?[\s\-.:]*"
    r"(cie\s*[-–]?\s*[0-9]*)?[\s\-.:]*"
    r"part\s*[-–.\s]*[abc]?[\s\-.:]*$",
    re.I,
)

TOKEN = re.compile(r"[a-z0-9][a-z0-9./]*")
STOP = {
    "the", "a", "an", "is", "are", "of", "to", "and", "in", "for", "it",
    "that", "this", "be", "on", "as", "with", "by", "from", "at", "or",
    "we", "can", "will", "not", "if", "so", "its", "which", "each",
    "what", "how", "why", "when", "then", "there", "their", "they",
}

# A proposal is confident when the winner clears this score and beats
# the runner-up by this margin. Below either, a person decides.
MIN_SCORE = 0.08
MIN_MARGIN = 0.35


def normalise(text):
    return [t for t in TOKEN.findall((text or "").lower()) if t not in STOP]


def is_header(part):
    """True if this part is a page heading rather than an answer."""

    text = part.prose.strip()
    if part.diagrams or part.gaps:
        return False
    if len(text) > HEADER_MAX_CHARS:
        return False
    if not HEADER_WORDS.search(text):
        return False
    # Either it matches the heading shape exactly, or it is so short
    # that it cannot be an answer to anything.
    return bool(HEADER_SHAPE.match(text)) or len(text) <= 12


def signatures(key):
    """{question id: Counter(term -> weight)} for one paper.

    Terms are drawn from the question, the model solution, each rubric
    point, and the rubric's own keywords and exact values. Keywords and
    exact values are weighted up: they were chosen by hand as the things
    that identify this question, which is precisely what is needed here.
    """

    raw = {}
    for spec in key["questions"]:
        counts = Counter()
        for term in normalise(spec["question"]):
            counts[term] += 1
        for term in normalise(spec["solution"]):
            counts[term] += 1
        for item in spec["rubric"]:
            for term in normalise(item["point"]):
                counts[term] += 2
            for group in item.get("keywords") or []:
                for alternate in group:
                    for term in normalise(alternate):
                        counts[term] += 4
            for value in item.get("exact") or []:
                for term in normalise(value):
                    counts[term] += 6
        raw[spec["id"]] = counts

    # Down-weight vocabulary shared across the whole paper: every
    # question on a networks paper says "packet".
    document_frequency = Counter()
    for counts in raw.values():
        for term in counts:
            document_frequency[term] += 1

    total = len(raw)
    weighted = {}
    for qid, counts in raw.items():
        weighted[qid] = {
            term: count * math.log(total / document_frequency[term] + 1)
            for term, count in counts.items()
        }
    return weighted


def score(part_terms, signature):
    """Cosine-ish overlap of an answer against a question's vocabulary."""

    if not part_terms or not signature:
        return 0.0
    hit = sum(signature.get(term, 0.0) for term in set(part_terms))
    norm = math.sqrt(sum(v * v for v in signature.values())) or 1.0
    return hit / norm


def candidates_from_label(label, valid_ids):
    """Narrow by structure before reading a single word of the answer."""

    if label in valid_ids:
        return [label]

    # "2" -> every part of question 2 on this paper.
    if label.isdigit():
        return [q for q in valid_ids if q[0] == label]

    # "1a" where the paper's question 1 has no parts -> "1".
    head = label[0]
    same = [q for q in valid_ids if q[0] == head]
    if same == [head]:
        return same
    return same or list(valid_ids)


def load_gold():
    if not paths.GOLD_CSV.exists():
        return {}
    with open(paths.GOLD_CSV, encoding="utf-8", newline="") as handle:
        return {r["booklet_id"]: r for r in csv.DictReader(handle)}


def examiner_marked(gold_row, qid):
    """Did the faculty award anything against this question?

    Returns True / False / None (no usable gold).

    Part C is checked at the ROW level, not the column level: the
    examiner uses the grid's a/b/c columns loosely there - one cover
    records question 3 in columns b and c on a paper whose question 3
    has no part c. Parts A and B track the question structure, so the
    cell is used for those.
    """

    if not gold_row or gold_row.get("confidence") == "no_gold":
        return None

    number = qid[0]
    if number in ("3", "4"):
        value = gold_row.get(f"q{number}t", "")
        if not value.strip():
            # The row total may be blank on a cover the examiner never
            # totalled; fall back to any part cell in that row.
            value = "".join(gold_row.get(f"q{number}{c}", "")
                            for c in "abcd")
        return bool(value.strip())

    letter = qid[1:] or "a"
    return bool((gold_row.get(f"q{number}{letter}") or "").strip())


def resolve(booklets, keys, gold):
    headers, proposals = [], []

    for booklet in booklets:
        key = keys[booklet.cie]
        valid = [q["id"] for q in key["questions"]]
        signature = signatures(key)
        gold_row = gold.get(booklet.booklet_id)

        claimed = {p.label for p in booklet.labelled if p.label in valid}

        for part in booklet.parts:
            label = part.label

            if label in valid:
                continue

            if is_header(part):
                headers.append((booklet, part))
                continue

            options = candidates_from_label(label or "", valid)
            terms = normalise(part.prose)

            ranked = sorted(
                ((score(terms, signature[q]), q) for q in options),
                reverse=True,
            )
            best_score, best = ranked[0] if ranked else (0.0, None)
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            margin = (best_score - runner_up) / best_score if best_score else 0.0

            supported = examiner_marked(gold_row, best) if best else None

            confident = (
                best is not None
                and (len(options) == 1
                     or (best_score >= MIN_SCORE and margin >= MIN_MARGIN))
            )

            proposals.append({
                "booklet_id": booklet.booklet_id,
                "cie": booklet.cie,
                "label": label,
                "raw_marker": part.raw_marker,
                "proposed": best,
                "score": round(best_score, 4),
                "margin": round(margin, 3),
                "options": [q for _, q in ranked],
                "already_claimed": best in claimed,
                "examiner_marked": supported,
                "confident": confident,
                "chars": len(part.prose),
                "figures": len(part.diagrams),
                "pages": part.pages,
                "excerpt": part.prose[:220],
            })

    return headers, proposals


def report(headers, proposals):
    print(f"structural headers dropped: {len(headers)}")
    lengths = [len(p.prose.strip()) for _, p in headers]
    if lengths:
        print(f"  longest {max(lengths)} chars, "
              f"figures {sum(len(p.diagrams) for _, p in headers)}, "
              f"gaps {sum(p.gaps for _, p in headers)}")
        print("  -> no answer content, so no marks are at risk")
    print()

    print(f"labels needing resolution: {len(proposals)}")
    for row in proposals:
        agree = {True: "yes", False: "NO", None: "n/a"}[row["examiner_marked"]]
        mark = "  " if row["confident"] else "??"
        merge = "  (merges into an existing part)" if row["already_claimed"] else ""
        print(f"  {mark} {row['booklet_id']:20} {str(row['label']):>4} "
              f"-> {row['proposed']:3}  score={row['score']:.3f} "
              f"margin={row['margin']:.2f}  examiner marked it: {agree}"
              f"{merge}")
    print()

    confident = sum(1 for r in proposals if r["confident"])
    agreed = sum(1 for r in proposals if r["examiner_marked"] is True)
    print(f"{confident}/{len(proposals)} confident on structure and content")
    print(f"{agreed}/{len(proposals)} independently corroborated by the "
          "examiner's grid")


def render(headers, proposals):
    out = ["# Alignment: parts with no usable question label", ""]
    out.append("Generated by `src/align.py`. Confirm each proposal below")
    out.append("against the booklet before it is applied - a confident wrong")
    out.append("question number attaches a student's work to the wrong")
    out.append("rubric, which is the one failure part 1 refused to risk.")
    out.append("")

    out.append("## Structural headers (dropped, no action needed)")
    out.append("")
    out.append(f"{len(headers)} parts carry a page heading and nothing else.")
    out.append("Part 1 flagged these as unlabelled because no question marker")
    out.append("preceded them; they are the \"PART - A\" the student wrote at")
    out.append("the top of the page. None carries a figure or a gap, and the")
    out.append(f"longest is "
               f"{max((len(p.prose.strip()) for _, p in headers), default=0)}"
               " characters.")
    out.append("")
    out.append("| booklet | text |")
    out.append("|---|---|")
    for booklet, part in headers:
        out.append(f"| `{booklet.booklet_id}` | `{part.prose.strip()}` |")
    out.append("")

    out.append("## Labels to confirm")
    out.append("")
    out.append("| ok | booklet | label | proposed | score | margin | "
               "examiner marked it | note |")
    out.append("|---|---|---|---|---|---|---|---|")
    for row in proposals:
        agree = {True: "yes", False: "**no**", None: "n/a"}[
            row["examiner_marked"]]
        note = []
        if row["already_claimed"]:
            note.append("merges into an existing part")
        if not row["confident"]:
            note.append("**low confidence**")
        out.append(
            f"| [ ] | `{row['booklet_id']}` | `{row['label']}` | "
            f"**`{row['proposed']}`** | {row['score']:.3f} | "
            f"{row['margin']:.2f} | {agree} | {', '.join(note) or '-'} |"
        )
    out.append("")

    for row in proposals:
        out.append(f"### `{row['booklet_id']}` — label `{row['label']}` "
                   f"→ `{row['proposed']}`")
        out.append("")
        out.append(f"Pages {', '.join(row['pages'])} · {row['chars']} chars "
                   f"· {row['figures']} figure(s)")
        out.append("")
        out.append(f"> {row['excerpt']}...")
        out.append("")
        out.append(f"Ranked candidates: "
                   f"{', '.join(f'`{q}`' for q in row['options'])}")
        out.append("")

    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    keys = {}
    for path in sorted(paths.KEYS_DIR.glob("cie*.json")):
        with open(path, encoding="utf-8") as handle:
            key = json.load(handle)
        keys[int(key["cie"])] = key

    _, booklets = lh.load_all()
    gold = load_gold()

    headers, proposals = resolve(booklets, keys, gold)

    if args.report:
        report(headers, proposals)
        return

    paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (paths.OUT_DIR / "alignment.md").write_text(
        render(headers, proposals), encoding="utf-8")
    with open(paths.OUT_DIR / "alignment.json", "w", encoding="utf-8") as handle:
        json.dump({
            "headers": [
                {"booklet_id": b.booklet_id, "text": p.prose.strip()}
                for b, p in headers
            ],
            "proposals": proposals,
        }, handle, indent=2)

    print(f"{len(headers)} headers, {len(proposals)} proposals")
    print(f"  {paths.OUT_DIR / 'alignment.md'}")
    print(f"  {paths.OUT_DIR / 'alignment.json'}")


if __name__ == "__main__":
    main()
