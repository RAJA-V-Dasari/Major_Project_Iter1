"""
Tier 2 - the student made the claim, but not in the key's words.

An embedding model compares each rubric point against the student's own
sentences and reports the best match. It runs on CPU (MiniLM, ~90MB) and
costs a couple of minutes for the whole corpus, so it needs no GPU and
no API.

SIMILARITY MAY CONFIRM EVIDENCE. IT MAY NEVER SUPPLY IT.
--------------------------------------------------------
This is the load-bearing rule in the whole ladder, and it is a
constraint against a specific, predictable failure.

Sentence embeddings measure what an answer is *about*, not whether it is
right. Every wrong subnet is about subnetting; every mangled handshake
is about handshakes. Set loose on the CIE-2 3a rubric, a single sentence
of plausible-sounding subnetting prose will score respectably against
all five rubric items at once - including the four it says nothing
about - because all five are on the same topic. Worse, nothing stops one
sentence being counted as the evidence for every item in a question.

So an item whose keywords are absent does NOT get decided here no matter
how topical it reads. Similarity is only allowed to settle an item that
already has real keyword support and is sitting in the band where the
count alone could not call it. The cost is a larger queue for the tier
above; the alternative is a grader that hands out marks for being on
subject, which is the one failure a marking system may not have.

THIS TIER NEVER RECORDS A ZERO
------------------------------
It can confirm that a claim was made in other words. It cannot
demonstrate that a claim was absent - low similarity might mean the
student phrased it unusually, or that the sentence splitter cut the
statement in half. Absence is the keyword tier's call, or a human's.

Optional by construction: `grade.py --no-semantic` must keep working, so
the import is lazy and its absence degrades the ladder rather than
breaking it.
"""

import re

from . import DEFAULT, exact

# The two numbers this module runs on - the similarity floor and the
# keyword support an item needs before similarity may confirm it - are
# `sem_confident` and `kw_corroborate` on `Thresholds`, not constants
# here.
#
# They lived here once, as SEM_CONFIDENT = 0.62 and KW_CORROBORATE =
# 0.50, and stayed after `decide()` moved to reading `th`. Nothing
# imported them and nothing updated them, so while the tier ran on 0.67
# the module still said 0.50 - and the comment above the stale one
# called it "the rule, not a knob", which is exactly the line a reader
# trusts. Anyone working out why this tier fires on so few items would
# have reasoned from a number no code uses.
#
# A threshold with two homes has one that is wrong. See tiers/__init__.

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Split on sentence enders, but also on the numbered/bulleted list
# markers students actually write - "1.", "(2)", "b)" - because an
# answer to a five-part question is usually a list, and one 400-word
# "sentence" defeats the comparison entirely.
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|\s(?=\(?\d{1,2}[.)])|\s(?=[a-d]\))")

_MODEL = None


def available():
    """Can this tier run at all?"""

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def sentences(text, minimum=15):
    """The student's answer as comparable units."""

    parts = [s.strip() for s in _SPLIT.split(text or "") if s]
    return [s for s in parts if len(s) >= minimum]


class Semantic:
    """Holds the model and the per-answer sentence embeddings."""

    def __init__(self):
        self.model = _model()
        self._cache = {}

    def _embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)

    def best_match(self, point, answer):
        """(similarity, sentence) for the answer's closest sentence."""

        pieces = self._cache.get(answer)
        if pieces is None:
            found = sentences(answer)
            pieces = self._cache[answer] = (
                (found, self._embed(found)) if found else ([], None)
            )
        found, vectors = pieces
        if not found:
            return 0.0, None

        point_vector = self._embed([point])[0]
        scores = vectors @ point_vector
        index = int(scores.argmax())
        return float(scores[index]), found[index]

    def decide(self, item, text, answer, *, no_zero=None, th=DEFAULT,
               similarity=None, quote=None):
        """Confirm an item that already has keyword support, or pass."""

        coverage = exact.keyword_coverage(item, text)
        if coverage is None or coverage < th.kw_corroborate:
            # No real evidence to corroborate. This is the rule.
            return None

        # The same rule applied to exact values, which it originally was
        # not. If an item names concrete things - 256, 1110, 57088 - and
        # the student named NONE of them, similarity must not rescue it.
        # Measured: student_23_cie_3 wrote 265 characters about Fletcher
        # checksums without one number from the scheme, and this tier
        # awarded 3 of 3 on topical similarity while the examiner gave
        # zero. Exact values are the substance; being on the subject is
        # not a substitute for them.
        values = item.get("exact") or []
        if values and not any(exact.contains(text, v) for v in values):
            return None

        if similarity is None:
            similarity, quote = self.best_match(item["point"], answer)
        sentence = quote
        if similarity < th.sem_confident:
            return None

        return {
            "awarded": item["marks"],
            "tier": "semantic",
            "why": f"keyword coverage {coverage:.0%} corroborated by "
                   f"similarity {similarity:.2f}",
            "evidence": exact.matched_keywords(item, text),
            "quote": sentence,
            "similarity": round(similarity, 4),
        }
