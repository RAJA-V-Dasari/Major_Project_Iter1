"""
Tier 1 - the answers that are right or wrong with no interpretation.

A rubric item carrying `exact` values names something that either
appears in the student's answer or does not: `57088`, `14.24.74.126`,
`/26`, a CRC remainder of `1110`. No model is consulted about those, and
none should be: a string comparison settles a final numeric answer more
reliably than any model would, and it can explain itself completely.

THE LITERAL IS NOT ENOUGH ON ITS OWN
------------------------------------
Some correct answers cannot be made distinctive. CIE-1 4b has to accept
`88` and `80` - they are the actual answers - and on their own they
would fire on any student who mentioned port 80. So this tier awards
only when **every `exact` value is present AND the item's keyword groups
are satisfied**: `88` counts when it arrives next to "length", not when
it arrives next to anything at all.

`validate_keys.py` reports every short numeric `exact` as a warning so
each one is a value a person decided to trust.

WHEN THIS TIER MAY RECORD A ZERO
--------------------------------
Awarding on evidence and zeroing on the absence of evidence are not
symmetric, and two cases are carved out:

  * **Chain questions may be awarded but never zeroed here.** On the
    subnetting, fragmentation, CRC and delay-cascade questions the marks
    hang off a sequence, and a student who took a wrong block size at
    step one produces correct-but-different values everywhere after it.
    Their `14.24.74.63` is absent from the key not because they failed
    but because they are consistently following their own arithmetic.
    Zeroing that is charging one slip five times; those items go up the
    ladder instead, where carry-forward can be judged.

  * **A part containing a `gap` is incomplete, not wrong.** One page of
    the corpus was lost before it was ever read. An item whose evidence
    might have been on that page is a human's decision, not a zero.

  * **A section whose evidence is a drawing cannot be zeroed on prose.**
    Measured on student_01_cie_2: the whole of question 2b's routing
    table lives in four crops behind 73 characters of prose, and half of
    question 1's answer was written onto the handshake diagram itself.
    The examiner gave 5/5 for both; scoring the prose alone gave 0 and
    2. Nothing below the human tier reads pixels, so absence of the
    claim in the text is not evidence of absence in the answer.

Everything else that names concrete things and names none of them is a
zero this tier is entitled to record.
"""

import re
import unicodedata

from . import DEFAULT

# Keyword coverage above this means the student named essentially
# everything the item asks for; below the low mark they named almost
# none of it. Between the two, the count is not evidence either way.
#
# PROVISIONAL. These are starting values, not measured ones. They are
# calibrated against the examiner's marks in step 7 (agreement.py) and
# the number that justifies them belongs there, not here.
KW_CONFIDENT = 0.75
KW_ABSENT = 0.20

_WHITESPACE = re.compile(r"\s+")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def normalise(text):
    """Lowercase, collapse whitespace, flatten the dash zoo.

    Punctuation is deliberately KEPT. The values this tier matches are
    things like `14.24.74.126`, `/26`, `image/gif` and `2^8` - stripping
    punctuation would destroy exactly the evidence it exists to find.
    """

    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_DASHES)
    return _WHITESPACE.sub(" ", text.lower()).strip()


def _pattern(term, stem):
    """Match `term` on word boundaries, so 8001 never hits inside 18001.

    A boundary is only applied where the term's own edge is
    alphanumeric: `/26` must be allowed to follow a digit, and
    `14.24.74.1` must be allowed to precede the `/` in `14.24.74.1/24`
    while still being blocked inside `14.24.74.126`.

    THE RIGHT-HAND BOUNDARY DIFFERS BY WHAT IS BEING MATCHED
    --------------------------------------------------------
    `exact` values are whole answers and are matched strictly at both
    ends. Keywords are not: the keys deliberately carry stems -
    "acknowledg", "reliab", "discard", "translat", "synchroniz" - chosen
    so one entry covers acknowledgment/acknowledged/acknowledges. A
    strict right boundary blocks every one of those, which would zero
    keyword items wholesale and silently.

    So for keywords the right boundary is dropped, EXCEPT where the term
    ends in a digit. That exception matters: keyword groups contain
    numbers too, and a stemming "80" would happily match inside "8080".
    Letters stem; digits do not.
    """

    term = normalise(term)
    if not term:
        return None
    left = r"(?<![a-z0-9])" if term[0].isalnum() else ""
    if stem and not term[-1].isdigit():
        right = ""
    else:
        right = r"(?![a-z0-9])" if term[-1].isalnum() else ""
    return re.compile(left + re.escape(term) + right)


_CACHE = {}


def contains(text, term, stem=False):
    """Is `term` present in already-normalised `text`?

    `stem=True` lets a keyword stem match the rest of its word. Use it
    for `keywords`, never for `exact`.
    """

    key = (term, stem)
    pattern = _CACHE.get(key)
    if pattern is None:
        pattern = _CACHE[key] = _pattern(term, stem)
    return bool(pattern and pattern.search(text))


def keyword_coverage(item, text):
    """Fraction of the item's keyword groups that the answer satisfies.

    A group is a set of alternates - ["ack", "acknowledg"] - and counts
    as satisfied if ANY of them appears. Coverage is then the fraction
    of groups satisfied, so an item needing both an ack spelling and the
    number 8000 scores 0.5 when only one is there.
    """

    groups = item.get("keywords") or []
    if not groups:
        return None
    hits = sum(
        1 for group in groups
        if any(contains(text, alternate, stem=True) for alternate in group)
    )
    return hits / len(groups)


def exact_coverage(item, text):
    """(found, total) over the item's exact values."""

    values = item.get("exact") or []
    if not values:
        return None
    found = [v for v in values if contains(text, v)]
    return len(found), len(values), found


def matched_keywords(item, text):
    """Which alternates actually matched - the evidence for a report."""

    found = []
    for group in item.get("keywords") or []:
        for alternate in group:
            if contains(text, alternate, stem=True):
                found.append(alternate)
                break
    return found


def decide(item, text, *, no_zero=None, th=DEFAULT):
    """Award, zero, or pass up the ladder.

    `no_zero` is a reason string when this tier is forbidden from
    recording a zero for this item - a chain question, a lost page, or a
    section whose evidence is a drawing. It may still AWARD: finding the
    evidence is positive proof, failing to find it is not.

    THE FOURTH CARVE-OUT LIVES HERE, NOT IN grade.py
    ------------------------------------------------
    `grade.py` passes three of the four reasons a zero is forbidden. The
    fourth - a long answer with no keyword overlap - is added below,
    because it is the only one that depends on the text this tier is
    about to read rather than on the question's metadata. Anyone
    auditing `grade_item()` for the carve-outs will find three and
    should come here for the fourth.

    Returns a dict with `awarded` (a number, or None to escalate), the
    tier that decided, and the evidence - or None if this tier has
    nothing to say.
    """

    if no_zero is None and len(text) > th.zero_max_chars:
        no_zero = (f"the answer runs to {len(text)} characters - too much "
                   "for keyword silence to mean the claim is absent")

    marks = item["marks"]
    coverage = keyword_coverage(item, text)
    exact = exact_coverage(item, text)

    # --- items with exact values: this tier's strongest case ---------
    if exact is not None:
        found, total, hits = exact
        keywords_ok = coverage is None or coverage >= th.kw_confident

        if found == total and keywords_ok:
            return {
                "awarded": marks,
                "tier": "exact",
                "why": f"all {total} exact value(s) present"
                       + ("" if coverage is None
                          else f", keyword coverage {coverage:.0%}"),
                "evidence": hits + matched_keywords(item, text),
            }

        if found == 0 and (coverage is None or coverage <= th.kw_absent):
            if no_zero:
                return {
                    "awarded": None,
                    "tier": "exact",
                    "why": f"no exact value matched, but {no_zero} - "
                           "not a zero this tier may record",
                    "evidence": [],
                }
            return {
                "awarded": 0.0,
                "tier": "exact",
                "why": f"none of the {total} exact value(s) present"
                       + ("" if coverage is None
                          else f", keyword coverage {coverage:.0%}"),
                "evidence": [],
            }

        return None  # partial - the ladder decides

    # --- items with keywords only ------------------------------------
    if coverage is None:
        return None

    if coverage >= th.kw_confident:
        return {
            "awarded": marks,
            "tier": "keyword",
            "why": f"keyword coverage {coverage:.0%}",
            "evidence": matched_keywords(item, text),
        }

    if coverage <= th.kw_absent:
        if no_zero:
            return {
                "awarded": None,
                "tier": "keyword",
                "why": f"keyword coverage {coverage:.0%}, but {no_zero} - "
                       "not a zero this tier may record",
                "evidence": [],
            }
        return {
            "awarded": 0.0,
            "tier": "keyword",
            "why": f"keyword coverage {coverage:.0%}",
            "evidence": [],
        }

    return None
