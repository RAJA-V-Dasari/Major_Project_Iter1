"""
The paper's own question numbering, and how to fit marks to it.

THE SCHEMA
----------
Every booklet in this corpus answers the same paper:

    1
    2a  2b  2c
    3a  3b        <- alternatives, a student answers one
    4a  4b        <- alternatives, a student answers one

and under any of those, sub-parts numbered either

    i  ii  iii ...        roman
    a  b  c ...           letter

WHY THIS IS THE WHOLE GAME
--------------------------
Judging each mark on its own was never going to work. TrOCR reads a
1-3 glyph mark badly - measured 2 of 8 correct, with `(1)` coming back
as `0` and `(4b)` as `46` - and geometry cannot tell `Ans:-` from
`Q2b)` because their size distributions overlap completely.

But the marks are not independent. They appear in a KNOWN ORDER, drawn
from a KNOWN eight-item set. That is a far stronger constraint than
anything available per-mark, and it does three jobs at once:

  - a spurious candidate (a star, a stray tick, a word) is rejected
    because it has nowhere to fit in the sequence
  - a garbled reading is recovered from its position - a mark between a
    confident `2a` and a confident `2c` is `2b` whatever OCR said
  - a missed mark shows up as a gap, so it can be reported rather than
    silently dropped

So this module does not ask "is this mark a question number". It asks
"what is the best assignment of these marks, in this order, to this
paper" - and answers with dynamic programming over the two sequences.
"""

import re

# Canonical order. Strictly increasing assignment against this list is
# what makes the alignment work; 3a/3b and 4a/4b being alternatives
# rather than both-answered needs no special case, because skipping a
# label is already allowed.
TOP_LEVEL = ["1", "2a", "2b", "2c", "3a", "3b", "4a", "4b"]

INDEX = {label: i for i, label in enumerate(TOP_LEVEL)}

ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
LETTERS = list("abcdefgh")

# --- costs ----------------------------------------------------------
# All in the same arbitrary unit; only their ratios matter. Set so that
# a confident reading beats position, an unreadable mark still gets
# placed if position is unambiguous, and a word never does.
# THE ONE RELATION THAT MATTERS:
#
#     COST_GARBLED  >  COST_REJECT + COST_SKIP
#
# Placing a candidate costs its match cost. NOT placing it costs
# COST_REJECT for the candidate plus COST_SKIP for the label nothing
# else claims. So if a garbled reading is cheaper than that pair, the
# solver places garbage - and it did: the first tuning here had
# GARBLED 3.0 against REJECT 4.0, and every booklet came back "full",
# with `http`, `vers` and `sea` assigned to real questions.
#
# Set the other way, an unreadable mark is dropped unless something
# else about it is convincing. That is the honest failure: a reported
# gap beats a confidently wrong label, because a gap can be looked at.
COST_EXACT = 0.0          # reading is the label
COST_NUMBER_ONLY = 0.5    # "2" for 2a - right question, lost the subpart
COST_LETTER_ONLY = 1.5    # "a" for 2a - right subpart, lost the question
COST_GARBLED = 5.0        # parsed as something, but not this label
COST_UNREADABLE = 5.0     # OCR returned nothing usable
COST_WORD = 20.0          # reads as prose - effectively forbidden

COST_REJECT = 1.0         # drop a candidate as spurious
COST_SKIP = 2.5           # a question the student did not answer

# Anything longer than this many characters is prose, not a mark.
WORD_LENGTH = 4

_NUMBER_SUBPART = re.compile(r"^([1-4])\s*([a-e])?$")


def split_reading(text):
    """(number, letter) from a normalised reading, or (None, None)."""

    if not text:
        return None, None

    match = _NUMBER_SUBPART.match(text)

    if not match:
        return None, None

    return match.group(1), match.group(2)


def match_cost(reading, label):
    """What it costs to call `reading` an instance of `label`."""

    if reading is None:
        return COST_UNREADABLE

    reading = reading.strip().lower()

    if not reading:
        return COST_UNREADABLE

    if len(reading) > WORD_LENGTH:
        return COST_WORD

    if reading == label:
        return COST_EXACT

    number, letter = split_reading(reading)
    want_number, want_letter = label[0], label[1:] or None

    if number is not None:

        if number == want_number and letter is None:
            # "2" offered for 2a/2b/2c - right question, subpart lost.
            # Deliberately the same cost for all three, so position
            # rather than the reading decides which one it is.
            return COST_NUMBER_ONLY

        if number == want_number and letter == want_letter:
            return COST_EXACT

        # a confident, well-formed reading of a DIFFERENT question is
        # strong evidence against this label
        return COST_GARBLED + 2.0

    if want_letter and reading == want_letter:
        return COST_LETTER_ONLY

    # parsed as nothing useful - a single stray char, punctuation
    return COST_GARBLED


def align(readings, labels=None):
    """Fit candidate readings, in page order, to the paper.

    `readings` is a list of normalised strings (or None where OCR gave
    nothing). Returns a list the same length, holding the assigned
    label or None where the candidate was rejected as spurious.

    Dynamic programme over candidates x labels. Three moves: assign,
    reject a candidate, skip a label. Assignment is strictly increasing
    in both, which is what enforces the paper's order.
    """

    labels = labels or TOP_LEVEL

    n, m = len(readings), len(labels)

    if n == 0:
        return []

    infinity = float("inf")

    # dp[i][j] - best cost having consumed i candidates and j labels
    dp = [[infinity] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):

            here = dp[i][j]
            if here == infinity:
                continue

            # reject candidate i
            if i < n:
                cost = here + COST_REJECT
                if cost < dp[i + 1][j]:
                    dp[i + 1][j] = cost
                    back[i + 1][j] = (i, j, None)

            # skip label j
            if j < m:
                cost = here + COST_SKIP
                if cost < dp[i][j + 1]:
                    dp[i][j + 1] = cost
                    back[i][j + 1] = (i, j, "skip")

            # assign candidate i to label j
            if i < n and j < m:
                cost = here + match_cost(readings[i], labels[j])
                if cost < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = cost
                    back[i + 1][j + 1] = (i, j, labels[j])

    # best end state: all candidates consumed, any number of labels used
    best_j = min(range(m + 1), key=lambda j: dp[n][j])

    assigned = [None] * n

    i, j = n, best_j

    while back[i][j] is not None:
        pi, pj, move = back[i][j]
        if move is not None and move != "skip":
            assigned[pi] = move
        i, j = pi, pj

    return assigned


def missing(assigned, labels=None):
    """Labels the paper expects that no candidate claimed."""

    labels = labels or TOP_LEVEL
    taken = {a for a in assigned if a}

    return [l for l in labels if l not in taken]
