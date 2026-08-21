"""
Decide whether a mark in the margin is a question number.

Geometry gets a candidate this far; it cannot get it further. Measured
over 282 candidates from 10 students, width runs 0.15-2.39 pitch for
BOTH real markers and the words students park in the margin - "Ans:-",
"Step1:-", "the", "web client" - and the two distributions overlap
completely. There is no threshold between them, so the mark has to be
read.

WHAT COUNTS AS A QUESTION NUMBER
--------------------------------
A number, or a single letter, or a roman numeral - optionally prefixed
"Q" and optionally trailed by ")" or ".". Taken from what the corpus
actually contains (see the survey in this module's README):

    1   2   4   20   128        bare number
    2a) 3b. 4b   2a.            number + subpart
    Q2b) Q3a) Q4b)              Q-prefixed
    a) b) c) d) e)  a. b.       bare letter
    i) ii) iii) iv) v)          roman

and NOT, from the same survey:

    Ans:-  1.Ans:-  Step1:-  the  We  or  HT  SC  eg  vers
    web client  Transmission  HTTP  AR  ->  =>

This is an allow-list on shape, not a block-list of known noise words.
A block-list only ever covers the words already seen; the allow-list
rejects every word by default and lets through the small, closed set of
things a question number can look like.

WHY THIS BEATS A TIGHTER GEOMETRIC GATE
---------------------------------------
The gate had to be strict to keep false positives down, and that
strictness is what made it miss student_07 entirely - that student
writes "2a)" just RIGHT of the margin rule, so nothing crosses it and
nothing was found on any of their three booklets. Reading the mark
means the gate can be loosened to "leftmost short token on a line",
covering both conventions with one mechanism, because a wrong candidate
is now rejected by what it says rather than by where it sits.
"""

import re

# Roman numerals i..xii - far past what an exam sub-part uses, but the
# pattern costs nothing to extend and stops at xii rather than pretending
# to be a general roman parser.
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii",
         "ix", "x", "xi", "xii"}

# number, optional single-letter subpart:   1  2a  128  4b
NUMBER = re.compile(r"^\d{1,3}[a-z]?$")

# single letter:  a  b  c
LETTER = re.compile(r"^[a-z]$")


def normalise(text):
    """Strip the decoration a question number is written with.

    Circles come back from OCR as parentheses, or as nothing at all, so
    both are stripped. Trailing ")" "." ":" "-" are separators, not part
    of the number. A leading "q" is the student writing "Q2b" for
    question 2b.
    """

    if text is None:
        return ""

    text = text.strip().lower()

    # OCR of a circled mark commonly comes back wrapped
    text = text.strip("()[]{}<>")

    # trailing separators
    text = text.rstrip(").:-,;")

    # leading separators, and the "Q" prefix
    text = text.lstrip("(.[")

    if text.startswith("q") and len(text) > 1:
        text = text[1:]

    # a stray space inside a short mark is OCR noise: "2 a" -> "2a"
    text = text.replace(" ", "")

    return text


def is_question_number(text):
    """True if `text` reads as a question number. See module docstring."""

    cleaned = normalise(text)

    if not cleaned or len(cleaned) > 4:
        return False

    if cleaned in ROMAN:
        return True

    if NUMBER.match(cleaned):
        return True

    if LETTER.match(cleaned):
        return True

    return False


def classify(text):
    """(accepted, normalised) - for logging why a candidate was dropped."""

    return is_question_number(text), normalise(text)
