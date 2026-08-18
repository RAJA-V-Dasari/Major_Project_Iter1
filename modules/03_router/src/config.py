"""
Routing table and thresholds.

Plain data, no side effects: importing this module must not create
directories or touch the filesystem. (The module this replaces called
mkdir() at import time, so merely importing it to read a constant
changed the working tree.)
"""

# --- routes ---------------------------------------------------------
# Where a region can be sent. Kept as constants rather than bare
# strings so a typo is an AttributeError instead of a region silently
# routed to a processor that does not exist.
TEXT_OCR = "text_ocr"
MATH_OCR = "math_ocr"
DIAGRAM = "diagram"
REVIEW = "review"

ROUTES = (TEXT_OCR, MATH_OCR, DIAGRAM, REVIEW)

# Everything that is not recognisably something else. Handwriting line
# recognisers cope fine with a short "2b)" or a stray "=", so this is
# the safe default rather than a guess.
DEFAULT_ROUTE = TEXT_OCR


# --- geometry -------------------------------------------------------
# Thresholds are in RULE-PITCH units, like the rest of the pipeline, so
# they follow a resolution change instead of silently going wrong.
# Pitch is ~58.4px in this corpus.

# Below this height a region cannot be a row of prose. It is usually a
# question number, an operator, or a leftover mark. It still goes to
# text OCR - see DEFAULT_ROUTE - but is tagged so a consumer can batch
# these separately if it wants to.
SHORT_HEIGHT_PITCH = 0.55

# A region wider than this many pitches with normal height is a full
# line of writing. Used only for tagging, not for routing.
LONG_WIDTH_PITCH = 6.0


# --- downstream compatibility ---------------------------------------
# 03_ocr reads a `processor` string, using the vocabulary the old
# label-based router used. Routes are named for what they are; these
# are what the consumer expects to see.
PROCESSOR_NAMES = {
    TEXT_OCR: "ocr",
    MATH_OCR: "math_ocr",
    DIAGRAM: "diagram_parser",
    REVIEW: "manual_review",
}


# --- content re-routing ---------------------------------------------
# THIS is where math is actually identified - after recognition, from
# the text, not before it from the pixels. See rules.py for why.
#
# A line is treated as mathematics when it carries an equals sign (or
# another relational operator) AND is mostly non-alphabetic. Both
# conditions together, because prose contains "=" occasionally and
# short labels are non-alphabetic without being equations.
MATH_OPERATORS = set("=<>+-×÷*/^√∑∫≤≥≠")

# Share of non-space characters that must be non-alphabetic.
MATH_NON_ALPHA_RATIO = 0.40

# Words that mark mathematics even in an otherwise wordy line.
MATH_WORDS = {"mod", "sqrt", "log", "lim", "sin", "cos", "tan", "det"}

# Provisional. There is no labelled maths in this corpus to tune
# against, so these are a documented starting point to be measured
# once real OCR text exists - not a validated classifier.
MATH_RULES_ARE_PROVISIONAL = True
