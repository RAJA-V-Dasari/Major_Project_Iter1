"""
The routing decisions, each one auditable.

Every rule returns a route AND the reason it fired, with the values it
fired on, so a routing decision can be explained after the fact
instead of being a number that appeared. That matters more than usual
here, because one of the decisions below is deliberately NOT made.

WHY MATHS IS NOT DETECTED FROM THE PIXELS
-----------------------------------------
The obvious design is to look at a crop and send equations to a maths
recogniser and prose to a handwriting one. That was measured on 600
real crops before being rejected. Four candidate features - ink
density, mean inter-symbol gap over line height, connected-component
size variation, and aspect ratio - are all UNIMODAL across the corpus:

    ink density        6  49 139 185 123  47  17  10   7   4   8   5
    gap / height     592   5   1   0   0   1   0   0   0   0   0   1
    component CV     176  19  27  64  66  68  49  56  26  28  13   8
    aspect w/h       169  96  79 144  74  25   7   2   2   0   1   1

There is no second peak to cut between, so any threshold would be an
invented boundary. The corpus has no labelled maths to validate one
against either, which is the same reason 02_segment does not classify
regions at all: a wrong label is a claim the rest of the pipeline has
to un-learn.

So maths is identified AFTER recognition, from the text, where "="
is directly observable rather than inferred from ink statistics. That
is what reroute_by_content() does, and it is the path that feeds a
maths OCR engine.

WHAT IS DECIDED FROM GEOMETRY
-----------------------------
Only what segmentation already established: a region it flagged `tall`
is a diagram, brace or long division rather than a row of writing, and
a line recogniser should not be handed it. Everything else defaults to
text OCR, because a handwriting recogniser reads a short "2b)" or a
stray "=" perfectly well, and sending it there costs nothing while
guessing costs accuracy.
"""

import config


def _tags(width, height, pitch):
    """Descriptive labels. These do not affect routing, only reporting."""

    tags = []

    if height < config.SHORT_HEIGHT_PITCH * pitch:
        tags.append("short")

    if width > config.LONG_WIDTH_PITCH * pitch:
        tags.append("full-line")

    return tags


def route_geometry(region, pitch):
    """
    Pre-recognition routing, from geometry alone.

    Returns (route, reason, tags).
    """

    width = region["x2"] - region["x1"]
    height = region["y2"] - region["y1"]

    tags = _tags(width, height, pitch)

    if region["tall"]:
        return (
            config.DIAGRAM,
            "segmentation flagged the region as taller than a line of "
            "writing, so it is a figure/brace rather than text",
            tags,
        )

    return (
        config.DEFAULT_ROUTE,
        "no evidence it is anything but a line of writing; geometry "
        "cannot distinguish maths from prose on this corpus",
        tags,
    )


def looks_like_maths(text):
    """
    Does recognised text read as an equation?

    Returns (verdict, reason). Provisional - see
    config.MATH_RULES_ARE_PROVISIONAL. There is no labelled maths here
    to tune against, so this is a documented starting point that should
    be measured once a real engine produces text.
    """

    stripped = [c for c in text if not c.isspace()]

    if not stripped:
        return False, "empty text"

    lowered = text.lower()

    for word in config.MATH_WORDS:
        if word in lowered.split():
            return True, f"contains the mathematical word {word!r}"

    has_operator = any(c in config.MATH_OPERATORS for c in stripped)

    if not has_operator:
        return False, "no relational or arithmetic operator present"

    non_alpha = sum(1 for c in stripped if not c.isalpha())

    ratio = non_alpha / len(stripped)

    if ratio < config.MATH_NON_ALPHA_RATIO:
        return (
            False,
            f"has an operator but is {(1 - ratio) * 100:.0f}% letters, "
            f"so it reads as prose mentioning one",
        )

    return (
        True,
        f"carries an operator and is {ratio * 100:.0f}% non-alphabetic",
    )


def reroute_by_content(routed, text):
    """
    Post-recognition re-routing.

    Takes a region already routed by geometry plus the text an OCR
    engine read from it, and returns the possibly-updated
    (route, reason). This is the only place maths is identified.

    Called by whatever runs the recognisers, not by this module's own
    CLI - the text does not exist yet at routing time.
    """

    if routed["route"] == config.DIAGRAM:
        return routed["route"], routed["reason"]

    verdict, why = looks_like_maths(text)

    if verdict:
        return config.MATH_OCR, f"re-routed after recognition: {why}"

    return routed["route"], routed["reason"]
