"""
Turn a routed crop into images a formula recogniser can actually read.

    a line crop from 02_segment  ->  one image per expression on it

WHY THIS STAGE EXISTS AT ALL
----------------------------
The obvious pipeline is router -> recogniser: hand the crop straight
to an image-to-LaTeX model. Measured on this corpus that produces
nonsense, and the failures are not random - they are two specific,
fixable properties of the crop.

FAULT 1: THE CROP STILL CONTAINS THE PRINTED RULES
--------------------------------------------------
02_segment removes the ruled lines from its MASK before segmenting,
but deliberately never touches the image on disk, so every crop cut
from a ruled page carries the rules through with it. A formula model
has exactly one thing it does with a long horizontal line: it calls
it a fraction bar.

Handed the raw crop of a handwritten "= 69" with a rule underneath,
sumen-base returns

    \\frac { \\tilde { \\varepsilon } \\underbrace { \\xi q } } { \\ldots }

and for "10011" sitting inside a table box it returns a \\frac whose
numerator and denominator are both \\lim expressions. Neither crop
contains a fraction. The rules have to come out before recognition.

FAULT 2: A LINE REGION IS NOT AN EXPRESSION
-------------------------------------------
02_segment finds LINES; a recogniser wants ONE formula. A single
line region on this corpus routinely holds two unrelated workings
side by side - "2^13 = 8192" in the left half and a second "2^13 ="
in the right - separated by a wide gap. Fed both at once the model
tries to make one formula out of them and merges the halves.

So a crop fans out to N expressions here, and the stage downstream
recognises each one separately. N is usually 1.

HOW RULES ARE TOLD FROM HANDWRITING
-----------------------------------
By SPAN and THINNESS, never by darkness. A printed rule crosses the
whole crop and touches both borders, because the crop was cut out of
a page the rule ran clear across. A handwritten stroke does not - the
longest one measured here spans under a third of its crop.

That distinction alone would still cut through any glyph sitting on
the rule, breaking a "9" into two arcs. So an erasure is only applied
where the ink is thin: a rule is 1-3px tall, whereas a pen stroke
crossing it belongs to something several times taller. Ink that is
part of a vertically thick structure is kept even where it lies on
the rule line, which leaves descenders and loops intact.

WHAT IS DELIBERATELY NOT DONE
-----------------------------
No binarisation, no resizing. The recogniser's own processor expects
greyscale at its own input size and does that itself; doing it twice
throws away the anti-aliasing that tells a light stroke from paper.
Only rules are erased and whitespace is trimmed - the handwriting is
passed through untouched.

Run:
    python prepare.py --preview        # before/after pairs, no writes
    python prepare.py --preview --limit 40
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

MODULES_DIR = STAGE_DIR.parent
SEGMENT_DIR = MODULES_DIR / "02_segment"

PREVIEW_DIR = STAGE_DIR / "preview"

# Ink threshold. 03_tone already mapped paper to white and ink to
# black, so a fixed cut is safe here and, unlike Otsu, it does not
# move when a crop happens to contain no ink at all - Otsu on blank
# paper invents a threshold in the middle of the paper peak and
# reports a crop full of ink.
INK_MAX = 200

# A rule must cross this fraction of the crop and reach within
# EDGE_TOLERANCE of both borders. Printed rules span the whole page,
# so within a crop they always run edge to edge; the longest
# handwritten horizontal stroke measured on this corpus reaches 31%
# of its crop, which leaves the gap between the two populations
# comfortably wide.
RULE_SPAN = 0.85
EDGE_TOLERANCE = 0.02

# Ink belonging to a structure at least this tall (in rule-pitch
# units) is never erased as a rule, even where it lies along one.
# Printed rules measure 1-3px on these scans against a pitch of ~58,
# so 0.10 * pitch ~ 6px sits well above the rule and well below any
# glyph stroke.
RULE_THICKNESS_PITCH = 0.10

# Ink shorter than this fraction of the rule pitch cannot be a
# glyph, so it is ignored when measuring glyph height. This is a
# floor on the MEASUREMENT only - the ink itself is kept, because a
# dot on an "i" or a decimal point is exactly this small.
GLYPH_FLOOR_PITCH = 0.12

# Column gap that separates two expressions, in x-heights. Gaps
# inside an expression are the spaces around operators - roughly one
# x-height. Two separate workings on one line sit several apart.
SPLIT_GAP_XHEIGHT = 2.2

# White margin left around each expression. Recognisers are trained
# on formulas with air around them, and a glyph flush to the border
# reads as clipped.
PAD_XHEIGHT = 0.35

# An expression below this much ink is a speck or a leftover rule
# fragment, not something to spend a recogniser call on.
MIN_INK_PIXELS = 25

PAD_VALUE = 255


def ink_mask(gray):
    """Ink as a boolean array. Paper is white after 03_tone."""

    return gray < INK_MAX


def _rule_pixels(mask, axis, span, thickness):
    """
    Pixels belonging to a printed rule running along `axis`.

    axis=1 finds horizontal rules (the ruling), axis=0 vertical ones
    (the table and margin lines students draw a box with).
    """

    height, width = mask.shape

    length = width if axis == 1 else height

    reach = int(round(span * length))

    if reach < 8:
        return np.zeros_like(mask)

    # Opening with a long thin kernel keeps only ink that continues
    # for `reach` px in a straight line along the axis.
    kernel = (reach, 1) if axis == 1 else (1, reach)

    straight = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones(kernel[::-1], np.uint8),
    ).astype(bool)

    if not straight.any():
        return straight

    # Opening finds long ink, but a line of separately-written glyphs
    # can also be long. A rule additionally reaches both borders,
    # because the crop was cut from a page it crossed entirely.
    edge = max(1, int(round(EDGE_TOLERANCE * length)))

    count, labels = cv2.connectedComponents(straight.astype(np.uint8))

    keep = np.zeros_like(straight)

    for label in range(1, count):

        component = labels == label

        coords = np.nonzero(component.any(axis=1 - axis))[0]

        if coords[0] > edge or coords[-1] < length - 1 - edge:
            continue

        keep |= component

    if not keep.any():
        return keep

    # Never erase ink that is part of something thick - that is a pen
    # stroke crossing the rule, and cutting it would break the glyph.
    girth = (1, thickness) if axis == 1 else (thickness, 1)

    thick = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones(girth[::-1], np.uint8),
    ).astype(bool)

    return keep & ~thick


def suppress_rules(gray, pitch):
    """
    Erase printed ruling and table lines, leaving handwriting intact.

    Returns the cleaned image and the number of pixels erased, so a
    caller can tell a crop that was full of ruling from one that
    never had any.
    """

    mask = ink_mask(gray)

    thickness = max(3, int(round(RULE_THICKNESS_PITCH * pitch)))

    rules = (
        _rule_pixels(mask, 1, RULE_SPAN, thickness)
        | _rule_pixels(mask, 0, RULE_SPAN, thickness)
    )

    cleaned = gray.copy()
    cleaned[rules] = PAD_VALUE

    return cleaned, int(rules.sum())


def x_height(mask, pitch):
    """
    Typical glyph height, used as the unit for gaps and padding.

    The median over connected components is deliberate: it ignores
    the one tall bracket or long division bar that a mean would
    follow.

    Specks have to be excluded first, and the reason is this stage's
    own doing: erasing a rule leaves a trail of 1-2px fragments
    wherever the rule was not quite straight. Counting those, the
    median glyph height on the test crops came out at 3px instead of
    35px, which shrank the split gap by a factor of ten and cut
    "= 69" into "=" and "69". Anything shorter than a fraction of the
    rule pitch cannot be a glyph, so it does not get a vote.
    """

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )

    floor = max(4.0, GLYPH_FLOOR_PITCH * pitch)

    heights = [
        stats[label, cv2.CC_STAT_HEIGHT]
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= 12
        and stats[label, cv2.CC_STAT_HEIGHT] >= floor
    ]

    if not heights:
        return max(1.0, 0.30 * pitch)

    return float(np.median(heights))


def glyph_mask(mask, pitch):
    """
    The ink, with rule debris dropped.

    Erasing a rule never comes out perfectly clean: wherever the
    printed line wobbled by a pixel, a short fragment of it survives.
    Those fragments lie strung out along the old rule, a few tens of
    px apart, which is close enough to bridge every gap in the column
    profile - on the test crop that merged two separate workings, one
    at each end of the line, into a single expression.

    So splitting and measuring run on this mask instead. Only the
    MASK loses the specks: the image handed to the recogniser is cut
    from the full cleaned crop, so a decimal point or the dot on an
    "i" is still there in the pixels even though it got no vote here.
    """

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )

    floor = max(4.0, GLYPH_FLOOR_PITCH * pitch)

    keep = np.zeros(count, bool)

    for label in range(1, count):

        tall = stats[label, cv2.CC_STAT_HEIGHT] >= floor
        wide = stats[label, cv2.CC_STAT_WIDTH] >= floor
        solid = stats[label, cv2.CC_STAT_AREA] >= 12

        keep[label] = solid and (tall or wide)

    return keep[labels]


def split_spans(mask, gap):
    """
    Column ranges holding one expression each, split on wide gaps.

    Everything is measured on the ink profile rather than on
    components, so a superscript that overhangs the glyph below it
    cannot open a false gap.
    """

    columns = mask.any(axis=0)

    if not columns.any():
        return []

    spans = []
    start = None
    blank = 0

    for x, inked in enumerate(columns):

        if inked:
            if start is None:
                start = x
            blank = 0
            end = x
        else:
            if start is None:
                continue
            blank += 1
            if blank >= gap:
                spans.append((start, end + 1))
                start = None

    if start is not None:
        spans.append((start, int(np.nonzero(columns)[0][-1]) + 1))

    return spans


def prepare(gray, pitch):
    """
    A routed crop in, one image per expression out.

    Each entry carries `offset` - the (x, y) of the expression's top
    left corner within the incoming crop - so a recognised formula
    can still be pointed back at its place on the page.
    """

    cleaned, erased = suppress_rules(gray, pitch)

    mask = glyph_mask(ink_mask(cleaned), pitch)

    if not mask.any():
        return [], {"rule_pixels_erased": erased, "x_height": 0.0}

    height = x_height(mask, pitch)

    gap = max(2, int(round(SPLIT_GAP_XHEIGHT * height)))
    pad = max(2, int(round(PAD_XHEIGHT * height)))

    expressions = []

    for x1, x2 in split_spans(mask, gap):

        window = mask[:, x1:x2]

        if window.sum() < MIN_INK_PIXELS:
            continue

        rows = np.nonzero(window.any(axis=1))[0]

        y1, y2 = int(rows[0]), int(rows[-1]) + 1

        tight = cleaned[y1:y2, x1:x2]

        padded = cv2.copyMakeBorder(
            tight, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT,
            value=PAD_VALUE,
        )

        expressions.append({
            "image": padded,
            "offset": (int(x1) - pad, int(y1) - pad),
            "ink_pixels": int(window.sum()),
        })

    return expressions, {
        "rule_pixels_erased": erased,
        "x_height": round(height, 2),
    }


def page_pitch():
    """Rule pitch per page id, the unit every threshold here is in."""

    path = SEGMENT_DIR / "output" / "segmentation.json"

    if not path.exists():
        raise SystemExit(f"{path} not found - run 02_segment first")

    with open(path) as handle:
        return {page["page_id"]: page["rule_pitch"] for page in json.load(handle)}


def _preview(limit):
    """
    Before/after strips: the crop as routed, then what the recogniser
    is actually handed. Written to preview/, never to output/.
    """

    pitches = page_pitch()

    crop_dir = SEGMENT_DIR / "crops"

    with open(crop_dir / "manifest.csv") as handle:
        rows = [row for row in csv.DictReader(handle) if row["crop"]]

    step = max(1, len(rows) // limit)
    rows = rows[::step][:limit]

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for row in rows:

        gray = cv2.imread(str(crop_dir / row["crop"]), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            continue

        expressions, stats = prepare(gray, pitches[row["page_id"]])

        width = max(
            [gray.shape[1]] + [e["image"].shape[1] for e in expressions]
        ) + 20

        strips = [_labelled(gray, width, "routed crop")]

        for index, expression in enumerate(expressions, start=1):
            strips.append(_labelled(
                expression["image"], width,
                f"expression {index}/{len(expressions)}",
            ))

        if len(expressions) == 0:
            strips.append(_labelled(
                np.full((30, width - 20), PAD_VALUE, np.uint8),
                width, "no expression - all ink was rule",
            ))

        name = Path(row["crop"]).stem
        cv2.imwrite(str(PREVIEW_DIR / f"{name}.png"), np.vstack(strips))

        print(
            f"{name}  {len(expressions)} expression(s)  "
            f"{stats['rule_pixels_erased']}px of rule erased"
        )

    print(f"\n-> {PREVIEW_DIR}")


def _labelled(image, width, caption):
    """One captioned strip of a preview sheet."""

    canvas = np.full((image.shape[0] + 26, width), PAD_VALUE, np.uint8)
    canvas[22:22 + image.shape[0], 10:10 + image.shape[1]] = image

    cv2.putText(
        canvas, caption, (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, 0, 1, cv2.LINE_AA,
    )

    canvas[-1, :] = 200

    return canvas


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", action="store_true",
        help="write before/after sheets to preview/ and exit",
    )
    parser.add_argument("--limit", type=int, default=24)

    args = parser.parse_args()

    if not args.preview:
        raise SystemExit(
            "prepare.py is a library for math_ocr.py; "
            "run it directly only with --preview"
        )

    _preview(args.limit)


if __name__ == "__main__":
    main()
