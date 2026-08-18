"""
Locate content on every prepared page: where it is, not what it is.

    02_segment/input/  (deskewed, cropped, toned)
        -> 02_segment/output/     region geometry, feeds the next module
        -> 02_segment/annotated/  the same pages with boxes drawn, to look at

Emits a page -> block -> line hierarchy as JSON plus flat CSVs, and a
rendered image per page.

WHY LOCALISATION ONLY, NO CLASSIFICATION
----------------------------------------
Calling a region "maths" or "figure" or "crossed out" needs training
data this corpus does not have - the earlier labelling pass found 27
figure, 10 code and 1 crossed_out examples in total. A wrong label is
a claim the rest of the pipeline then has to un-learn. Where content
sits is a much easier question, and the one OCR actually needs
answered.

WHY THE RULES COME OUT FIRST
----------------------------
The printed rules put ink on every ruled row. Left in, they connect
everything to everything: a horizontal projection has no valleys and
the whole page smears into one region. So they are removed from the
MASK before segmenting - the image on disk is untouched.

Rules are told from handwriting by LENGTH, not by darkness or
straightness: a rule spans most of the page, a pen stroke does not.
Short horizontal strokes are deliberately KEPT, because a
strikethrough is exactly a short dense horizontal stroke sitting in a
line of writing, and deleting those would erase the fact that the
student crossed something out.

WHY RUN-LENGTH SMOOTHING, NOT PROJECTION PROFILING
--------------------------------------------------
Projection profiling finds horizontal bands of ink, so it is blind to
anything not laid out in rows. Measured previously on this corpus it
put only 42-95% of the handwriting inside a box: it missed isolated
question numbers ("2b)") and every arrow and vertical stroke of a
diagram. Smearing ink then taking connected components means every
stroke joins some region by construction. Ink that no region covers
is content that silently never reaches OCR, which is the whole point
of preferring this.

EVERY THRESHOLD IS A MULTIPLE OF THE MEASURED RULE PITCH
--------------------------------------------------------
Not a pixel count. The previous generation of this code (scan_doc_v2)
was tuned at 300 DPI with absolute pixel thresholds, and when the
corpus moved to 200 DPI every one of them was a third too big - it did
not error, it silently mis-segmented. Pitch is measured per page from
the rules that were just removed, so the same code follows the corpus
to another resolution on its own.

A REGION IS A UNIT OF CONTENT, NOT A CONNECTED COMPONENT
-------------------------------------------------------
Finding every stroke is not the same as grouping them correctly, and
this module used to do only the first. It put 99.8% of the ink inside
some box and then handed the next stage 30 boxes a page, because the
pieces of one thing kept arriving as several regions:

  - a descender is not on its line's baseline, so it failed the snap
    and was emitted alone. 90% of all unsnapped components, 25% of
    every region produced. See ABSORB_* and `absorb`.
  - a table's cell borders lie along the printed rules, so each of its
    rows snapped like a line of writing. Scored against the annotated
    pages, one table came out as 10 regions. See GRID_* and
    `find_grids` / `merge_grids`.

Measured over 33 annotated pages by `modules/06_evaluation`, fixing
both took regions from 30.0 to 21.8 a page and fragments per
hand-drawn region from 5.60 to 4.04 - figures 8.31 -> 3.23, maths
6.79 -> 4.61, tables 10 -> 1 - with ink coverage unchanged at 99.8%.
Coverage staying flat is the point: this is regrouping, not discarding.

COVER PAGES ARE SKIPPED
-----------------------
page_01 of every booklet is the printed identity block - name, USN,
signature, marks. It is not answer content, and pipelines that do not
need identity data should not be handling it.

Run:
    python segment.py                  # every content page
    python segment.py --limit 40       # a sample
    python segment.py --no-images      # geometry only, much faster
"""

import argparse
import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

SOURCE_DIR = STAGE_DIR / "input"

# Machine-readable geometry - this is what the next module consumes.
OUT_DIR = STAGE_DIR / "output"

# The same pages with the regions drawn on, for a human to page through
# and judge. Nothing downstream reads these.
ANNOTATED_DIR = STAGE_DIR / "annotated"

# page_01 is the identity cover sheet.
COVER_PAGE = 1

# --- ink ------------------------------------------------------------
# 03_tone leaves paper at 255 and ink dark, so a fixed cut is enough
# here and an adaptive threshold would only re-introduce the paper
# texture that stage removed. Measured across the corpus the ink
# fraction moves only 0.031 -> 0.048 between cuts of 100 and 220, i.e.
# there is a wide flat basin and nothing delicate about landing in it.
INK_THRESHOLD = 180

# --- rules ----------------------------------------------------------
# Probe for horizontal ink. Expressed as a fraction of page width so it
# survives a resolution change.
RULE_PROBE_FRACTION = 20        # kernel = width / 20

# Curved or faint rules break into fragments; rejoin them before
# judging length, or a long rule is scored as several short strokes.
RULE_BRIDGE_FRACTION = 0.04     # of page width

# A component this much of the page wide is a printed rule. Well clear
# of the longest handwritten strikethrough, which is a few words wide.
RULE_WIDTH_FRACTION = 0.55

# Grow the rule mask by this much before subtracting it.
#
# The rule is found by opening with a long flat kernel, which returns
# the rule's core but not its anti-aliased edge. Left behind, that
# 1px fringe survives as its own long flat component and every rule
# turns into a spurious "line": measured on one page, 125 flat slivers
# (h<=6px, w>60px) at 0 dilation versus 1 at a single pixel. Larger
# values keep removing ink for no further gain and start eating the
# handwriting that crosses a rule, so this stays at the minimum that
# works.
RULE_DILATE_PX = 1

# The booklet also has ONE printed vertical rule - the margin line -
# and it has to come out for the same reason as the horizontal ones.
# It is easy to miss: by the time the horizontal rules are removed it
# has been chopped into pitch-tall fragments, each of which looks like
# an innocent little component, and every one becomes a spurious
# one-line region 3px wide. So it is detected on the ORIGINAL mask,
# while it is still continuous.
#
# Measured across random pages: exactly one per page, at x=157-301
# (the margin position varies with how the booklet was printed and
# trimmed). Requiring half the page height keeps it clear of any
# handwritten vertical stroke, which tops out around two line heights.
VERTICAL_PROBE_FRACTION = 25     # kernel = height / 25
VERTICAL_BRIDGE_FRACTION = 0.04  # of page height
VERTICAL_RULE_HEIGHT_FRACTION = 0.5

# Fallback if a page yields too few rules to measure - the corpus
# median. Only used to keep a degenerate page from crashing.
FALLBACK_PITCH = 58.5
MIN_RULES_FOR_PITCH = 5

# --- smearing -------------------------------------------------------
# Horizontal smear only, joining strokes within a word and words
# within a line. In pitch units, so it follows the resolution.
#
# THERE IS DELIBERATELY NO VERTICAL SMEAR. It looks harmless - a few
# pixels to join the parts of one character - and it is not: a
# descender on one line and an ascender on the next sit within a few
# pixels of each other, so any vertical smear chains them, and the
# chain runs the length of the page. Measured on one page, the tallest
# component was 1892px of a 2177px page at 0.17 pitch, and still
# 1880px at 0.10; with no vertical smear it is 116px. That runaway
# then swallows the block grouping too.
#
# Nothing is lost by dropping it, because assembling a line is
# find_lines' job now - it snaps components to the printed rule they
# sit on, so a line arriving as several fragments is reassembled
# there. The smear does not have to do that work, and does it badly.
H_SMEAR_PITCH = 0.55

# Anything smaller is a speck of scanner noise. In pitch^2 so it
# scales with resolution.
#
# 0.04 (about 135px^2 here) was tried first and was far too eager: it
# discarded punctuation, the dot of an i, and small digits, which
# showed up as ink coverage falling to ~95%. Dropped ink is content
# that silently never reaches OCR, so this is set well below the size
# of the smallest real mark.
MIN_REGION_AREA_PITCH = 0.006

# --- grouping -------------------------------------------------------
# A component taller than this is not text laid out in a row - a
# diagram, a brace, a long division - and is kept as its own region
# rather than dragged into a line of writing.
TALL_REGION_PITCH = 1.6

# How far a component's baseline may sit from a printed rule and still
# count as writing on that rule. Just under half the pitch, so the
# nearest rule is never ambiguous; beyond it the component is treated
# as sitting on no line at all. See find_lines for the measurements.
LINE_SNAP_PITCH = 0.45

# --- absorbing what did not snap ------------------------------------
# The band a written line actually occupies, measured from the printed
# rule it rests on: up to the ascender, down to the descender. A
# component that did not snap but lies in this band, over a line that
# did, is PART of that line and not a region of its own.
#
# This exists because the snap above is a baseline test, and a
# descender does not have the baseline of its own word. Measured over
# 50 pages: 390 components failed the snap, and 351 of them - 90% -
# were descenders hanging below a line that snapped perfectly well.
# Emitted alone they became 25% of every region the module produced:
# the loop of a 'g', the tail of a 'y', each cropped as its own "line"
# and queued for OCR. A further 2.8% were superscripts sitting above
# their line, which is worse than cosmetic, because the superscript of
# 2^{m-1} carries the meaning of the expression it was cut out of.
#
# Asymmetric on purpose: ascenders and superscripts reach higher above
# the rule than descenders fall below it, and the gap above a rule is
# shared with the line before, so the upward reach is the value that
# has to stay short of a full pitch.
ABSORB_ASCENT_PITCH = 0.85
ABSORB_DESCENT_PITCH = 0.50

# A component is only absorbed into a line it genuinely sits over -
# this much of its width must be within the line's horizontal extent.
# Without it, a diagram label level with a paragraph on the far side of
# the page would be swallowed by it.
ABSORB_OVERLAP_FRACTION = 0.5

# --- ruled structure (tables, boxed diagrams) -----------------------
# A hand-drawn grid is the one layout this module cannot treat as text
# and must not: its cell borders lie along the printed rules, so every
# row of a table snaps like a line of writing and the table comes out
# shredded. Measured against the annotated pages, a table was cut into
# a median of 26 regions - by far the worst class, and the reason this
# section exists.
#
# What identifies a grid is the pen strokes drawing it: long, straight
# and thin, unlike any letter. Rule removal has already taken out the
# PRINTED horizontals (anything past RULE_WIDTH_FRACTION of the page),
# so what is left at this length is drawn by hand.
#
# Measured over the annotated boxes: 13 of 14 tables contain 6 or more
# such strokes and 93% contain at least two, against 1% of paragraphs -
# 61 of 68 paragraphs contain none at all. That gap is what the
# threshold below sits in.
# The lengths themselves were swept against the annotated pages rather
# than picked. Loosening them from 2.5/1.0 to 1.5/0.8 pitch cut figure
# fragments from 7.3 to 3.4 and maths from 5.6 to 4.6, while paragraph
# fragments went 3.91 -> 3.85 and paragraph spill 12.7% -> 12.9% - i.e.
# it costs prose nothing measurable, because prose contains no strokes
# of this shape for the detector to fire on. Below 1.5/0.8 the curve is
# flat (3.38 -> 3.00 over the whole remaining range) and 13 annotated
# figures is far too few to justify chasing it, so this sits at the
# knee with margin rather than at the minimum.
GRID_H_STROKE_PITCH = 1.5       # a horizontal stroke this long is drawn
GRID_V_STROKE_PITCH = 0.8       # a vertical stroke this tall is drawn
GRID_STROKE_THIN_PITCH = 0.35   # ...and no thicker than this
GRID_H_ASPECT = 6.0
GRID_V_ASPECT = 4.0

# Strokes closer than this belong to the same drawing.
GRID_LINK_PITCH = 0.75

# One stroke is an underline under a heading, not a table. Requiring
# two is what keeps this off ordinary prose - see the 1% above.
MIN_GRID_STROKES = 2

# How far beyond its own strokes a drawing may reach, vertically.
#
# Merging has to grow the box - absorbing one row of a table usually
# extends it over the next - but growth that is only bounded by what it
# touches does not stop. Observed on s57_c1_p07: the box crept upward a
# line at a time and took three lines of ordinary prose into the
# diagram. Content pulled into a drawing never reaches the recogniser,
# which is the same failure as dropping it.
#
# The strokes say where the drawing is; nothing above them does. So
# vertical reach is capped at the stroke cluster's own extent plus this
# margin - enough for a row of labels over or under the drawing - while
# horizontal reach stays free, because a label to one side is at the
# drawing's own height and cannot run away up the page.
#
# Capping the area a region must have inside the box was tried first
# and is worse than either extreme: at 0.5 it broke the growth that
# makes merging work at all, because a wide table row overlaps a narrow
# stroke cluster by well under half its own area, and figure fragments
# went from 3.2 back up to 9.5.
#
# Swept from 1.0 pitch to unbounded. Figure fragments bottom out at
# 3.23 by 2.0 and do not improve after; the number that keeps moving is
# a single tall diagram (s57_c1_p07: 40 regions at 1.0, 27 at 2.5, 24
# unbounded). Text swallowed by a drawing is 1 box at 1.0 and 2 at
# every larger value including unbounded, so prose loss is not what
# this trades against - it is flat.
#
# 2.5 therefore takes the whole measurable gain. It stays a finite cap
# rather than going unbounded because the remaining 3 regions on one
# page are not worth giving up the guarantee that no page can collapse
# into a single region, which is the failure this bound exists for.
GRID_GROW_PITCH = 2.5

# NOTE ON FIGURES. This finds drawn structure, and a hand-drawn diagram
# usually has some - so it helps figures too, measurably. What it does
# NOT do is CLASSIFY a region as a figure, and that distinction is the
# whole point of where the line is drawn here.
#
# Every signal that might tell a diagram from prose was measured over
# the annotated pages and none separates them: ink whose baseline snaps
# to a printed rule is 98% inside figure boxes against 100% inside
# paragraphs, tall-component mass is 0.00 for both, and ink density is
# 0.031 against 0.066 - overlapping distributions in every case.
# Students draw diagrams on ruled paper and rest them on the rules
# exactly as they rest their writing.
#
# Merging works anyway because it does not need to answer "is this a
# figure". It only needs "are these two pieces part of one drawn
# object", and a stroke joining them answers that locally. A diagram
# with no straight strokes in it - a free-hand curve, a sketch - is
# still missed, and no threshold here will catch it. That case needs a
# learned detector; see modules/06_evaluation/README.md.
#
# Lines separated by more than this belong to different blocks.
BLOCK_GAP_PITCH = 1.6

# --- rendering ------------------------------------------------------
LINE_COLOUR = (32, 94, 27)       # dark green, BGR
BLOCK_COLOUR = (140, 20, 74)     # dark purple
RULE_TINT = (190, 190, 190)      # what was removed, ghosted in


def ink_mask(gray):
    """Ink white (255), paper black."""

    return ((gray < INK_THRESHOLD) * 255).astype(np.uint8)


def _vertical_rules(mask):
    """The printed margin rule, as a mask. See VERTICAL_* above."""

    height, width = mask.shape

    verticals = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(1, height // VERTICAL_PROBE_FRACTION))
        ),
    )

    bridged = cv2.morphologyEx(
        verticals, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(1, int(height * VERTICAL_BRIDGE_FRACTION)))
        ),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, 8)

    found = np.zeros_like(mask)

    for label in range(1, count):

        if stats[label, cv2.CC_STAT_HEIGHT] < (
                VERTICAL_RULE_HEIGHT_FRACTION * height):
            continue

        found[(labels == label) & (verticals > 0)] = 255

    if RULE_DILATE_PX:
        found = cv2.dilate(
            found,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (2 * RULE_DILATE_PX + 1, 1)
            ),
        )

    return found


def split_rules(mask):
    """
    Separate printed rules from handwriting.

    Returns (clean, rules, pitch, rule_rows): `clean` is the mask with
    rules removed, `rules` the removed pixels, `pitch` the measured
    rule-to-rule spacing, `rule_rows` the y of each rule - which is
    the line grid the writing sits on, so find_lines needs it.
    """

    height, width = mask.shape

    horizontals = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(1, width // RULE_PROBE_FRACTION), 1)
        ),
    )

    bridged = cv2.morphologyEx(
        horizontals, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(1, int(width * RULE_BRIDGE_FRACTION)), 1)
        ),
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bridged, connectivity=8
    )

    rules = np.zeros_like(mask)
    centres = []

    for label in range(1, count):

        if stats[label, cv2.CC_STAT_WIDTH] < RULE_WIDTH_FRACTION * width:
            continue

        # only the real ink, not the pixels the bridge invented
        rules[(labels == label) & (horizontals > 0)] = 255
        centres.append(centroids[label][1])

    if RULE_DILATE_PX:
        rules = cv2.dilate(
            rules,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, 2 * RULE_DILATE_PX + 1)
            ),
        )

    rules = cv2.bitwise_or(rules, _vertical_rules(mask))

    clean = cv2.bitwise_and(mask, cv2.bitwise_not(rules))

    pitch = FALLBACK_PITCH

    if len(centres) >= MIN_RULES_FOR_PITCH:

        gaps = np.diff(sorted(centres))

        # a gap far from the mode is a skipped (blank or faint) rule,
        # not a pitch; the median over plausible gaps ignores those
        plausible = gaps[(gaps > 0.5 * FALLBACK_PITCH)
                         & (gaps < 1.5 * FALLBACK_PITCH)]

        if len(plausible):
            pitch = float(np.median(plausible))

    return clean, rules, pitch, sorted(centres)


def find_regions(clean, pitch):
    """Smear ink, then take connected components as raw regions."""

    smeared = cv2.morphologyEx(
        clean, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(1, int(H_SMEAR_PITCH * pitch)), 1)
        ),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(smeared, 8)

    minimum_area = MIN_REGION_AREA_PITCH * pitch * pitch

    boxes = []

    for label in range(1, count):

        if stats[label, cv2.CC_STAT_AREA] < minimum_area:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    return boxes


def find_lines(boxes, pitch, rule_rows):
    """
    One region per line of writing; tall regions kept on their own.

    Components are assigned to the printed rule their BASELINE sits
    on. Two things make that the right anchor, both measured rather
    than assumed:

    The rules are a fixed grid we have already detected, so assignment
    cannot drift. An earlier version clustered component centres
    against a running mean instead, and single writing lines came out
    fragmented - the mean moves as it absorbs each component, so a
    line splits the moment the drift exceeds the tolerance. (Merging
    overlapping vertical ranges, tried before that, fails the opposite
    way: the merged band only grows, so one tall item swallows the
    page.) Fixed anchors cannot do either.

    The baseline is the stable edge. Measured against the nearest
    rule, component BOTTOMS land in IQR [0.4, 7.9] px while their
    centres spread over [-13.3, 1.9] - because a word's centre moves
    with whether it happens to contain a descender, while its baseline
    does not. Handwriting rests ON the rule; that is the whole reason
    the rule is there.

    Anything taller than a line and a half is not text laid out in a
    row - a diagram, a brace, a long division - and is kept whole
    rather than snapped to some rule.
    """

    tall = [b for b in boxes if (b[3] - b[1]) > TALL_REGION_PITCH * pitch]
    flat = [b for b in boxes if (b[3] - b[1]) <= TALL_REGION_PITCH * pitch]

    lines = []

    if rule_rows:

        rows = np.asarray(rule_rows, dtype=np.float64)

        tolerance = LINE_SNAP_PITCH * pitch

        grouped = {}
        unsnapped = []

        for box in flat:

            index = int(np.argmin(np.abs(rows - box[3])))

            if abs(rows[index] - box[3]) <= tolerance:
                grouped.setdefault(index, []).append(box)
            else:
                unsnapped.append(box)

        orphans = absorb(grouped, unsnapped, rows, pitch)

        for members in grouped.values():
            lines.append({
                "bbox": [min(b[0] for b in members),
                         min(b[1] for b in members),
                         max(b[2] for b in members),
                         max(b[3] for b in members)],
                "parts": len(members),
                "tall": False,
            })

        for box in orphans:
            lines.append({"bbox": box, "parts": 1, "tall": False})

    else:
        # no rules found (a blank or badly damaged page) - fall back to
        # one region per component rather than inventing a grid
        for box in flat:
            lines.append({"bbox": box, "parts": 1, "tall": False})

    for box in tall:
        lines.append({"bbox": box, "parts": 1, "tall": True})

    lines.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    return lines


def absorb(grouped, unsnapped, rows, pitch):
    """
    Put each unsnapped component back into the line it belongs to.

    The snap in find_lines is a BASELINE test, and there are parts of a
    written line that do not share their line's baseline: a descender
    hangs below it, a superscript floats above it, a stray dot sits
    beside it. Each of those failed the snap and used to be emitted as
    a region of its own - 25% of everything this module produced, and
    the single largest source of crops too small to read.

    A component belongs to a line when it lies in that line's band -
    from the ascender height above the printed rule to the descender
    depth below it - AND sits horizontally over writing that snapped to
    that rule. Both conditions are needed. The band alone would pull in
    a diagram label that happens to be level with a paragraph on the
    other side of the page; the overlap alone would pull in the line
    above and the line below.

    Where two bands overlap - and they do, because ascent plus descent
    is more than a pitch - the component goes to the band it overlaps
    more, so a descender that dips into the next line's ascender zone
    still returns to its own line.

    Mutates `grouped`. Returns the components that genuinely belong to
    no line, which stay regions in their own right: labels inside a
    diagram, a mark in the margin.
    """

    if not grouped:
        return unsnapped

    extents = {index: (min(b[0] for b in members), max(b[2] for b in members))
               for index, members in grouped.items()}

    ascent = ABSORB_ASCENT_PITCH * pitch
    descent = ABSORB_DESCENT_PITCH * pitch

    orphans = []

    for box in unsnapped:

        width = max(box[2] - box[0], 1)

        best, best_overlap = None, 0.0

        for index, (left, right) in extents.items():

            if (min(box[2], right) - max(box[0], left)) < (
                    ABSORB_OVERLAP_FRACTION * width):
                continue

            top = rows[index] - ascent
            bottom = rows[index] + descent

            overlap = min(box[3], bottom) - max(box[1], top)

            if overlap > best_overlap:
                best, best_overlap = index, overlap

        if best is None:
            orphans.append(box)
        else:
            grouped[best].append(box)

    return orphans


def find_grids(clean, pitch):
    """
    Areas of hand-drawn ruled structure, as boxes. See the GRID_* notes.

    A table's cell borders lie along the printed rules, so every row of
    it snaps to a rule exactly like a line of writing and the table is
    emitted one row - often one cell - at a time. Nothing downstream can
    reassemble it, because by then it is only a list of boxes.

    The strokes that draw the grid are the evidence, and they are
    unmistakable: far longer than a letter, and far thinner. The
    printed rules are already gone by this point, so a horizontal
    stroke still this long was drawn by hand.
    """

    count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)

    strokes = np.zeros_like(clean)
    found = 0

    for label in range(1, count):

        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        horizontal = (w >= GRID_H_STROKE_PITCH * pitch
                      and h <= GRID_STROKE_THIN_PITCH * pitch
                      and w >= GRID_H_ASPECT * max(h, 1))

        vertical = (h >= GRID_V_STROKE_PITCH * pitch
                    and w <= GRID_STROKE_THIN_PITCH * pitch
                    and h >= GRID_V_ASPECT * max(w, 1))

        if horizontal or vertical:
            strokes[labels == label] = 255
            found += 1

    if found < MIN_GRID_STROKES:
        return []

    link = max(1, int(GRID_LINK_PITCH * pitch))

    joined = cv2.dilate(
        strokes,
        cv2.getStructuringElement(cv2.MORPH_RECT, (link, link)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)

    grids = []

    for label in range(1, count):

        # how many separate strokes fell into this cluster - one is an
        # underline under a heading, not a grid
        piece = (labels == label) & (strokes > 0)

        pieces = cv2.connectedComponentsWithStats(
            piece.astype(np.uint8) * 255, 8
        )[0] - 1

        if pieces < MIN_GRID_STROKES:
            continue

        x = stats[label, cv2.CC_STAT_LEFT] + link // 2
        y = stats[label, cv2.CC_STAT_TOP] + link // 2
        w = stats[label, cv2.CC_STAT_WIDTH] - link
        h = stats[label, cv2.CC_STAT_HEIGHT] - link

        grids.append([int(x), int(y), int(x + w), int(y + h)])

    return grids


def merge_grids(lines, grids, pitch):
    """
    Collapse everything inside a drawn grid into one region.

    A table is one object. Handing OCR its 26 pieces separately loses
    the only thing that made it a table - which cell sits beside which -
    and hands a recogniser trained on lines of prose a row of digits
    with no context.

    The grid box grows to contain the regions it touches, repeatedly,
    because absorbing one row usually extends it over the next. Growth
    is bounded vertically by GRID_GROW_PITCH past the strokes that
    started it, so it cannot walk up the page into the prose above.
    Regions no grid claims pass through untouched.
    """

    if not grids:
        return lines

    def overlaps(a, b):
        return (min(a[2], b[2]) > max(a[0], b[0])
                and min(a[3], b[3]) > max(a[1], b[1]))

    grow = GRID_GROW_PITCH * pitch

    boxes = [list(g) for g in grids]

    # where each drawing's own strokes reached, vertically - the box may
    # grow past this by GRID_GROW_PITCH and no further
    limits = [[g[1] - grow, g[3] + grow] for g in grids]

    def claims(index, region):
        """Does drawing `index` own `region`?"""

        box = boxes[index]

        if not overlaps(region, box):
            return False

        top, bottom = limits[index]

        centre = (region[1] + region[3]) / 2

        return top <= centre <= bottom

    claimed = [False] * len(lines)

    changed = True

    while changed:

        changed = False

        for index, line in enumerate(lines):

            if claimed[index]:
                continue

            for position, box in enumerate(boxes):

                if not claims(position, line["bbox"]):
                    continue

                claimed[index] = True

                x1, y1, x2, y2 = line["bbox"]
                box[0] = min(box[0], x1)
                box[1] = min(box[1], y1)
                box[2] = max(box[2], x2)
                box[3] = max(box[3], y2)

                changed = True
                break

    # two grids that grew into each other are one grid
    merged = True

    while merged:

        merged = False

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not overlaps(boxes[i], boxes[j]):
                    continue
                boxes[i] = [min(boxes[i][0], boxes[j][0]),
                            min(boxes[i][1], boxes[j][1]),
                            max(boxes[i][2], boxes[j][2]),
                            max(boxes[i][3], boxes[j][3])]
                limits[i] = [min(limits[i][0], limits[j][0]),
                             max(limits[i][1], limits[j][1])]
                boxes.pop(j)
                limits.pop(j)
                merged = True
                break
            if merged:
                break

    kept = [line for index, line in enumerate(lines) if not claimed[index]]

    for box in boxes:
        kept.append({"bbox": [int(v) for v in box],
                     "parts": 1, "tall": True, "grid": True})

    kept.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    return kept


def group_blocks(lines, pitch):
    """Consecutive lines separated by less than the gap are one block."""

    if not lines:
        return []

    threshold = BLOCK_GAP_PITCH * pitch

    blocks = []
    current = [lines[0]]

    for line in lines[1:]:

        if line["bbox"][1] - current[-1]["bbox"][3] > threshold:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)

    blocks.append(current)

    grouped = []

    for members in blocks:
        grouped.append({
            "bbox": [min(m["bbox"][0] for m in members),
                     min(m["bbox"][1] for m in members),
                     max(m["bbox"][2] for m in members),
                     max(m["bbox"][3] for m in members)],
            "lines": members,
        })

    return grouped


def segment_page(path):
    """Returns a page record, or None if unreadable."""

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        return None, None, None

    mask = ink_mask(gray)

    clean, rules, pitch, rule_rows = split_rules(mask)

    boxes = find_regions(clean, pitch)
    lines = find_lines(boxes, pitch, rule_rows)

    grids = find_grids(clean, pitch)
    lines = merge_grids(lines, grids, pitch)

    blocks = group_blocks(lines, pitch)

    # what fraction of the handwriting ended up inside a line box -
    # the number that says whether anything is being dropped
    covered = np.zeros_like(clean)

    for line in lines:
        x1, y1, x2, y2 = line["bbox"]
        covered[y1:y2, x1:x2] = 255

    total_ink = int((clean > 0).sum())

    inside = int(((clean > 0) & (covered > 0)).sum())

    record = {
        "size": [int(gray.shape[1]), int(gray.shape[0])],
        "rule_pitch": round(pitch, 2),
        "rules_removed": int((rules > 0).sum()),
        "grids": len(grids),
        "ink_pixels": total_ink,
        "ink_covered": round(inside / total_ink, 4) if total_ink else 0.0,
        "blocks": blocks,
    }

    return record, gray, rules


def render(gray, rules, record):
    """The page with blocks and lines drawn on it."""

    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ghost the removed rules so it is obvious what the segmenter did
    # not see, rather than leaving the viewer to guess
    canvas[rules > 0] = RULE_TINT

    for block in record["blocks"]:
        x1, y1, x2, y2 = block["bbox"]
        cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                      BLOCK_COLOUR, 2)

        for line in block["lines"]:
            lx1, ly1, lx2, ly2 = line["bbox"]
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), LINE_COLOUR, 1)

    return canvas


def page_id(path):

    student = int(re.search(r"student_(\d+)", path.parts[-3]).group(1))
    cie = int(re.search(r"cie_(\d+)", path.parts[-2]).group(1))
    number = int(re.search(r"(\d+)", path.stem).group(1))

    return f"s{student:02d}_c{cie}_p{number:02d}", student, cie, number


def content_pages():
    """Every page except the identity cover sheets."""

    pages = []

    for path in sorted(SOURCE_DIR.glob("student_*/cie_*/page_*.png")):

        if int(re.search(r"(\d+)", path.stem).group(1)) == COVER_PAGE:
            continue

        pages.append(str(path.relative_to(SOURCE_DIR)))

    return pages


def _worker(args):

    relative, write_image = args

    path = SOURCE_DIR / relative

    record, gray, rules = segment_page(path)

    if record is None:
        return relative, None, "unreadable"

    identifier, student, cie, number = page_id(path)

    record["page_id"] = identifier
    record["student"] = student
    record["cie"] = cie
    record["page"] = number
    record["source"] = relative

    if write_image:
        target = ANNOTATED_DIR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), render(gray, rules, record))

    return relative, record, None


def write_outputs(records):
    """JSON hierarchy plus flat tables, for whatever the next module wants."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "segmentation.json", "w") as handle:
        json.dump(records, handle, indent=1)

    with open(OUT_DIR / "pages.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["page_id", "student", "cie", "page", "width",
                         "height", "rule_pitch", "blocks", "lines",
                         "ink_covered", "source"])
        for r in records:
            writer.writerow([
                r["page_id"], r["student"], r["cie"], r["page"],
                r["size"][0], r["size"][1], r["rule_pitch"],
                len(r["blocks"]), sum(len(b["lines"]) for b in r["blocks"]),
                r["ink_covered"], r["source"],
            ])

    with open(OUT_DIR / "blocks.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["page_id", "block_id", "x1", "y1", "x2", "y2",
                         "lines"])
        for r in records:
            for index, block in enumerate(r["blocks"]):
                writer.writerow([r["page_id"], index, *block["bbox"],
                                 len(block["lines"])])

    with open(OUT_DIR / "lines.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["page_id", "block_id", "line_id", "x1", "y1",
                         "x2", "y2", "tall"])
        for r in records:
            for bindex, block in enumerate(r["blocks"]):
                for lindex, line in enumerate(block["lines"]):
                    writer.writerow([r["page_id"], bindex, lindex,
                                     *line["bbox"], int(line["tall"])])


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-images", action="store_true",
                        help="skip the annotated renders")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))

    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        raise SystemExit(f"source not found: {SOURCE_DIR}")

    pages = content_pages()

    if args.limit:
        pages = pages[:args.limit]

    if not pages:
        raise SystemExit(f"no pages under {SOURCE_DIR}")

    write_image = not args.no_images

    print(f"Segmenting {len(pages)} content page(s) "
          f"with {args.workers} worker(s)")
    print(f"Cover pages (page_{COVER_PAGE:02d}) skipped\n")

    records = []
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        for index, (relative, record, error) in enumerate(
                pool.map(_worker, [(p, write_image) for p in pages],
                         chunksize=8), start=1):

            if index % 200 == 0:
                print(f"  {index}/{len(pages)}", flush=True)

            if error:
                failures.append((relative, error))
            else:
                records.append(record)

    records.sort(key=lambda r: r["page_id"])

    write_outputs(records)

    lines = [sum(len(b["lines"]) for b in r["blocks"]) for r in records]
    blocks = [len(r["blocks"]) for r in records]
    covered = [r["ink_covered"] for r in records]
    pitches = [r["rule_pitch"] for r in records]

    print(f"\nPages     : {len(records)}")
    print(f"Blocks    : median {int(np.median(blocks))} per page")
    print(f"Lines     : median {int(np.median(lines))} per page, "
          f"{sum(lines)} total")
    print(f"Rule pitch: median {np.median(pitches):.1f} px "
          f"(min {min(pitches):.1f}, max {max(pitches):.1f})")
    print(f"Ink inside a line box: median {np.median(covered) * 100:.1f}%, "
          f"worst {min(covered) * 100:.1f}%")

    print(f"\nGeometry  : {OUT_DIR}")

    if write_image:
        print(f"Annotated : {ANNOTATED_DIR}")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures[:5]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
