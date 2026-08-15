"""
Per-block measurements used to classify a region.

Kept separate from the classifier so the thresholds in
classify_blocks.py can be tuned - or replaced by a learned model -
without touching how the numbers are produced.

Every feature is scale-free (a ratio, or a multiple of the page's
measured pitch) so it transfers across scan resolutions.
"""

import cv2
import numpy as np

from normalize_page import split_horizontal_strokes, ink_mask, scaled


# Vertical stroke probe, as a fraction of page height. Table column
# dividers are short; the page margin is long. Both are found here and
# the margin is excluded by position afterwards.
VSTROKE_PROBE_FRACTION = 60

# Columns within this many px of the measured margin are ignored when
# looking for table dividers - otherwise the margin line crosses every
# ruled row and every page looks like a grid. Declared at
# normalize_page.REFERENCE_WIDTH and scaled to the page.
MARGIN_EXCLUSION = 25

# A column of the block counts as "blank" below this share of its
# busiest column; runs of blank columns are the gaps between table
# columns.
BLANK_COLUMN_THRESHOLD = 0.05


def page_masks(image_bgr, margin_x=None):
    """
    Produce every mask the features need, once per page.

    Returns dict with:
        raw     all ink
        clean   ink minus the printed ruling
        rules   the printed ruling (and table borders, which are also
                long horizontals)
        strikes short horizontal pen strokes: underlines/strikethroughs
        verts   vertical strokes, margin line excluded
    """

    raw = ink_mask(image_bgr)

    clean, rules, strikes = split_horizontal_strokes(raw)

    height, width = raw.shape

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(1, height // VSTROKE_PROBE_FRACTION))
    )

    verts = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)

    if margin_x is not None:
        exclusion = scaled(MARGIN_EXCLUSION, width)
        low = max(0, margin_x - exclusion)
        high = min(width, margin_x + exclusion)
        verts[:, low:high] = 0

    return {
        "raw": raw,
        "clean": clean,
        "rules": rules,
        "strikes": strikes,
        "verts": verts,
    }


def crop(mask, bbox):

    x1, y1, x2, y2 = bbox

    return mask[y1:y2, x1:x2]


def block_features(masks, bbox, pitch, page_width, margin_x):
    """
    Measure one block. Returns a dict of scale-free numbers.
    """

    x1, y1, x2, y2 = bbox

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    area = float(width * height)

    clean = crop(masks["clean"], bbox)
    strikes = crop(masks["strikes"], bbox)
    rules = crop(masks["rules"], bbox)
    verts = crop(masks["verts"], bbox)

    ink = float(np.count_nonzero(clean))

    # --- size -------------------------------------------------------
    line_span = height / pitch if pitch else 0.0

    width_fraction = width / float(page_width)

    # --- crossed out ------------------------------------------------
    # measured against the block's own ink, not its area, and split
    # from underlines so an underlined answer is not called crossed out
    strike_px, underline_px = strike_analysis(clean, strikes, pitch)

    strike_ratio = strike_px / ink if ink > 0 else 0.0
    underline_ratio = underline_px / ink if ink > 0 else 0.0

    # --- table ------------------------------------------------------
    # a grid is horizontals and verticals crossing each other; count the
    # crossings rather than the lines, so a lone underline or the margin
    # cannot fake it
    grid = cv2.bitwise_and(
        cv2.dilate(rules, np.ones((9, 9), np.uint8)),
        cv2.dilate(verts, np.ones((9, 9), np.uint8)),
    )

    crossings = cv2.connectedComponentsWithStats(grid, connectivity=8)[0] - 1

    vert_px = float(np.count_nonzero(verts))

    # --- column structure -------------------------------------------
    columns = clean.sum(axis=0).astype(np.float64) / 255.0

    if columns.max() > 0:
        blank = columns < BLANK_COLUMN_THRESHOLD * columns.max()
    else:
        blank = np.ones_like(columns, dtype=bool)

    gap_runs = count_runs(blank, minimum=max(4, int(0.02 * width)))

    # --- density and regularity -------------------------------------
    ink_density = ink / area

    rows = clean.sum(axis=1).astype(np.float64) / 255.0

    row_regularity = periodicity(rows, pitch)

    # how much of the block's own width the writing actually fills,
    # averaged over the rows that have ink - prose fills its line,
    # an isolated equation does not
    inked_rows = rows[rows > 0]

    fill = float(inked_rows.mean() / width) if inked_rows.size else 0.0

    # --- glyph shape ------------------------------------------------
    # cursive prose connects letters into wide components; maths is
    # isolated digits and operators, so its components are narrower.
    # This is the main prose/maths separator available without OCR.
    # strokes are themselves wide components, so an underlined line
    # would otherwise read as very wide glyphs and never look like maths
    glyphs = cv2.subtract(clean, strikes)

    component_width, component_height, component_count = component_shape(glyphs)

    return {
        "line_span": round(line_span, 2),
        "width_fraction": round(width_fraction, 3),
        "ink_density": round(ink_density, 4),
        "strike_ratio": round(strike_ratio, 4),
        "underline_ratio": round(underline_ratio, 4),
        "component_width": round(component_width, 1),
        "component_height": round(component_height, 1),
        "component_count": int(component_count),
        "crossings": int(crossings),
        "vert_density": round(vert_px / area, 5),
        "gap_runs": int(gap_runs),
        "row_regularity": round(row_regularity, 3),
        "fill": round(fill, 4),
        "left_offset_pitches": round(
            (x1 - margin_x) / pitch, 2
        ) if (pitch and margin_x is not None) else None,
    }


def component_shape(clean, min_area=20):
    """
    Mean size of the block's ink components, ignoring speckle.

    Returns:
        (mean_width, mean_height, count)
    """

    count, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)

    widths = []
    heights = []

    for label in range(1, count):

        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue

        widths.append(stats[label, cv2.CC_STAT_WIDTH])
        heights.append(stats[label, cv2.CC_STAT_HEIGHT])

    if not widths:
        return 0.0, 0.0, 0

    return float(np.mean(widths)), float(np.mean(heights)), len(widths)


def strike_analysis(clean, strikes, pitch, min_glyph_area=40):
    """
    Tell strikethroughs from underlines and from the bars in worked
    arithmetic.

    All three are short horizontal pen strokes, so the stroke detector
    alone cannot separate them. Conflating them mislabels underlined
    answers and whole long-division workings as crossed out.

    The test is whether the stroke PIERCES a glyph. A strikethrough
    runs through the middle of the characters it cancels, so glyphs
    extend both above and below its centre line. An underline sits
    under the writing and a division bar sits in the gap between two
    rows of digits - neither pierces anything.

    ("ink above and below" alone is not enough: a division bar has
    digits above and digits below too. It has to be the SAME glyph.)

    Returns:
        (strike_px, underline_px)
    """

    glyphs = cv2.subtract(clean, strikes)

    # Text bands of this block, measured with the strokes taken out.
    # A struck line still forms one band: the glyph fragments above and
    # below the strike are both ink at those rows.
    rows = glyphs.sum(axis=1).astype(np.float64) / 255.0

    if rows.max() <= 0:
        return 0, 0

    inked = rows >= 0.05 * rows.max()

    bands = []

    start = None

    for y, is_ink in enumerate(inked):

        if is_ink and start is None:
            start = y
        elif not is_ink and start is not None:
            bands.append((start, y))
            start = None

    if start is not None:
        bands.append((start, len(inked)))

    minimum_band = max(4, 0.15 * pitch) if pitch else 4

    bands = [(a, b) for a, b in bands if (b - a) >= minimum_band]

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        strikes, connectivity=8
    )

    strike_px = 0
    underline_px = 0

    for label in range(1, count):

        sy = stats[label, cv2.CC_STAT_TOP]
        sh = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        centre = sy + sh / 2.0

        band = next(((a, b) for a, b in bands if a <= centre <= b), None)

        if band is None:
            # sits in the whitespace between two lines: an underline, or
            # the bar of a long division - not a cancellation
            underline_px += area
            continue

        top, bottom = band

        position = (centre - top) / max(1.0, float(bottom - top))

        # through the middle of the line = struck out; riding the
        # bottom edge = underline
        if 0.25 <= position <= 0.72:
            strike_px += area
        else:
            underline_px += area

    return strike_px, underline_px


def count_runs(flags, minimum):
    """Number of runs of True at least `minimum` long."""

    runs = 0
    length = 0

    for flag in flags:

        if flag:
            length += 1
        else:
            if length >= minimum:
                runs += 1
            length = 0

    if length >= minimum:
        runs += 1

    return runs


def periodicity(profile, pitch):
    """
    How strongly a profile repeats at the page pitch. Text lines score
    high; a drawing scores low.
    """

    if not pitch or len(profile) < 2 * int(pitch):
        return 0.0

    centred = profile - profile.mean()

    if not np.any(centred):
        return 0.0

    correlation = np.correlate(centred, centred, mode="full")[len(centred) - 1:]

    if correlation[0] <= 0:
        return 0.0

    correlation = correlation / correlation[0]

    lag = int(pitch)

    window = correlation[max(1, lag - 8): lag + 9]

    return float(window.max()) if window.size else 0.0
