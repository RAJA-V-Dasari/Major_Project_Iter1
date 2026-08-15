"""
Step 1: measure the structural priors of a ruled answer page.

Everything downstream depends on these three numbers being right, so
this stage does nothing else:

    pitch        vertical spacing of ruled lines / text baselines, px
    margin_x     x-position of the printed left margin line, px
    skew_angle   page rotation, degrees

Method notes (these were chosen against the real scans, not assumed):

  * Pitch comes from the AUTOCORRELATION of the horizontal ink
    projection, not from detecting individual rule lines. These are
    phone-camera scans with curved pages, so a rule drifts across
    several rows and a straight-kernel line detector misses most of
    them (it found 2 of ~25 on one page). Autocorrelation measures the
    periodicity of the whole page at once and is unbothered by that
    curvature.

  * `pitch_strength` (the autocorrelation peak height) doubles as a
    page-type signal: ruled answer pages score ~0.5-0.65, while the
    printed cover sheet scores ~0.23 because it has no uniform ruling.

  * The margin probe is deliberately short (height/60) and its
    threshold low, because the margin line curves too.

Run:
    python normalize_page.py --visualize
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


INPUT_DIR = Path("input/pages")
OUTPUT_DIR = Path("output")

JSON_PATH = OUTPUT_DIR / "structure.json"
DEBUG_DIR = OUTPUT_DIR / "normalize_debug"

# Every pixel-length threshold in this module and its siblings was
# measured on 2480x3509 scans (A4 at 300 DPI). The main corpus is 200
# DPI (1700x2338), so the same physical feature is 1.46x smaller there
# and a raw pixel threshold silently means something different.
#
# Lengths are therefore declared at this reference width and scaled to
# the page actually being processed, via `scaled()`. Fractions of the
# image dimensions need no such treatment and are used directly
# wherever they suffice.
REFERENCE_WIDTH = 2480


def scaled(length, width):
    """A length tuned at REFERENCE_WIDTH, expressed for this page."""

    return max(1, int(round(length * width / REFERENCE_WIDTH)))


# Plausible range for ruled-line spacing, at REFERENCE_WIDTH (px).
PITCH_MIN = 40
PITCH_MAX = 250

# A lag counts as the fundamental if it reaches this share of the best
# peak. Guards against locking onto the 2x harmonic, which happens when
# alternate rules are left blank.
HARMONIC_TOLERANCE = 0.85

# Below this autocorrelation peak the page has no dependable ruling
# (cover sheets, near-empty pages) and pitch should not be trusted.
MIN_PITCH_STRENGTH = 0.30

# Margin line probe: short, because the printed margin curves on these
# scans, and a long kernel simply misses it.
MARGIN_PROBE_FRACTION = 60
MARGIN_MIN_HEIGHT_FRACTION = 0.15

# The margin sits this far into the page; a vertical line outside this
# band is a scan edge or a table border, not the margin.
MARGIN_MIN_X_FRACTION = 0.03
MARGIN_MAX_X_FRACTION = 0.35

# Deskew search. The scans are near-upright; the residual is curvature,
# which rotation cannot fix, so a wide sweep would be false precision.
SKEW_LIMIT = 3.0
SKEW_STEP = 0.25
SKEW_DOWNSCALE = 4


def ink_mask(image_bgr):
    """Binarise so ink is white (255) and paper black."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )


def horizontal_profile(mask):
    """Ink per row - the horizontal projection histogram."""

    return mask.sum(axis=1).astype(np.float64) / 255.0


# Probe length for finding horizontal strokes, as a fraction of page
# width. Long enough that handwriting rarely qualifies, short enough to
# follow the curvature of these phone scans.
STROKE_PROBE_FRACTION = 25

# A horizontal stroke reaching this share of the page width is printed
# ruling; anything shorter is a pen stroke (underline or strikethrough).
RULE_WIDTH_FRACTION = 0.50

# Curvature breaks one printed rule into several fragments, none of them
# wide enough to pass the test above. Fragments are bridged across gaps
# this wide before their width is measured, so a broken rule is judged
# as the single line it really is.
#
# Lowering RULE_WIDTH_FRACTION instead does not work: a strikethrough
# over a long expression is itself ~25% of the page, so it gets eaten as
# ruling and the cancellation is missed entirely.
RULE_BRIDGE_GAP = 120


def split_horizontal_strokes(mask):
    """
    Separate horizontal ink into printed rules and pen strokes.

    The rules must come out of the mask before any projection profile
    is useful: they put ink on every ruled row, so the profile has no
    valleys and blocks merge into one page-sized lump.

    The short strokes are kept rather than discarded, because a
    strikethrough is exactly a short dense horizontal stroke sitting
    inside a line of writing - the primary `crossed_out` signal.

    Returns:
        (clean, rules, strokes) - all uint8 masks
    """

    height, width = mask.shape

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(1, width // STROKE_PROBE_FRACTION), 1)
    )

    horizontals = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # rejoin fragments of the same curved rule before judging length
    bridged = cv2.morphologyEx(
        horizontals,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (scaled(RULE_BRIDGE_GAP, width), 1)
        ),
    )

    rules = np.zeros_like(horizontals)
    strokes = np.zeros_like(horizontals)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        bridged, connectivity=8
    )

    minimum_rule_width = RULE_WIDTH_FRACTION * width

    for label in range(1, count):

        # keep only the real ink, not the bridging pixels
        component = (labels == label) & (horizontals > 0)

        if stats[label, cv2.CC_STAT_WIDTH] >= minimum_rule_width:
            rules[component] = 255
        else:
            strokes[component] = 255

    # grow the rules slightly so their full printed thickness goes,
    # leaving no residue to re-connect blocks
    thick_rules = cv2.dilate(
        rules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    )

    clean = cv2.subtract(mask, thick_rules)

    return clean, rules, strokes


def estimate_pitch(mask):
    """
    Dominant vertical period of the page, via autocorrelation of the
    horizontal projection profile.

    Returns:
        (pitch_px | None, peak_strength)
    """

    profile = horizontal_profile(mask)

    centred = profile - profile.mean()

    if not np.any(centred):
        return None, 0.0

    correlation = np.correlate(centred, centred, mode="full")[len(centred) - 1:]

    if correlation[0] <= 0:
        return None, 0.0

    correlation = correlation / correlation[0]

    page_width = mask.shape[1]

    pitch_min = scaled(PITCH_MIN, page_width)
    pitch_max = scaled(PITCH_MAX, page_width)

    high = min(pitch_max, len(correlation) - 1)

    if high <= pitch_min:
        return None, 0.0

    window = correlation[pitch_min:high]

    best_lag = int(np.argmax(window)) + pitch_min
    best_value = float(correlation[best_lag])

    # Prefer the earliest lag that is nearly as strong as the best one,
    # so a 2x harmonic does not masquerade as the true pitch.
    fundamental = best_lag

    for lag in range(pitch_min, best_lag):

        if correlation[lag] >= HARMONIC_TOLERANCE * best_value:
            fundamental = lag
            break

    return fundamental, float(correlation[fundamental])


def estimate_phase(mask, pitch):
    """
    Where the pitch grid actually starts (px offset of the first text
    row). Pitch alone gives the spacing; without phase the grid is
    drawn in an arbitrary place and cannot be checked by eye or used to
    snap block boundaries.
    """

    if not pitch:
        return 0

    profile = horizontal_profile(mask)

    best_offset = 0
    best_total = -1.0

    for offset in range(int(pitch)):

        total = float(profile[offset::int(pitch)].sum())

        if total > best_total:
            best_total = total
            best_offset = offset

    return best_offset


def estimate_margin(mask):
    """x-position of the printed left margin line, or None."""

    height, width = mask.shape

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(1, height // MARGIN_PROBE_FRACTION))
    )

    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    column_ink = vertical.sum(axis=0) / 255.0

    threshold = MARGIN_MIN_HEIGHT_FRACTION * height

    candidates = np.where(column_ink >= threshold)[0]

    if len(candidates) == 0:
        return None

    low = MARGIN_MIN_X_FRACTION * width
    high = MARGIN_MAX_X_FRACTION * width

    in_band = [x for x in candidates if low < x < high]

    if not in_band:
        return None

    # take the strongest column in the band, then report the centre of
    # its run so a 4px-thick line does not bias the position
    strongest = max(in_band, key=lambda x: column_ink[x])

    run = [x for x in in_band if abs(x - strongest) <= 6]

    return int(np.mean(run))


def estimate_skew(mask):
    """
    Page rotation, in degrees, found by maximising the roughness of the
    horizontal projection: when text rows align with image rows, the
    profile has the sharpest peaks and valleys.
    """

    small = cv2.resize(
        mask,
        (mask.shape[1] // SKEW_DOWNSCALE, mask.shape[0] // SKEW_DOWNSCALE),
    )

    centre = (small.shape[1] / 2, small.shape[0] / 2)

    best_angle = 0.0
    best_score = -1.0

    for angle in np.arange(-SKEW_LIMIT, SKEW_LIMIT + 1e-9, SKEW_STEP):

        matrix = cv2.getRotationMatrix2D(centre, float(angle), 1.0)

        rotated = cv2.warpAffine(
            small, matrix, (small.shape[1], small.shape[0])
        )

        score = float(np.var(np.diff(rotated.sum(axis=1).astype(np.float64))))

        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return best_angle


def analyze_page(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    height, width = image.shape[:2]

    mask = ink_mask(image)

    pitch, strength = estimate_pitch(mask)
    phase = estimate_phase(mask, pitch)
    margin_x = estimate_margin(mask)
    skew = estimate_skew(mask)

    ruled = strength >= MIN_PITCH_STRENGTH

    return {
        "width": width,
        "height": height,
        "pitch": pitch,
        "pitch_phase": phase,
        "pitch_strength": round(strength, 4),
        # False for cover sheets and near-empty pages, where the pitch
        # figure exists but means nothing
        "is_ruled_page": bool(ruled),
        "margin_x": margin_x,
        "skew_angle": round(skew, 3),
    }, image, mask


def annotate(image, structure, output_path):
    """Draw the measured priors so they can be checked by eye."""

    vis = image.copy()

    height, width = structure["height"], structure["width"]

    pitch = structure["pitch"]

    if pitch and structure["is_ruled_page"]:

        # phase-aligned so the grid should land on the real text rows -
        # that alignment is the point of the visual check
        for y in range(structure["pitch_phase"], height, int(pitch)):
            cv2.line(vis, (0, y), (width, y), (0, 0, 255), 2)

    if structure["margin_x"] is not None:
        cv2.line(
            vis,
            (structure["margin_x"], 0),
            (structure["margin_x"], height),
            (255, 0, 0),
            6,
        )

    small = cv2.resize(vis, (width // 3, height // 3))

    cv2.imwrite(str(output_path), small)


def page_sort_key(path):

    digits = "".join(c for c in path.stem if c.isdigit())

    return int(digits) if digits else 0


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    images = sorted(INPUT_DIR.glob("*.png"), key=page_sort_key)

    if not images:
        print(f"No images found in {INPUT_DIR}")
        return

    if args.visualize:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    document = {}

    for image_path in images:

        structure, image, _ = analyze_page(image_path)

        flag = "" if structure["is_ruled_page"] else "   (not ruled)"

        print(
            f"{image_path.name:>14}: pitch={structure['pitch']} "
            f"strength={structure['pitch_strength']:.2f} "
            f"margin_x={structure['margin_x']} "
            f"skew={structure['skew_angle']:+.2f}{flag}"
        )

        document[image_path.name] = structure

        if args.visualize:
            annotate(image, structure, DEBUG_DIR / image_path.name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "w") as f:
        json.dump(document, f, indent=2)

    ruled = sum(1 for s in document.values() if s["is_ruled_page"])

    print()
    print(f"Pages       : {len(document)}")
    print(f"Ruled pages : {ruled}")
    print(f"Saved       : {JSON_PATH}")


if __name__ == "__main__":
    main()
