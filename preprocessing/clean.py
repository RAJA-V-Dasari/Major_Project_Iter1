"""
Clean the normalised scans into a uniform, OCR-ready page set.

    output/  (normalised: 1700x2338 PNG)  ->  cleaned/  (same, but clean)

Four stages, in this order, because each depends on the last:

1. DESKEW using the printed rule lines
   The rules are the one landmark on every page, and they are dead
   straight, so their angle IS the page angle. Detected with a
   probabilistic Hough transform over horizontally-opened ink.

   Hough was chosen over two alternatives after measuring all three on
   the same pages:
     - connected components of long horizontals found between 2 and 37
       rules per page, because page curvature breaks a rule into
       fragments that individually fail a width test;
     - a rotation sweep maximising projection variance agreed with
       Hough to within 0.11 degrees but is quantised to the sweep step
       and ~80x more work.
   Hough finds 67-138 segments on every page tested and returns a
   continuous angle.

2. TRIM the scanner lip
   A thin dark band where the scanner meets the page edge. Found by
   walking inward from each border over a rolling minimum of the
   row/column means.

3. FLATTEN and TONE
   Divide out the illumination gradient, then a linear stretch that
   puts ink at black and paper - including bleed-through from the
   reverse side - at white.

4. CANONICALISE the size
   Every page comes out at exactly CANONICAL_SIZE. Trimming removes a
   different number of pixels per page, so the trimmed pages are
   pad-centred back to one size rather than resized: padding keeps the
   scale exact, and the rule pitch (measured at a consistent 59-60 px
   across the corpus) stays comparable between pages. Resizing would
   quietly make it differ.

Run:
    python clean.py --preview        # before/after pairs, no writes
    python clean.py                  # clean everything
    python clean.py --verify         # check the written output
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "output"
CLEAN_DIR = BASE_DIR / "cleaned"

TRANSFORMS_PATH = CLEAN_DIR / "transforms.json"
PREVIEW_DIR = BASE_DIR / "clean_preview"

# Every cleaned page is exactly this. The normalised input is already
# 1700x2338, and cleaning only ever removes border, so this is reached
# by padding and never by scaling.
CANONICAL_SIZE = (1700, 2338)

# A page smaller than this share of canonical in either dimension is
# treated as a low-resolution outlier and scaled up before padding,
# so its rule pitch matches the rest of the corpus.
UNDERSIZED_FRACTION = 0.90

# --- deskew ---------------------------------------------------------
# Rules run the width of the page; this opening keeps them and drops
# handwriting.
RULE_KERNEL = 60

# A page is never tilted more than a couple of degrees; anything beyond
# this is a mis-detection, not a rotation.
MAX_SKEW = 4.0

# Hough needs enough collinear ink to call something a line.
HOUGH_THRESHOLD = 200
HOUGH_MIN_LENGTH_FRACTION = 5
HOUGH_MAX_GAP = 40

# Below this many agreeing segments the angle is not trustworthy and
# the page is left unrotated.
MIN_RULE_SEGMENTS = 12

# --- trim -----------------------------------------------------------
# A row/column is paper above this share of the page median.
#
# 0.60 was tried first and only caught the darkest core of the edge.
# The scanner shadow is a GRADIENT, not a sharp band: on
# student_01/cie_2/page_09 the top ramps 106 -> 243 over 18 rows and
# the left ramps 145 -> 227 over ~30 columns, so a low threshold trims
# the black and leaves a grey ledge behind. 0.85 of the median cuts
# the whole ramp. Paper edges on clean pages sit at 0.93+ of median
# and are still kept.
# 0.85 still stopped three rows short on student_10/cie_3/page_03,
# whose top ramps 63 -> 205; 0.90 takes the tail as well. Clean page
# edges sit at 0.93+ of the median and survive, so this is the last
# notch before real paper starts being eaten.
EDGE_BRIGHTNESS_RATIO = 0.90

# Rolling-minimum window, so one bright line inside the lip cannot stop
# the walk. Without it, a white final row left 15 dark rows in place.
EDGE_SMOOTHING = 9

# Applied inward. Expanding outward simply puts the lip back.
TRIM_MARGIN = 4

# --- booklet bottom edge --------------------------------------------
# Search only this far down the page. The edge measures y 2197-2209 on
# a 2338-row page; searching higher returns handwriting instead.
BOOKLET_EDGE_SEARCH_FROM = 0.93

# A row this much darker than the page median is the edge.
BOOKLET_EDGE_DARKNESS = 0.85

# Cut this far ABOVE the detected edge.
#
# 6 was not enough: the edge casts its own soft shadow just above it,
# which is thin and often DIAGONAL after deskew, so it barely moves a
# full-width row mean and slipped through every row-based test. It
# showed up as a 260x5 px streak in the corner of
# student_02/cie_2/page_10.
#
# Measured headroom between the edge and the last real content is
# 56-107 px across the corpus, so 35 clears the shadow with room to
# spare and never reaches the page number or the seal.
BOOKLET_EDGE_MARGIN = 35

# --- tone -----------------------------------------------------------
# Background estimate; must exceed stroke width or strokes become their
# own background and vanish.
BACKGROUND_KERNEL = 51

# Ink under 110, ruling 120-200, bleed-through 215-245. PAPER_WHITE
# sits above the ruling so the rules survive and the bleed-through
# does not.
INK_BLACK = 90
PAPER_WHITE = 215


def rule_angle(gray):
    """
    Page tilt in degrees, from the printed rule lines.

    Returns (angle, segment_count). angle is 0.0 when there is not
    enough evidence - refusing to rotate beats rotating on noise.
    """

    height, width = gray.shape

    ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    horizontals = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (RULE_KERNEL, 1)),
    )

    lines = cv2.HoughLinesP(
        horizontals,
        rho=1,
        theta=np.pi / 1440,          # 0.125 degree resolution
        threshold=HOUGH_THRESHOLD,
        minLineLength=width // HOUGH_MIN_LENGTH_FRACTION,
        maxLineGap=HOUGH_MAX_GAP,
    )

    if lines is None:
        return 0.0, 0

    angles = []

    for line in lines:

        x1, y1, x2, y2 = (float(v) for v in line.ravel()[:4])

        if x2 == x1:
            continue

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if abs(angle) <= MAX_SKEW:
            angles.append(angle)

    if len(angles) < MIN_RULE_SEGMENTS:
        return 0.0, len(angles)

    # median, not mean: a few handwriting strokes survive the opening
    # and would drag an average
    return round(float(np.median(angles)), 3), len(angles)


def trim_edges(means, floor):
    """Walk inward from both ends while the line is darker than `floor`."""

    count = len(means)

    half = EDGE_SMOOTHING // 2

    padded = np.pad(means, (half, half), mode="edge")

    smoothed = np.lib.stride_tricks.sliding_window_view(
        padded, EDGE_SMOOTHING
    ).min(axis=1)

    start = 0
    while start < count and smoothed[start] < floor:
        start += 1

    end = count - 1
    while end > start and smoothed[end] < floor:
        end -= 1

    return start, end


def booklet_bottom(gray):
    """
    Row where the booklet's bottom edge sits, or None.

    This is a separate detector from the generic lip trim because the
    two look nothing alike. Below the booklet edge the scan continues
    onto the desk/next sheet, which is BRIGHT - so every
    brightness-based search walks straight past it, and trimming
    inward from row H-1 stops on that white area without ever reaching
    the edge. Measured: the edge lands at y 2197-2209 on every page
    sampled, with 129-141 rows of off-page below it.

    Found by scanning only the bottom slice for the first
    markedly-dark row. Restricting the search is what makes it safe:
    the same rule applied to the whole page happily returns a line of
    handwriting at y~1780.
    """

    height, width = gray.shape

    start = int(height * BOOKLET_EDGE_SEARCH_FROM)

    means = gray[start:].mean(axis=1)

    floor = BOOKLET_EDGE_DARKNESS * float(np.median(gray))

    dark = np.where(means < floor)[0]

    if len(dark) == 0:
        return None

    return start + int(dark[0])


def paper_bbox(gray):
    """Bounding box of the sheet with the scanner lip removed."""

    height, width = gray.shape

    floor = EDGE_BRIGHTNESS_RATIO * float(np.median(gray))

    row_start, row_end = trim_edges(gray.mean(axis=1), floor)
    col_start, col_end = trim_edges(gray.mean(axis=0), floor)

    y1 = min(height, row_start + TRIM_MARGIN)
    y2 = max(0, row_end + 1 - TRIM_MARGIN)
    x1 = min(width, col_start + TRIM_MARGIN)
    x2 = max(0, col_end + 1 - TRIM_MARGIN)

    # this much trimming means detection failed; a wrong crop is worse
    # than none
    if (x2 - x1) * (y2 - y1) < 0.5 * width * height:
        return 0, 0, width, height

    return x1, y1, x2, y2


def flatten(gray):
    """Divide out the illumination gradient; paper lands near 255."""

    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (BACKGROUND_KERNEL, BACKGROUND_KERNEL)
        ),
    )

    background = np.maximum(background, 1).astype(np.float32)

    flat = gray.astype(np.float32) / background * 255.0

    return np.clip(flat, 0, 255).astype(np.uint8)


def tone(gray):
    """
    Ink to black, paper and bleed-through to white, mid-tones stretched.

    Grey levels are kept rather than binarised: OCR reads stroke
    weight, and a hard threshold throws that away.
    """

    out = gray.astype(np.float32)

    out = (out - INK_BLACK) / float(PAPER_WHITE - INK_BLACK)

    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def canonicalise(gray):
    """
    Pad (or centre-crop) to exactly CANONICAL_SIZE.

    Padding rather than resizing: trimming removes a different amount
    per page, and resizing to compensate would change the scale by a
    percent or so per page, making the rule pitch - the one physical
    constant in this corpus - differ between pages.

    The exception is a page that arrives far smaller than canonical.
    student_19/cie_2/page_14 is stored at 637x876, about 37% scale,
    the single odd page in an otherwise 1700x2338 corpus. Padding that
    one would give it the right dimensions but a rule pitch near 22px
    against 59px everywhere else - uniform in size while being wrong
    in scale, which is worse than obvious breakage because it looks
    fine. So a markedly undersized page is scaled up to fit first, and
    only then padded.
    """

    target_w, target_h = CANONICAL_SIZE

    height, width = gray.shape

    if (width < UNDERSIZED_FRACTION * target_w
            or height < UNDERSIZED_FRACTION * target_h):

        scale = min(target_w / width, target_h / height)

        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))),
             max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )

        height, width = gray.shape

    # centre-crop anything larger than the target (does not occur with
    # the current input, but a different source should not corrupt)
    if width > target_w:
        left = (width - target_w) // 2
        gray = gray[:, left:left + target_w]
        width = target_w

    if height > target_h:
        top = (height - target_h) // 2
        gray = gray[top:top + target_h, :]
        height = target_h

    canvas = np.full((target_h, target_w), 255, np.uint8)

    x = (target_w - width) // 2
    y = (target_h - height) // 2

    canvas[y:y + height, x:x + width] = gray

    return canvas, (x, y)


def clean_page(path):
    """Returns (cleaned_page, transform) or (None, None)."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None, None

    height, width = image.shape

    angle, segments = rule_angle(image)

    if angle != 0.0:

        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

        image = cv2.warpAffine(
            image, matrix, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )

    x1, y1, x2, y2 = paper_bbox(image)

    # the booklet edge sits above the generic lip, with bright off-page
    # area below it, so it needs its own detector and the tighter of
    # the two bottoms wins
    edge = booklet_bottom(image)

    if edge is not None:
        y2 = min(y2, max(0, edge - BOOKLET_EDGE_MARGIN))

    image = image[y1:y2, x1:x2]

    image = flatten(image)
    image = tone(image)

    # single-pixel speckle only; anything larger is real ink
    image = cv2.medianBlur(image, 3)

    trimmed_h, trimmed_w = image.shape

    image, offset = canonicalise(image)

    transform = {
        "angle": angle,
        "rule_segments": segments,
        "crop": [int(x1), int(y1), int(x2), int(y2)],
        "booklet_edge": edge,
        "source_size": [int(width), int(height)],
        "trimmed_size": [int(trimmed_w), int(trimmed_h)],
        "pad_offset": [int(offset[0]), int(offset[1])],
        "final_size": list(CANONICAL_SIZE),
    }

    return image, transform


def _worker(relative):

    cleaned, transform = clean_page(SOURCE_DIR / relative)

    if cleaned is None:
        return relative, None, "unreadable"

    target = CLEAN_DIR / relative

    target.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(target), cleaned)

    return relative, transform, None


def page_list():

    return [
        str(p.relative_to(SOURCE_DIR))
        for p in sorted(SOURCE_DIR.glob("student_*/cie_*/page_*.png"))
    ]


def preview():

    picks = [
        "student_57/cie_1/page_08.png",
        "student_23/cie_2/page_05.png",
        "student_09/cie_2/page_07.png",
        "student_35/cie_3/page_09.png",
        "student_17/cie_3/page_08.png",
        "student_34/cie_3/page_06.png",
    ]

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for relative in picks:

        source = SOURCE_DIR / relative

        if not source.exists():
            continue

        before = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)

        after, transform = clean_page(source)

        if after is None:
            continue

        # a few source pages are 2339 rather than 2338 - per-page
        # rounding in the original conversion, and exactly what
        # canonicalise() exists to remove
        height = max(before.shape[0], after.shape[0])

        def pad(image):
            out = np.full((height, image.shape[1]), 255, np.uint8)
            out[:image.shape[0]] = image
            return out

        gap = np.full((height, 14), 90, np.uint8)

        pair = np.hstack([pad(before), gap, pad(after)])

        cv2.imwrite(str(PREVIEW_DIR / relative.replace("/", "_")), pair)

        print(f"  {relative:<32} angle={transform['angle']:+.3f} "
              f"({transform['rule_segments']} segs)  "
              f"trim {transform['source_size']}->{transform['trimmed_size']}"
              f"  final {transform['final_size']}")

    print(f"\nBefore | after pairs: {PREVIEW_DIR}")


def verify():
    """Confirm every written page really is the same size."""

    sizes = {}

    unreadable = []

    pages = sorted(CLEAN_DIR.glob("student_*/cie_*/page_*.png"))

    if not pages:
        raise SystemExit(f"nothing under {CLEAN_DIR}")

    for path in pages:

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            unreadable.append(str(path))
            continue

        sizes[(image.shape[1], image.shape[0])] = \
            sizes.get((image.shape[1], image.shape[0]), 0) + 1

    print(f"Pages checked : {len(pages)}")
    print("Sizes present :")

    for size, count in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"  {size[0]} x {size[1]}   {count}")

    if unreadable:
        print(f"\nUNREADABLE: {len(unreadable)} - {unreadable[:5]}")

    if len(sizes) == 1 and not unreadable:
        print("\nUniform: every page is the same size.")
        return 0

    print("\nNOT uniform.")
    return 1


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))

    args = parser.parse_args()

    if args.preview:
        preview()
        return 0

    if args.verify:
        return verify()

    pages = page_list()

    if args.limit:
        pages = pages[:args.limit]

    if not pages:
        raise SystemExit(f"no pages under {SOURCE_DIR}")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Cleaning {len(pages)} page(s) with {args.workers} worker(s)")
    print(f"Canonical size: {CANONICAL_SIZE[0]} x {CANONICAL_SIZE[1]}\n")

    transforms = {}
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        for index, (relative, transform, error) in enumerate(
                pool.map(_worker, pages, chunksize=8), start=1):

            if index % 200 == 0:
                print(f"  {index}/{len(pages)}", flush=True)

            if error:
                failures.append((relative, error))
            else:
                transforms[relative] = transform

    with open(TRANSFORMS_PATH, "w") as handle:
        json.dump(transforms, handle, indent=1)

    angles = [t["angle"] for t in transforms.values()]
    segments = [t["rule_segments"] for t in transforms.values()]

    rotated = sum(1 for a in angles if a != 0.0)
    weak = sum(1 for s in segments if s < MIN_RULE_SEGMENTS)

    print(f"\nCleaned : {len(transforms)}")
    print(f"Rotated : {rotated}")
    print(f"Skew    : min {min(angles):+.2f}  max {max(angles):+.2f}")
    print(f"Rule segments per page: median {int(np.median(segments))}, "
          f"min {min(segments)}")

    if weak:
        print(f"  {weak} page(s) had too few rules to trust an angle "
              f"and were left unrotated")

    print(f"\nOutput     : {CLEAN_DIR}")
    print(f"Transforms : {TRANSFORMS_PATH}")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures[:5]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
