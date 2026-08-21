"""
Reconstruct one booklet as one image, cut at question boundaries.

    07_reconstruct/input/         (deskewed, cropped, toned pages)
    07_reconstruct/segmentation/  (02_segment's geometry - pitch only)
        -> 07_reconstruct/output/student_NN/cie_C.png            whole booklet
        -> 07_reconstruct/output/student_NN/cie_C_chunks/*.png   one file per question
        -> 07_reconstruct/output/chunks.csv   what went into each PNG

WHY MARGIN-CROSSING, NOT BLOCK GAPS
------------------------------------
02_segment's blocks are separated by blank space, and one answer is
routinely several blocks - a bullet list with a blank line between
each point is one question and three blocks. Block gaps say "these
lines are not touching"; they do not say "this is a new question".

What the student actually writes for that is a mark low in the
LEFT MARGIN, this booklet's printed vertical rule: "3 (a)", a bare
digit, or a circled number - see the pages checked for student_01,
cie_1 (page_04's "(2c)", page_05's "(3b)" and "(2a)"). It crosses the
margin rule as an ordinary printed section number would, everything
else in the booklet stays inside it. That is a strong, purely
geometric signal that needs no OCR: find ink whose box sits to the
LEFT of the margin, and every one of them is a question start.

Numbered list markers inside an answer - "(1) client", "(2) server"
on student_01/cie_1/page_05 - do NOT trigger this: they sit to the
RIGHT of the margin, same as the prose around them. Only a mark that
crosses out of the writing area counts.

WHY THIS SIDESTEPS THE LINE-FRAGMENTATION PROBLEM ENTIRELY
------------------------------------------------------------
02_segment's line/block regions still shatter a diagram that crosses
several rule rows (see 02_segment/README.md) - useful for OCR, where
every crop needs a baseline, unusable for reconstruction, where
reassembling a client-server handshake out of seven arrow fragments
would look worse than the original page. So this module never touches
line or block geometry for the crop itself. A chunk is a Y-RANGE on
the source page(s), read straight off the prepared image between one
marker and the next. A diagram, however it was internally segmented,
is just pixels in that band and comes out whole.

MARGIN POSITION IS PER BOOKLET, NOT PER PAGE
----------------------------------------------
The margin is one printed rule, physically the same position on every
page of one student's one CIE. Detecting it fresh per page is
unreliable - measured on student_01/cie_1, page_02 has no detectable
candidate at all (short of the half-page-height test, most likely
because the writing crowds the gutter) while page_04 and page_05 do.
So detection runs on every page, and pages that come up empty borrow
the booklet's median over pages that didn't. A booklet with no
detection anywhere gets a warning and is reconstructed as one
unbroken chunk, rather than guessed at.

CONTENT BEFORE THE FIRST MARKER IS KEPT, NOT DROPPED
-------------------------------------------------------
If a booklet's first marker is not at the very top of its first
content page - continued working, a preamble, a missed marker - that
ink still goes into chunk 0 rather than disappearing. Nothing here
ever discards a pixel; it only decides where the cut lines go.

Run:
    python reconstruct.py --student 1                  # one student, all CIEs
    python reconstruct.py --student 1 --cie 1           # one booklet
    python reconstruct.py                                # whole corpus
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

sys.path.insert(0, str(STAGE_DIR.parent / "02_segment" / "src"))
import segment as S  # noqa: E402  (ink_mask, split_rules, pitch/margin constants)

SOURCE_DIR = STAGE_DIR / "input"
GEOMETRY_DIR = STAGE_DIR / "segmentation"
OUT_DIR = STAGE_DIR / "output"

COVER_PAGE = 1

# A relaxed second pass if the strict (segment.py) height requirement
# finds nothing on a page. Still a real vertical stroke, just shorter -
# a heavily-written page can crowd the gutter and shorten what survives
# rule-opening. Never used to invent a margin from noise: still needs
# the same length-vs-thickness shape, just over less of the page.
MARGIN_RELAXED_HEIGHT_FRACTION = 0.22

# A question mark sits fully left of the margin, with a little slack
# for a circle that just touches it. In margin-widths, not pitch - the
# margin's distance from the edge is what bounds how big the mark can
# be.
MARKER_MAX_WIDTH_FRACTION = 0.95   # of the gutter width
MARKER_TOUCH_SLACK_PX = 6

# Marker size bounds, in pitch. A bare digit is short; "(2c)" circled
# runs about a pitch and a half. Below the floor is scanner noise or a
# stray full stop; above the ceiling is something in the gutter that
# is not a question mark - a margin doodle, a ruler smudge.
MARKER_MIN_HEIGHT_PITCH = 0.25
MARKER_MAX_HEIGHT_PITCH = 2.0
MARKER_MIN_WIDTH_PITCH = 0.12

# How far a mark's own right edge may sit short of the margin and still
# count. Real markers are written hard against the rule - measured over
# every confirmed marker on student_01/cie_1, the gap never exceeds a
# third of a pitch. A heavily-bound booklet can show a sliver of the
# facing page at the very edge of the scan (see DATASET.md's binding-seam
# note); on student_01/cie_3/page_02 that sliver produced eleven
# components that pass every other test - all of them 3+ pitch short of
# the margin, because bleed-through sits at the physical page edge, not
# beside the rule. This is what tells the two apart.
MARKER_MAX_GAP_PITCH = 0.6

# A circle and the digit inside it are usually two components; close
# the gutter mask by this much first so they merge into one marker
# rather than being counted, and later merged, separately.
MARKER_CLOSE_PITCH = 0.35

# Chunks shorter than this (both here and its neighbour once merged)
# are not a real question start - most often a stray mark the height
# filter let through anyway. Folded into the chunk before them rather
# than shown as their own sliver.
MIN_CHUNK_PITCH = 0.6

# The mark itself is not always where the question starts. A student
# writes the circle beside whichever line it lands next to, which is
# routinely a line or two below a section heading or the blank line
# that actually separates two answers - measured on student_01: "(2a)"
# sits 2.6 pitch below the blank line before "PART-B", "①" sits 3.0
# pitch below the one before "PART-A". The true cut is the nearest
# blank band above the mark, not the mark's own position.
#
# Bounded, because "nearest blank band" breaks the moment the content
# above the mark is a diagram rather than prose - a hand-drawn table
# has no blank-line convention between its rows. On
# student_01/cie_3/page_04, "(2b)" sits directly under an ARP diagram
# with no gap anywhere near it; searching unbounded walked 13 pitch
# back up through the whole diagram before finding blank paper above
# its header. GAP_MAX_SEARCH_PITCH sits above both confirmed real
# corrections (2.6, 3.0) and below that failure (13.13), so a mark with
# no genuine nearby gap keeps its own position instead of being pulled
# somewhere clearly wrong.
GAP_MIN_RUN_PITCH = 0.3     # this many consecutive blank rows is a real gap
GAP_MAX_SEARCH_PITCH = 4.0  # further than this, trust the mark itself

# Last-resort padding above a mark's top edge when the page is so dense
# that no blank row exists above it at all. See cut_above tier 3.
CUT_PAD_PX = 2

# Two marks closer than this vertically are one question start - a
# fragmented mark, or a struck-out wrong number with the right one
# rewritten on the line below. See find_markers.
MARKER_MERGE_PITCH = 1.3

SEPARATOR_PX = 10
SEPARATOR_COLOUR = (210, 150, 40)  # BGR, shows up against grey paper


def page_id(path):

    student = int(re.search(r"student_(\d+)", path.parts[-3]).group(1))
    cie = int(re.search(r"cie_(\d+)", path.parts[-2]).group(1))
    number = int(re.search(r"(\d+)", path.stem).group(1))

    return student, cie, number


def booklet_pages(student, cie):
    """Content pages of one booklet, in order. Cover page excluded."""

    pages = []

    for path in sorted(
            (SOURCE_DIR / f"student_{student:02d}" / f"cie_{cie}").glob(
                "page_*.png")):

        number = int(re.search(r"(\d+)", path.stem).group(1))

        if number == COVER_PAGE:
            continue

        pages.append((number, path))

    return pages


def _vertical_candidates(mask, height_fraction):
    """Reimplements segment._vertical_rules, but returns x, not a mask.

    Kept parametrised over height_fraction so the relaxed pass can
    reuse the exact same shape test at a lower bar, rather than a
    second unrelated heuristic.
    """

    height, width = mask.shape

    verticals = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(1, height // S.VERTICAL_PROBE_FRACTION))
        ),
    )

    bridged = cv2.morphologyEx(
        verticals, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(1, int(height * S.VERTICAL_BRIDGE_FRACTION)))
        ),
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(bridged, 8)

    best = None

    for label in range(1, count):

        if stats[label, cv2.CC_STAT_HEIGHT] < height_fraction * height:
            continue

        if best is None or stats[label, cv2.CC_STAT_HEIGHT] > best[1]:
            best = (float(centroids[label][0]), stats[label, cv2.CC_STAT_HEIGHT])

    return best[0] if best else None


def detect_margin(gray):
    """This page's margin x, or None if no candidate survives either pass."""

    mask = S.ink_mask(gray)

    x = _vertical_candidates(mask, S.VERTICAL_RULE_HEIGHT_FRACTION)

    if x is None:
        x = _vertical_candidates(mask, MARGIN_RELAXED_HEIGHT_FRACTION)

    return x


def booklet_margin(pages):
    """Per-page margin x, falling back to the booklet median where a
    page has none. Returns (margins, source) - source records which
    pages the median was borrowed from, for the manifest."""

    detected = {}

    for number, path in pages:

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            continue

        x = detect_margin(gray)

        if x is not None:
            detected[number] = x

    if not detected:
        return {}, "none detected"

    fallback = float(np.median(list(detected.values())))

    margins = {number: detected.get(number, fallback) for number, _ in pages}

    borrowed = [n for n, _ in pages if n not in detected]

    source = (f"{len(detected)}/{len(pages)} pages direct"
              + (f", {len(borrowed)} borrowed the median" if borrowed else ""))

    return margins, source


def find_markers(clean, margin_x, pitch):
    """Question-start marks: ink whose box sits left of the margin.

    Takes the rule-cleaned mask directly (not the source image) so a
    caller that also needs `clean` for gap_above_marker - which is every
    caller - is not paying for split_rules twice.
    """

    if margin_x is None:
        return []

    height, width = clean.shape

    gutter_width = margin_x - MARKER_TOUCH_SLACK_PX

    if gutter_width < 4:
        return []

    gutter = clean.copy()                 # the margin rule's own ink must
    gutter[:, int(margin_x) + 1:] = 0     # not be mistaken for a marker

    # A hand-drawn box drawn hard against the margin (the request/response
    # table on student_01/cie_1/page_02 does this) can bleed a 1-2px
    # sliver of its own border past the cutoff column. A real mark is
    # several pixels thick; open it away first, or the close below
    # bridges that sliver straight into the marker next to it and
    # inflates it well past a real mark's height.
    gutter = cv2.morphologyEx(
        gutter, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    close = max(1, int(MARKER_CLOSE_PITCH * pitch))
    gutter = cv2.morphologyEx(
        gutter, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close, close)),
    )

    count, _, stats, centroids = cv2.connectedComponentsWithStats(gutter, 8)

    markers = []

    for label in range(1, count):

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        cx = centroids[label][0]

        if x + w > margin_x + MARKER_TOUCH_SLACK_PX:
            continue  # reaches past the margin - not a marker

        if cx > margin_x:
            # A list enumerator sitting just inside the writing area
            # (student_01/cie_1/page_06's circled "1" next to "(4b)")
            # can still graze the margin with one edge of its stroke.
            # A genuine question mark is centred IN the gutter, not
            # merely touching it from the right, so the centroid - not
            # either edge - is the real test of which side it is on.
            continue

        if w > MARKER_MAX_WIDTH_FRACTION * gutter_width:
            continue  # spans the whole gutter - a stain, not a mark

        if margin_x - (x + w) > MARKER_MAX_GAP_PITCH * pitch:
            continue  # sits at the page edge, not beside the rule -
                       # facing-page bleed-through, not a mark

        if not (MARKER_MIN_HEIGHT_PITCH * pitch <= h
                <= MARKER_MAX_HEIGHT_PITCH * pitch):
            continue

        if w < MARKER_MIN_WIDTH_PITCH * pitch:
            continue

        markers.append({"top": int(y), "bottom": int(y + h),
                        "left": int(x), "right": int(x + w),
                        "height": int(h), "width": int(w)})

    markers.sort(key=lambda m: m["top"])

    # Two marks this close vertically are one question start, merged to
    # the topmost. Two causes, and neither wants its own chunk:
    #
    #   - a thin mark can open() into two pieces (an arc sliver plus
    #     the rest) that both pass every check above.
    #   - a student who writes the WRONG question number, strikes it
    #     out, and writes the right one on the line below. Both are
    #     real marks. The question still starts where the first one is,
    #     so cutting above the upper one is correct either way, and
    #     which of the two is authoritative only matters when something
    #     later reads the number - see README.
    merged = []
    for mark in markers:
        if merged and mark["top"] - merged[-1]["top"] < MARKER_MERGE_PITCH * pitch:
            previous = merged[-1]
            previous["bottom"] = max(previous["bottom"], mark["bottom"])
            previous["left"] = min(previous["left"], mark["left"])
            previous["right"] = max(previous["right"], mark["right"])
            previous["restated"] = True
            continue
        merged.append(mark)

    return merged


def writing_rows(clean, margin_x):
    """Which rows carry ink IN THE WRITING AREA - right of the margin.

    Blankness has to be judged there and nowhere else. Measured full
    width it is meaningless on any page with binding-seam bleed-through
    down the left edge: on student_01/cie_3/page_03 that strip puts ink
    at column 0 of almost every row, so the page reports 657 blank rows
    against the writing area's 875, and a cut looking for the first
    blank row above a mark walked from y=265 to y=7 before finding one.
    The gutter marks themselves would confuse it the same way.
    """

    if margin_x is None:
        return clean.any(axis=1)

    return clean[:, int(margin_x) + 1:].any(axis=1)


def cut_above(row_has_ink, marker, floor, pitch):
    """Where to cut for a question that starts at `marker`.

    Never returns a row that has ink on it. Cutting mid-line slices
    words in half - observed on student_01/cie_3/page_03, where the
    mark sits on a dense page with no paragraph break near it and the
    old code fell back to the mark's own vertical centre, straight
    through the writing it sat beside.

    Three tiers, most-preferred first:

    1. A paragraph gap - a run of blank rows at least GAP_MIN_RUN_PITCH
       tall - within GAP_MAX_SEARCH_PITCH above the mark. This is the
       blank line the student actually used to separate two answers,
       and it is usually above whatever heading precedes the question.
    2. Failing that, the nearest single blank row above the mark's TOP
       edge: the ordinary gap between two written lines. Not as
       semantically right as (1), but it is a real boundary and it
       cannot cut a word.
    3. Failing even that - writing so dense that no row above the mark
       is free of ink - the mark's own top edge, less a hair of
       padding. Worst case, and still never mid-glyph, because a
       component's top edge is by definition above all of its own ink.

    Bounded below by `floor` so a cut can never cross back over the
    previous question's start on the same page. `row_has_ink` comes
    from writing_rows - full-width blankness is not usable here.
    """

    top = int(marker["top"])
    floor = int(max(0, floor))

    # --- tier 1: a paragraph gap -------------------------------------
    min_run = max(1, int(GAP_MIN_RUN_PITCH * pitch))
    ceiling = max(floor, int(top - GAP_MAX_SEARCH_PITCH * pitch))

    run_end = run_start = None

    for row in range(top, ceiling - 1, -1):

        if not row_has_ink[row]:
            if run_end is None:
                run_end = row
            run_start = row
            continue

        if run_end is not None and (run_end - run_start + 1) >= min_run:
            return (run_start + run_end) / 2

        run_end = run_start = None

    if run_end is not None and (run_end - run_start + 1) >= min_run:
        return (run_start + run_end) / 2

    # --- tier 2: any blank row above the mark ------------------------
    # Same ceiling as tier 1: this is meant to find the ordinary gap
    # between two written lines, which is within a pitch. Unbounded, it
    # would happily run to the top of the page on a dense one.
    for row in range(top, ceiling - 1, -1):
        if not row_has_ink[row]:
            return float(row)

    # --- tier 3: the mark's own top edge -----------------------------
    return float(max(floor, top - CUT_PAD_PX))


def build_boundaries(pages, page_markers, page_heights):
    """[(page_number, y), ...] in reading order. Always starts at the
    top of the first page, so nothing before the first marker is
    dropped."""

    boundaries = [(pages[0][0], 0)]

    for number, _ in pages:
        for y in page_markers.get(number, []):
            boundaries.append((number, y))

    boundaries.append((pages[-1][0], page_heights[pages[-1][0]]))

    return boundaries


def chunk_spans(page_numbers, boundaries):
    """Consecutive boundaries -> [(chunk_index, [(page, y0, y1), ...])].

    `page_numbers` is every content page of the booklet, in order - not
    just the ones that happen to carry a boundary. A page between two
    markers that has no marker of its own is still part of the chunk
    that started before it, and must appear in its span whole; using
    only the pages boundaries mention would silently skip it.
    """

    order = {p: i for i, p in enumerate(page_numbers)}

    chunks = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):

        (p0, y0), (p1, y1) = start, end

        if order[p1] == order[p0]:
            chunks.append([(p0, y0, y1)])
        else:
            span = [(p0, y0, None)]
            for p in page_numbers[order[p0] + 1:order[p1]]:
                span.append((p, 0, None))
            span.append((p1, 0, y1))
            chunks.append(span)

    return chunks


def render_booklet(student, cie, pages):

    heights, widths, images = {}, {}, {}

    for number, path in pages:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        images[number] = gray
        heights[number] = gray.shape[0]
        widths[number] = gray.shape[1]

    if not images:
        return None, [], [], "no readable pages"

    pages = [(n, p) for n, p in pages if n in images]

    geometry = {}
    cleans = {}
    for number, _ in pages:
        rec = load_geometry(student, cie, number)
        geometry[number] = rec["rule_pitch"] if rec else S.FALLBACK_PITCH
        clean, _, _, _ = S.split_rules(S.ink_mask(images[number]))
        cleans[number] = clean

    margins, margin_source = booklet_margin(pages)

    page_markers = {}
    for number, _ in pages:

        pitch = geometry[number]

        raw = find_markers(cleans[number], margins.get(number), pitch)

        rows = writing_rows(cleans[number], margins.get(number))

        # Snap each mark back to the boundary that actually starts the
        # question - see cut_above. Bounded per page by the mark before
        # it, so two adjusted cuts on one page can never cross.
        adjusted = []
        floor = 0
        for mark in raw:
            adjusted.append(cut_above(rows, mark, floor, pitch))
            floor = mark["bottom"]  # next mark's search stops below THIS
                                    # mark's ink, not at its snapped cut,
                                    # whose band has just been claimed

        page_markers[number] = adjusted

    boundaries = build_boundaries(pages, page_markers, heights)
    spans = chunk_spans([n for n, _ in pages], boundaries)

    width = widths[pages[0][0]]

    strips = []
    manifest_rows = []

    for index, span in enumerate(spans):

        band_height = 0
        pieces = []

        for number, y0, y1 in span:
            top = y0 if y0 is not None else 0
            bottom = y1 if y1 is not None else heights[number]
            top = int(round(max(0, top)))
            bottom = int(round(min(heights[number], bottom)))
            if bottom <= top:
                continue
            pieces.append(images[number][top:bottom, 0:width])
            band_height += bottom - top

        pitch = geometry[span[0][0]]

        if band_height < MIN_CHUNK_PITCH * pitch and strips:
            # too small to be its own question - fold into the chunk
            # before it rather than showing a sliver
            strips[-1] = np.vstack([strips[-1], *pieces]) if pieces else strips[-1]
            if manifest_rows:
                manifest_rows[-1]["end_page"] = span[-1][0]
            continue

        if not pieces:
            continue

        chunk_img = np.vstack(pieces) if len(pieces) > 1 else pieces[0]
        strips.append(chunk_img)

        manifest_rows.append({
            "student": student, "cie": cie, "chunk": len(strips) - 1,
            "start_page": span[0][0], "end_page": span[-1][0],
            "height_px": band_height,
            "marker": index > 0,
        })

    if not strips:
        return None, [], [], "no content"

    separator = np.full((SEPARATOR_PX, width, 3), 255, dtype=np.uint8)
    separator[:] = SEPARATOR_COLOUR

    canvas_pieces = []
    for i, strip in enumerate(strips):
        colour = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
        if i:
            canvas_pieces.append(separator)
        canvas_pieces.append(colour)

    composite = np.vstack(canvas_pieces)

    n_markers = sum(1 for r in manifest_rows if r["marker"])

    return composite, strips, manifest_rows, (
        f"{len(strips)} chunk(s), {n_markers} marker(s), margin: {margin_source}")


def load_geometry(student, cie, page):

    import json

    path = GEOMETRY_DIR / "segmentation.json"

    if not hasattr(load_geometry, "_cache"):
        if not path.exists():
            load_geometry._cache = {}
        else:
            with open(path) as handle:
                records = json.load(handle)
            load_geometry._cache = {r["page_id"]: r for r in records}

    return load_geometry._cache.get(f"s{student:02d}_c{cie}_p{page:02d}")


def all_booklets():

    booklets = []

    for student_dir in sorted(SOURCE_DIR.glob("student_*")):
        student = int(re.search(r"(\d+)", student_dir.name).group(1))
        for cie_dir in sorted(student_dir.glob("cie_*")):
            cie = int(re.search(r"(\d+)", cie_dir.name).group(1))
            booklets.append((student, cie))

    return booklets


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=int)
    parser.add_argument("--cie", type=int)
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        raise SystemExit(f"source not found: {SOURCE_DIR}")

    booklets = all_booklets()

    if args.student:
        booklets = [b for b in booklets if b[0] == args.student]
    if args.cie:
        booklets = [b for b in booklets if b[1] == args.cie]

    if not booklets:
        raise SystemExit("no matching booklets")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for student, cie in booklets:

        pages = booklet_pages(student, cie)

        if not pages:
            continue

        composite, strips, rows, note = render_booklet(student, cie, pages)

        if composite is None:
            print(f"student_{student:02d}/cie_{cie}: skipped - {note}")
            continue

        target_dir = OUT_DIR / f"student_{student:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"cie_{cie}.png"
        cv2.imwrite(str(out_path), composite)

        # each chunk on its own too - the composite is for a quick
        # scroll through the whole booklet, this is for looking at one
        # question at a time without the rest of the page around it
        chunk_dir = target_dir / f"cie_{cie}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for old in chunk_dir.glob("*.png"):
            old.unlink()  # stale crops from a previous run with a
                          # different chunk count would otherwise linger

        for row, strip in zip(rows, strips):
            chunk_path = chunk_dir / f"chunk_{row['chunk']:02d}.png"
            cv2.imwrite(str(chunk_path), strip)
            row["chunk_path"] = str(chunk_path.relative_to(OUT_DIR))

        for row in rows:
            row["path"] = str(out_path.relative_to(OUT_DIR))
        all_rows.extend(rows)

        print(f"student_{student:02d}/cie_{cie}: {note} "
              f"-> {out_path.relative_to(OUT_DIR)} "
              f"({composite.shape[1]}x{composite.shape[0]})")

    if all_rows:
        with open(OUT_DIR / "chunks.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nManifest: {OUT_DIR / 'chunks.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
