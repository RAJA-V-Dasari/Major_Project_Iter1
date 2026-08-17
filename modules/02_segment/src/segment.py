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
        orphans = []

        for box in flat:

            index = int(np.argmin(np.abs(rows - box[3])))

            if abs(rows[index] - box[3]) <= tolerance:
                grouped.setdefault(index, []).append(box)
            else:
                # writing that sits on no rule at all - a superscript,
                # a label floating in a diagram. Keep it rather than
                # forcing it onto the nearest line.
                orphans.append(box)

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
