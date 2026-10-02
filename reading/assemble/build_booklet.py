"""
Turn one booklet's page transcriptions into a single Markdown document.

    02_read/output/<engine>/  (one .md per page)
    01_prepare/03_tone/output/  (the page images)
        -> 03_assemble/output/booklets/<booklet>/booklet.md
        -> 03_assemble/output/booklets/<booklet>/figures/*.png

WHY THE FIGURE POSITION COMES FROM TWO PLACES
---------------------------------------------
The reader and the geometry each know half of where a figure is, and
neither knows the other half.

The reader knows the READING ORDER. On s01_c1_p04 it put its marker
immediately after "* Eg:", which is exactly where the handshake diagram
belongs in the answer - and then gave coordinates (105,210,634,840)
pointing at prose at the top of the page. Measured over batch00, one
box in six enclosed its figure, because the numbers come back as round
values in the model's own resized space: guesses, not measurements.

`segment.py` knows the PIXELS. On the same page it puts the diagram at
(399,1669)-(1312,1843), which is right. What it cannot do is say which
region is a figure and which is prose - nothing in this corpus
separates them, since students rest drawings on the ruled lines exactly
as they rest their writing.

So markers are matched to regions IN ORDER: the nth figure the reader
mentions is the nth drawn region down the page. Order is the one thing
both agree on.

A "drawn region" is a line box far taller than the rule pitch. Ordinary
handwriting is one pitch high by construction; a drawing is not.

WHAT HAPPENS TO WHAT DOES NOT MATCH
-----------------------------------
Unmatched regions are still cropped and still appear, flagged, at the
end of the page they came from. 17 pages of batch00 flattened a diagram
into a run of labels and emitted no marker at all, so dropping
unreferenced regions would silently lose exactly the figures the reader
is worst at noticing.

A hand-drawn table therefore tends to appear twice: once as the
Markdown the reader transcribed from it, and again as a picture. That
is deliberate and was accepted rather than overlooked. Telling a table
the reader read correctly from one it quietly mangled is not something
this stage can do, and deleting the second kind loses an answer, while
keeping the first kind costs a marker one duplicated glance.

Run:
    python build_booklet.py --engine batch00_new --booklet student_01/cie_1
    python build_booklet.py --engine batch00_new --all
    python build_booklet.py --list
"""

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES = STAGE_DIR.parent
REPO = MODULES.parent

PAGES = MODULES / "01_prepare" / "03_tone" / "output"
READ = MODULES / "02_read" / "output"
OUT_DIR = STAGE_DIR / "output" / "booklets"

COVER_PAGE = 1

sys.path.insert(0, str(SRC_DIR))
import segment                                       # noqa: E402

# The question schema these booklets are set from, in the order the
# answers should appear. 3a/3b and 4a/4b are alternatives - a student
# answers one of each - so a booklet is not expected to hold them all.
SCHEMA = ["1", "2a", "2b", "2c", "3a", "3b", "4a", "4b"]

PAGE_ID = re.compile(r"^s(\d+)_c(\d+)_p(\d+)$")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

# Both marker spellings the reader actually produces, plus the two
# malformed shapes seen in batch00_new: ![fig: LAN] and
# ![](diagram: Polling Protocol). They are figures that were named
# slightly wrong, not figures that do not exist.
MARKER = re.compile(
    r"^!\[\s*\]?\(?\s*(diagram|table|fig|figure)\s*[:\]]\s*([^\])]*)[\)\]]*\s*$",
    re.I)

FENCE = re.compile(r"^```")

# What counts as a drawing.
#
# `segment.py` flags a region "grid" when it merged hand-drawn strokes
# into it - long, straight, thin marks unlike any letter. That is the
# strongest evidence available that a student drew rather than wrote,
# and it is why the merge exists: 13 of 14 annotated tables contain six
# or more such strokes against 1% of paragraphs.
#
# A region with no strokes in it but far taller than a line of writing
# is the other case - a free-hand sketch with no straight edges. Two
# and a half pitches, because 1.6 (segment.py's own "not a text line")
# also catches a two-line paragraph whose descenders merged.
TALL_PITCH = 2.5

# Below this a region is a stray mark - a margin tick, a scanner edge -
# and cropping it yields a picture of nothing.
#
# The floor was 90000/90 and that lost exactly the diagrams the reader
# also fails on. A graph of circled nodes joined by thin edges is a big
# drawing made of small strokes: on s07_c2_p09 segment.py found the two
# stroke clusters and both fell under the old floor, so a page whose
# transcription had already broken down produced no figure either.
MIN_AREA = 24000
MIN_SIDE = 45

# A line repeated this many times running is the reader looping rather
# than the student writing. Kept low because the run is contiguous:
# nothing legitimate repeats one identical line six times in a row.
LOOP_RUN = 6

# The repeating unit can be a short block rather than one line, so
# periods up to this are checked. Four covers a fenced value - fence,
# value, fence, blank - which is the longest seen here.
LOOP_PERIOD = 4

# Characters of transcription per unit of ink, above which the reader
# was generating rather than reading. The corpus median is ~15,000 and
# audit_pages.py calls 1.9x median "dense"; this sits at 2.7x so only a
# page that cannot possibly hold its own transcription trips it.
CONFABULATION_YIELD = 40000

# Vertical gap, in rule pitches, below which two drawn boxes are one
# drawing. The rules cut a figure into bands, so this reassembles it.
MERGE_PITCH = 1.6

# How far past its own strokes a drawing may reach, in rule pitches.
# Enough for the rows of labels above and below a figure; not enough to
# swallow the paragraph before it. segment.py bounds its own grid
# growth the same way and for the same reason.
GROW_PITCH = 4.0

# --- the diagram pass ------------------------------------------------
#
# When `--diagrams` is given, the drawings enumerated by the dedicated
# Qwen pass replace the reading pass's markers as the figure source.
# The reading pass marks 52 of 1,000 pages, because finding figures was
# a side clause in a prompt whose real job was transcription; the
# dedicated pass fires on 24% and catches sparse drawings the geometry
# rule scores 0.00 on.

# How close an anchor must match a line of the transcription to place a
# drawing there. The anchor is words the model READ, and it reads at
# CER 0.099, so a real match scores far above this - 0.55 is set to
# admit a line the reader spelt differently, not to guess.
ANCHOR_MIN_RATIO = 0.55

# An anchor shorter than this is not evidence. "the" matches everywhere.
ANCHOR_MIN_CHARS = 10

# Padding around a span when geometry has no region for it, as a
# fraction of page height. The model's spans came back at roughly half
# their true height and anchored at the top, so this leans generous on
# purpose: a crop carrying a spare line of writing is fine in the
# finished Markdown, a crop that clips an arrow is not.
SPAN_PAD_FRAC = 0.06

# A geometry region counts as the same drawing as a reported span when
# they overlap by this fraction of the shorter of the two. Deliberately
# loose - the question is "same thing?", not "same edges?", and the
# model's edges are known to be poor.
REGION_OVERLAP = 0.25

# Breathing room left around the ink when a crop is trimmed to width.
INK_MARGIN = 18


def page_id_of(path):
    m = re.search(r"student_(\d+)[\\/]cie_(\d+)[\\/]page_(\d+)", str(path))
    s, c, p = (int(g) for g in m.groups())
    return f"s{s:02d}_c{c}_p{p:02d}"


# A heading is a label, optionally followed by the student's own title:
# "2a)", "2 a )", "2a) HTTP request". Only the label is matched, because
# requiring the whole heading to be one filed "### 2a) HTTP request" as
# untitled prose and lost the question it announced.
# Students bracket the sub-part and prefix the number, and the reader
# copies them faithfully - as it should. Over the 1,000 pages read, 79
# headings carried a real question label in a form a plain "2a)" match
# rejects: "2(b)", "Q3)", "Q(2a)", "Q. 4(a).", "4(b) - C.R.C.". Three
# booklets came out with no question at all because of it.
LABEL = re.compile(
    r"^\s*(?:Q(?:ues)?\.?\s*)?"        # an optional "Q", "Q." or "Ques"
    r"\(?\s*([1-9])\s*[).]?\s*"        # the number, possibly bracketed
    r"[(\[]?\s*([a-d])?\s*[)\]]?"      # the sub-part, possibly bracketed
    r"\s*[).:-]*\s*(.*)$",             # trailing punctuation, then a title
    re.I)


def split_label(text, seen=()):
    """('2a', 'HTTP request') for a schema heading, else (None, text).

    `seen` is the labels already assigned in this booklet, which is what
    decides a bare "1".
    """
    m = LABEL.match(text)
    if not m:
        return None, text
    number, letter, rest = m.group(1), (m.group(2) or "").lower(), m.group(3)

    # "2a) HTTP request" is question 2a. "2 hosts can act as manager" is
    # a sentence that happens to start with a digit, so a label is only
    # accepted when the text after it is a title, not a paragraph.
    if len(rest.split()) > 8:
        return None, text

    label = f"{number}{letter}"
    if label not in SCHEMA:
        return None, text

    # A bare "1" is the one label with no letter to confirm it, so it is
    # also the one an enumerated list forges. student_01/cie_3 put page
    # 09 under question 1 on the strength of a "1" written after 2a, 2b
    # and 2c had already been answered - a list item, not a question.
    # Booklets run forwards, so a "1" arriving after a later question is
    # not question 1.
    if label == "1" and seen:
        latest = max(SCHEMA.index(q) for q in seen)
        if latest > SCHEMA.index("1"):
            return None, text

    return label, rest.strip()


def clean_page(body):
    """Strip reader scaffolding.

    A whole-page ```markdown wrapper is scaffolding. A fence around PART
    of a page is the student's own block - s07_c2_p08 keeps a Dijkstra
    working table that way - so a balanced pair around the whole page is
    removed and an inner pair is kept.

    A page cut off at the token ceiling has an OPENING fence and no
    closing one, because the closing fence was never reached. That is
    not the student's block either, so an unmatched opening fence goes
    too - otherwise it fences the rest of the booklet.
    """
    lines = body.strip().splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    fences = sum(1 for l in lines if FENCE.match(l.strip()))

    if (len(lines) >= 2 and fences == 2
            and FENCE.match(lines[0].strip())
            and FENCE.match(lines[-1].strip())):
        lines = lines[1:-1]
    elif fences % 2 and lines and FENCE.match(lines[0].strip()):
        lines = lines[1:]

    return lines


def loop_start(lines):
    """Index where the reader stopped reading and began repeating.

    Everything before it is a real transcription and gradeable; the
    Dijkstra table on s10_c2_p10 is correct right up to the point the
    page runs out of text. Everything after is invented, and pasting it
    into a booklet puts 636 empty rows in front of a marker with nothing
    to say which half to believe.

    The repeating unit is not always one line. On s43_c3_p06 it is a
    three-line block - a fence, a row of zeroes, a fence - so nothing
    repeats on consecutive lines and a single-line test sees nothing
    while 174 fences pile into the booklet. So periods up to a few
    lines are checked, not just period one.

    Returns None if the page is fine.
    """
    # A row of empty cells is the commonest thing a loop repeats, so it
    # needs a key of its own - collapsing it to "" makes it look like an
    # ordinary blank line, and blank lines are everywhere and must not
    # count as evidence.
    keys = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            keys.append("")
            continue
        bare = re.sub(r"[\s|*`-]+", "", stripped)
        keys.append(bare or "\x00empty-row")

    best = None
    for period in range(1, LOOP_PERIOD + 1):
        i = 0
        while i + period * LOOP_RUN <= len(keys):
            unit = keys[i:i + period]
            if any(unit):                                 # not all blank
                repeats = 1
                j = i + period
                while (j + period <= len(keys)
                       and keys[j:j + period] == unit):
                    repeats += 1
                    j += period
                if repeats >= LOOP_RUN:
                    best = i if best is None else min(best, i)
                    break
                i = j if repeats > 1 else i + 1
            else:
                i += 1

    return best


def figure_regions(image_path):
    """(regions, ink, clean) - drawn boxes, page ink, rule-free mask.

    The ink fraction comes free with the segmentation and is what
    catches the one loop shape no repetition test can see: a page that
    invents a NOVEL sequence. s07_c2_p09 produced 36 numbered sections
    walking the alphabet, every line different, so nothing repeats -
    but the page carries 0.8% ink and cannot support 1,800 characters
    whatever they say.

    `clean` is the ink mask with the printed rules taken out, and it
    comes free the same way. It is what `ink_columns` trims a crop
    against: the raw mask cannot be used, because a printed rule spans
    the full page width and would make every band look full-width.
    """
    record, gray, rules = segment.segment_page(image_path)
    if record is None:
        return [], 0.0, None

    clean = segment.ink_mask(gray)
    if rules is not None:
        clean[rules > 0] = 0

    pitch = record["rule_pitch"] or 1

    every = [l for b in record["blocks"] for l in b["lines"]]
    every.sort(key=lambda l: l["bbox"][1])

    def is_drawn(line):
        x1, y1, x2, y2 = line["bbox"]
        return bool(line.get("grid")) or (y2 - y1) >= TALL_PITCH * pitch

    # A diagram's labels - "client", "Server", "First handshake" - are
    # ordinary text-height boxes, so the strokes alone crop to the middle
    # of the figure with its ends sliced off.
    #
    # Width does not separate a label from prose. On s01_c1_p04 the row
    # under the handshakes is 854px against a 1036px text column, but the
    # row under THAT is the full 1036 and is still part of the figure - a
    # width rule keeps one and drops the other, which is how the
    # termination half of that diagram went missing.
    #
    # So growth is bounded by DISTANCE instead, the way segment.py bounds
    # the same thing: the strokes say where the drawing is, it may reach
    # a few lines past them, and it may not run away up the page.
    # Over-reaching puts a caption in the crop; under-reaching loses half
    # the diagram. The caption is the cheaper mistake.
    found = []
    for i, line in enumerate(every):
        if not is_drawn(line):
            continue
        x1, y1, x2, y2 = line["bbox"]
        if (x2 - x1) < MIN_SIDE or (y2 - y1) < MIN_SIDE:
            continue
        if (x2 - x1) * (y2 - y1) < MIN_AREA:
            continue

        top, bottom = y1, y2                    # the strokes' own extent
        for step in (-1, 1):                    # grow up, then down
            j = i + step
            while 0 <= j < len(every):
                bx1, by1, bx2, by2 = every[j]["bbox"]
                gap = by1 - y2 if step > 0 else y1 - by2
                if gap > MERGE_PITCH * pitch:
                    break
                if min(x2, bx2) <= max(x1, bx1):
                    break                       # not above or below it
                reach = (by2 - bottom) if step > 0 else (top - by1)
                if reach > GROW_PITCH * pitch:
                    break                       # far enough from the strokes
                x1, y1 = min(x1, bx1), min(y1, by1)
                x2, y2 = max(x2, bx2), max(y2, by2)
                j += step

        found.append((x1, y1, x2, y2))

    ink = record["ink_pixels"] / float(record["size"][0] * record["size"][1])

    found.sort(key=lambda b: b[1])

    # One drawing arrives as several boxes. segment.py splits on the
    # printed rules, so the three-way handshake on s01_c1_p04 comes back
    # as a middle band without the client and server boxes above it or
    # the termination below - a crop that is the right diagram with its
    # ends cut off. Boxes closer than a rule pitch are the same drawing.
    merged = []
    for box in found:
        if merged:
            px1, py1, px2, py2 = merged[-1]
            x1, y1, x2, y2 = box
            overlaps = min(px2, x2) > max(px1, x1)
            if overlaps and y1 - py2 <= MERGE_PITCH * pitch:
                merged[-1] = (min(px1, x1), min(py1, y1),
                              max(px2, x2), max(py2, y2))
                continue
        merged.append(box)

    return merged, ink, clean


def ink_columns(clean, y1, y2, margin=INK_MARGIN):
    """(x1, x2) - the horizontal extent of the ink in one band.

    A band taken from a fraction of page height runs the full width, so
    a crop made from one is a letterboxed strip with the drawing adrift
    in the middle of it. This is the fix, and it closes the open finding
    in 04_evaluate/RESULTS.md that figure boxes are bands, not crops.

    `clean` must be the RULE-FREE mask from figure_regions. Run it
    against the raw ink and every band comes back full-width, because
    that is exactly how wide a printed rule is.
    """
    if clean is None:
        return None
    band = clean[max(0, y1):max(0, y2)]
    if band.size == 0:
        return None
    col = (band > 0).sum(axis=0)
    # A couple of pixels in a column is speckle left by tone
    # flattening, not a stroke.
    idx = np.flatnonzero(col > 2)
    if idx.size == 0:
        return None
    return (max(0, int(idx[0]) - margin),
            min(clean.shape[1], int(idx[-1]) + margin + 1))


WORDS = re.compile(r"[a-z0-9]+")


def normalise(text):
    """Lowercase words only - markdown, punctuation and spacing gone."""
    return " ".join(WORDS.findall(text.lower()))


def anchor_index(lines, anchor):
    """Index of the line the anchor names, or None.

    The anchor is the last few words of handwriting above a drawing, so
    it matches the TAIL of a line rather than the whole of it. Both are
    tried and the better score wins, which is what lets a six-word
    anchor find its place in a forty-word paragraph.
    """
    want = normalise(anchor)
    if len(want) < ANCHOR_MIN_CHARS:
        return None

    best, best_i = 0.0, None
    for i, line in enumerate(lines):
        have = normalise(line)
        if not have:
            continue
        score = difflib.SequenceMatcher(None, want, have).ratio()
        if len(have) > len(want):
            tail = have[-len(want):]
            score = max(score,
                        difflib.SequenceMatcher(None, want, tail).ratio())
        if score > best:
            best, best_i = score, i

    return best_i if best >= ANCHOR_MIN_RATIO else None


def place_drawings(lines, drawings, marker_at):
    """Where each drawing goes. Returns (replace_at, after, how).

    `replace_at` maps a line index to the drawing that stands in its
    place - used only when the reading pass happened to emit exactly as
    many markers as the pass found drawings, because then its positions
    are true reading order and better than any match.

    `after` maps a line index to the drawings emitted below that line,
    found by matching each drawing's anchor text. -1 means the top of
    the page. Proportional placement is the fallback for a drawing whose
    anchor matched nothing: crude, but handwriting runs top to bottom,
    so it lands in roughly the right part of the answer.

    `how` records which of the three placed each drawing, so the run can
    report how much of the result rests on the fallback.
    """
    if not drawings:
        return {}, {}, {}

    if marker_at and len(marker_at) == len(drawings):
        return (dict(zip(marker_at, drawings)), {},
                {id(d): "marker" for d in drawings})

    after, how = {}, {}
    n = len(lines)
    for d in drawings:
        i = anchor_index(lines, d.get("anchor", ""))
        how[id(d)] = "anchor" if i is not None else "position"
        if i is None:
            i = int(round(float(d.get("from", 0.0)) * n)) - 1
        after.setdefault(max(-1, min(n - 1, i)), []).append(d)
    return {}, after, how


def drawing_box(drawing, regions, clean, shape, claimed=()):
    """The pixels to crop for one reported drawing.

    Geometry first: where a region overlaps the reported span it has the
    better edges and is already trusted by the rest of this file. Where
    none does - the sparse node-and-edge case geometry is blind to, which
    is the whole reason this pass exists - the span itself is used,
    padded generously and trimmed to the width of its own ink.

    A region already `claimed` by an earlier drawing is skipped. Two
    drawings stacked on one page often both overlap the same tall
    region, and without this they are handed the same box and the
    booklet shows the same picture twice under two captions.
    """
    h, w = shape
    y1 = int(max(0.0, float(drawing.get("from", 0.0))) * h)
    y2 = int(min(1.0, float(drawing.get("to", 0.0))) * h)
    if y2 <= y1:
        y1, y2 = max(0, y1 - h // 20), min(h, y1 + h // 20)

    for region in regions:
        if region in claimed:
            continue
        rx1, ry1, rx2, ry2 = region
        shorter = max(1, min(y2 - y1, ry2 - ry1))
        if (min(y2, ry2) - max(y1, ry1)) / shorter >= REGION_OVERLAP:
            return (rx1, ry1, rx2, ry2), True

    pad = int(SPAN_PAD_FRAC * h)
    y1, y2 = max(0, y1 - pad), min(h, y2 + pad)
    span = ink_columns(clean, y1, y2)
    x1, x2 = span if span else (0, w)
    return (x1, y1, x2, y2), False


def load_diagrams(path):
    """{pid: [drawing, ...]} from the diagram pass, sorted down the page."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for pid, items in raw.items():
        if not items:
            continue
        out[pid] = sorted((d for d in items if isinstance(d, dict)),
                          key=lambda d: float(d.get("from", 0.0)))
    return out


def load_rejects(path):
    """{pid: {index, ...}} of drawings a human threw out in review."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {pid: set(int(i) for i in idx) for pid, idx in raw.items()}


def crop(image_path, box, target, pad=12):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(target), img[y1:y2, x1:x2]))


def build(engine, booklet, out_root=OUT_DIR, quiet=False,
          diagrams=None, rejects=None):
    """Assemble one booklet. Returns a stats dict."""
    student, cie = booklet.split("/")
    page_dir = PAGES / student / cie
    read_dir = READ / engine
    diagrams = diagrams or {}
    rejects = rejects or {}

    images = sorted(p for p in page_dir.glob("page_*.png")
                    if int(re.search(r"(\d+)", p.stem).group(1)) != COVER_PAGE)
    if not images:
        raise SystemExit(f"no content pages under {page_dir}")

    name = f"{student}_{cie}"
    dest = out_root / name
    figures = dest / "figures"

    # Crops are numbered per booklet, so a run that emits fewer figures
    # than the last one leaves orphans behind - and a rebuild that
    # renumbers leaves a crop whose name no longer means what it did.
    # The folder is derived output; clear it and let this run own it.
    for old in figures.glob("*.png"):
        old.unlink()

    stats = {"pages": 0, "missing": 0, "markers": 0, "matched": 0,
             "unreferenced": 0, "figures": 0, "recovered": 0,
             "truncated": 0, "confabulated": 0,
             "drawings": 0, "by_anchor": 0, "by_marker": 0,
             "by_position": 0, "from_geometry": 0, "from_span": 0,
             "rejected": 0}

    # question label -> list of markdown lines, in page order
    answers = {}
    order = []
    current = None
    stray = []
    emitted_figures = []

    for image in images:
        pid = page_id_of(image)
        md = read_dir / f"{pid}.md"
        if not md.exists():
            stats["missing"] += 1
            continue
        stats["pages"] += 1

        lines = clean_page(md.read_text(encoding="utf-8"))

        # Cut a runaway page at the point it stopped reading. What comes
        # before is a real transcription and is kept; what comes after is
        # the model completing its own pattern, and the whole page image
        # goes in below so a marker can read what was lost.
        cut = loop_start(lines)
        if cut is not None:
            lines = lines[:cut]
            stats["truncated"] += 1

        regions, ink, clean = figure_regions(image)

        # More text than the page can physically support: the reader was
        # generating, not reading. Nothing repeats on such a page, so
        # there is no point to cut at - the whole transcription is
        # suspect and the image is the only trustworthy record.
        body_chars = sum(len(l) for l in lines)
        if ink > 0 and body_chars / ink > CONFABULATION_YIELD:
            lines, cut = [], 0
            stats["confabulated"] += 1

        marker_at = [i for i, l in enumerate(lines) if MARKER.match(l.strip())]
        stats["markers"] += len(marker_at)

        on_page = diagrams.get(pid, [])
        index_of = {id(d): i for i, d in enumerate(on_page)}
        drawn = [d for i, d in enumerate(on_page)
                 if i not in rejects.get(pid, ())]
        stats["rejected"] += len(on_page) - len(drawn)

        if diagrams:
            # The diagram pass is the authority on what is a drawing.
            # Its spans choose the boxes and its anchors choose the
            # places; geometry is consulted only for better edges.
            stats["drawings"] += len(drawn)
            replace_at, after, how = place_drawings(lines, drawn, marker_at)
            for kind in how.values():
                stats[f"by_{kind}"] += 1
            if clean is not None:
                shape = clean.shape
            else:                       # segmentation failed on this page
                probe = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
                shape = probe.shape if probe is not None else (1, 1)
            boxes, claimed = {}, set()
            for d in drawn:
                box, from_geo = drawing_box(d, regions, clean, shape, claimed)
                if from_geo:
                    claimed.add(box)
                boxes[id(d)] = box
                stats["from_geometry" if from_geo else "from_span"] += 1
            pairs, used = {}, set()
        else:
            # nth marker <-> nth region down the page
            pairs = dict(zip(marker_at, regions))
            stats["matched"] += len(pairs)
            used = set(pairs.values())
            replace_at, after, boxes = {}, {}, {}
            how, claimed = {}, set()

        def emit_drawing(d, into):
            """Write one crop and append its Markdown reference."""
            n = stats["figures"] + 1
            rel = f"figures/{pid}_f{n:02d}.png"
            if not crop(image, boxes[id(d)], dest / rel):
                return
            stats["figures"] = n
            stats["matched"] += 1
            caption = (d.get("caption") or d.get("kind") or "figure").strip()
            into.append("")
            into.append(f"![{caption}]({rel})")
            into.append("")
            # Provenance for review_figures.py: which drawing on which
            # page this crop came from, so a reviewer's reject can be
            # written back against the diagram pass's own index.
            emitted_figures.append({
                "file": rel, "pid": pid, "index": index_of[id(d)],
                "caption": caption, "kind": d.get("kind", ""),
                "placed": how.get(id(d), "marker"),
                "box": "geometry" if boxes[id(d)] in claimed else "span",
            })

        emitted = []
        for d in after.get(-1, []):
            emit_drawing(d, emitted)
        for i, line in enumerate(lines):
            stripped = line.strip()

            head = HEADING.match(stripped)
            if head:
                label, title = split_label(head.group(2), order)
                if label:
                    current = label
                    if label not in answers:
                        answers[label] = []
                        order.append(label)
                    if title:
                        emitted.append(f"**{title}**")
                    continue
                # Not a schema label. A numbered point inside an answer
                # became a heading on 2 pages of batch00 and would file
                # one answer under several invented questions, so the
                # text is kept and the heading is not.
                depth = len(head.group(1))
                emitted.append(f"**{head.group(2)}**" if depth >= 4
                               else f"*{head.group(2)}*")
                continue

            if i in replace_at:
                emit_drawing(replace_at[i], emitted)
                continue

            if i in pairs:
                box = pairs[i]
                n = stats["figures"] + 1
                rel = f"figures/{pid}_f{n:02d}.png"
                if crop(image, box, dest / rel):
                    stats["figures"] = n
                    kind, caption = MARKER.match(stripped).groups()
                    caption = caption.strip() or kind.lower()
                    emitted.append(f"![{caption}]({rel})")
                else:
                    emitted.append(stripped)
                continue

            if MARKER.match(stripped):
                # A marker the diagram pass did not corroborate. Dropped
                # rather than kept: with the pass as the authority, a
                # lone marker is a disagreement, and printing it as a
                # note puts scaffolding in front of the marker on a page
                # where nothing is missing.
                if not diagrams:
                    emitted.append(f"> _figure not located: {stripped}_")
                continue

            emitted.append(line)

            for d in after.get(i, []):
                emit_drawing(d, emitted)

        # Regions the reader never mentioned. 17 pages of batch00
        # flattened a diagram into labels and emitted no marker, so
        # these were shown rather than dropped.
        #
        # Only without --diagrams. The geometry rule fires on 37% of
        # pages and cannot tell a ruled margin from a ruled table, so
        # once a model is saying what is and is not a drawing, appending
        # every unclaimed region is how false figures got into the
        # booklets - and appending them HERE, after the answer, is how
        # the real ones ended up in the wrong place.
        for box in ([] if diagrams else regions):
            if box in used:
                continue
            n = stats["figures"] + 1
            rel = f"figures/{pid}_f{n:02d}.png"
            if crop(image, box, dest / rel):
                stats["figures"] = n
                stats["unreferenced"] += 1
                emitted.append("")
                emitted.append(f"![unreferenced figure]({rel})")

        # The whole page, for a page whose reading broke down. The
        # transcription above it stops where the reading did, so this is
        # the only record of the rest - and on these pages the geometry
        # usually misses the figures too, because a graph of circles and
        # thin edges is sparse enough to fall under the size floor.
        if cut is not None:
            rel = f"figures/{pid}_page.png"
            if crop(image, (0, 0, 10**6, 10**6), dest / rel, pad=0):
                emitted.append("")
                emitted.append(f"> **Reading failed partway down this "
                               f"page.** Everything above is transcribed "
                               f"from it; the rest was not readable. The "
                               f"full page is below - mark from the "
                               f"image, not from the text.")
                emitted.append("")
                emitted.append(f"![{pid} full page]({rel})")

        body = "\n".join(emitted).strip()
        if not body:
            continue

        if current is None:
            stray.extend([f"<!-- {pid} -->", body, ""])
        else:
            answers[current].extend([f"<!-- {pid} -->", body, ""])

    # Schema order first, then anything else the reader claimed.
    ordered = [q for q in SCHEMA if q in answers]
    ordered += [q for q in order if q not in SCHEMA]

    # Pages written before the first question number appears. A student
    # starts under "PART-A" without repeating the number, so the opening
    # answer carries no label - 16 of 250 pages, nearly all of them a
    # booklet's first page.
    #
    # If question 1 was never labelled anywhere, this is question 1: it
    # comes before whatever the first labelled question is, and the paper
    # begins at 1. If 1 IS labelled elsewhere, the owner is genuinely
    # unknown and guessing would file an answer under the wrong question,
    # so it stays visible and unattached instead.
    unplaced = []
    if stray:
        if ordered and "1" not in answers:
            answers["1"] = stray + answers.get("1", [])
            ordered.insert(0, "1")
            stats["recovered"] = 1
        else:
            unplaced = stray

    doc = [f"# {student.replace('_', ' ')} — {cie.replace('_', ' ')}", ""]

    if unplaced:
        doc += ["## Before the first question number", ""] + unplaced

    for q in ordered:
        doc += [f"## {q})", ""] + answers[q]

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "booklet.md").write_text("\n".join(doc).rstrip() + "\n",
                                     encoding="utf-8")

    if emitted_figures:
        (dest / "figures.json").write_text(
            json.dumps(emitted_figures, indent=1), encoding="utf-8")

    stats["questions"] = len(ordered)
    stats["found"] = ordered
    stats["booklet"] = name

    if not quiet:
        print(f"{name}: {stats['pages']} pages, {stats['questions']} questions "
              f"{ordered}, {stats['figures']} figures "
              f"({stats['matched']} matched, {stats['unreferenced']} "
              f"unreferenced), {stats['truncated']} pages cut at a loop, "
              f"{stats['missing']} unread")
        print(f"  {dest / 'booklet.md'}")

    return stats


def booklets():
    found = []
    for d in sorted(PAGES.glob("student_*/cie_*")):
        if any(d.glob("page_*.png")):
            found.append(f"{d.parent.name}/{d.name}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default="batch00_new")
    ap.add_argument("--booklet")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--diagrams", help="diagrams.json from the diagram pass; "
                                       "becomes the figure source")
    ap.add_argument("--rejects", help="rejects.json from review_figures.py")
    args = ap.parse_args()

    if args.list:
        for b in booklets():
            print(" ", b)
        return 0

    diagrams = load_diagrams(args.diagrams) if args.diagrams else None
    rejects = load_rejects(args.rejects) if args.rejects else None
    if diagrams:
        n = sum(len(v) for v in diagrams.values())
        cut = sum(len(v) for v in (rejects or {}).values())
        print(f"{n} drawings on {len(diagrams)} pages"
              + (f", {cut} rejected in review" if cut else ""))

    if args.all:
        rows = []
        for b in booklets():
            read_dir = READ / args.engine
            student, cie = b.split("/")
            s = int(re.search(r"(\d+)", student).group(1))
            c = int(re.search(r"(\d+)", cie).group(1))
            if not list(read_dir.glob(f"s{s:02d}_c{c}_p*.md")):
                continue                      # not read yet
            rows.append(build(args.engine, b, diagrams=diagrams,
                              rejects=rejects))

        if not rows:
            raise SystemExit(f"no transcriptions under {READ / args.engine}")

        report = STAGE_DIR / "output" / "booklets.csv"
        with open(report, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["booklet", "pages", "questions", "figures",
                        "matched", "unreferenced", "truncated", "missing",
                        "found"])
            for r in rows:
                w.writerow([r["booklet"], r["pages"], r["questions"],
                            r["figures"], r["matched"], r["unreferenced"],
                            r["truncated"], r["missing"],
                            " ".join(r["found"])])

        def total(k):
            return sum(r[k] for r in rows)

        print(f"\n{len(rows)} booklets, {total('pages')} pages, "
              f"{total('figures')} figures "
              f"({total('matched')} placed, "
              f"{total('unreferenced')} unreferenced), "
              f"{total('truncated')} pages cut at a loop")
        if diagrams:
            print(f"  placed by  anchor {total('by_anchor')}  "
                  f"marker {total('by_marker')}  "
                  f"position {total('by_position')}")
            print(f"  box from   geometry {total('from_geometry')}  "
                  f"span {total('from_span')}")
            lost = total('drawings') - total('matched')
            if lost:
                print(f"  {lost} drawings could not be cropped")
        print(f"  {report}")
        return 0

    if not args.booklet:
        raise SystemExit("give --booklet student_01/cie_1, or --all")

    build(args.engine, args.booklet, diagrams=diagrams, rejects=rejects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
