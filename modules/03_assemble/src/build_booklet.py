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
import re
import sys
from pathlib import Path

import cv2

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
MIN_AREA = 90000
MIN_SIDE = 90

# Vertical gap, in rule pitches, below which two drawn boxes are one
# drawing. The rules cut a figure into bands, so this reassembles it.
MERGE_PITCH = 1.6

# How far past its own strokes a drawing may reach, in rule pitches.
# Enough for the rows of labels above and below a figure; not enough to
# swallow the paragraph before it. segment.py bounds its own grid
# growth the same way and for the same reason.
GROW_PITCH = 4.0


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
    """Strip reader scaffolding; return (lines, list of marker indices).

    A whole-page ```markdown wrapper is scaffolding. A fence around PART
    of a page is the student's own block - s07_c2_p08 keeps a Dijkstra
    working table that way - so only a fence that opens the first line
    and closes the last is removed.
    """
    lines = body.strip().splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if (len(lines) >= 2 and FENCE.match(lines[0].strip())
            and FENCE.match(lines[-1].strip())
            and sum(1 for l in lines if FENCE.match(l.strip())) == 2):
        lines = lines[1:-1]

    return lines


def figure_regions(image_path):
    """Drawn regions on a page, top to bottom, as (x1, y1, x2, y2)."""
    record, _, _ = segment.segment_page(image_path)
    if record is None:
        return []

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

    return merged


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


def build(engine, booklet, out_root=OUT_DIR, quiet=False):
    """Assemble one booklet. Returns a stats dict."""
    student, cie = booklet.split("/")
    page_dir = PAGES / student / cie
    read_dir = READ / engine

    images = sorted(p for p in page_dir.glob("page_*.png")
                    if int(re.search(r"(\d+)", p.stem).group(1)) != COVER_PAGE)
    if not images:
        raise SystemExit(f"no content pages under {page_dir}")

    name = f"{student}_{cie}"
    dest = out_root / name
    figures = dest / "figures"

    stats = {"pages": 0, "missing": 0, "markers": 0, "matched": 0,
             "unreferenced": 0, "figures": 0, "recovered": 0}

    # question label -> list of markdown lines, in page order
    answers = {}
    order = []
    current = None
    stray = []

    for image in images:
        pid = page_id_of(image)
        md = read_dir / f"{pid}.md"
        if not md.exists():
            stats["missing"] += 1
            continue
        stats["pages"] += 1

        lines = clean_page(md.read_text(encoding="utf-8"))
        regions = figure_regions(image)

        marker_at = [i for i, l in enumerate(lines) if MARKER.match(l.strip())]
        stats["markers"] += len(marker_at)

        # nth marker <-> nth region down the page
        pairs = dict(zip(marker_at, regions))
        stats["matched"] += len(pairs)
        used = set(pairs.values())

        emitted = []
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
                # a marker with no region to point at
                emitted.append(f"> _figure not located: {stripped}_")
                continue

            emitted.append(line)

        # Regions the reader never mentioned. 17 pages of batch00
        # flattened a diagram into labels and emitted no marker, so
        # these are shown rather than dropped.
        for box in regions:
            if box in used:
                continue
            n = stats["figures"] + 1
            rel = f"figures/{pid}_f{n:02d}.png"
            if crop(image, box, dest / rel):
                stats["figures"] = n
                stats["unreferenced"] += 1
                emitted.append("")
                emitted.append(f"![unreferenced figure]({rel})")

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

    stats["questions"] = len(ordered)
    stats["found"] = ordered
    stats["booklet"] = name

    if not quiet:
        print(f"{name}: {stats['pages']} pages, {stats['questions']} questions "
              f"{ordered}, {stats['figures']} figures "
              f"({stats['matched']} matched, {stats['unreferenced']} "
              f"unreferenced), {stats['missing']} pages unread")
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
    args = ap.parse_args()

    if args.list:
        for b in booklets():
            print(" ", b)
        return 0

    if args.all:
        rows = []
        for b in booklets():
            read_dir = READ / args.engine
            student, cie = b.split("/")
            s = int(re.search(r"(\d+)", student).group(1))
            c = int(re.search(r"(\d+)", cie).group(1))
            if not list(read_dir.glob(f"s{s:02d}_c{c}_p*.md")):
                continue                      # not read yet
            rows.append(build(args.engine, b))

        if not rows:
            raise SystemExit(f"no transcriptions under {READ / args.engine}")

        report = STAGE_DIR / "output" / "booklets.csv"
        with open(report, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["booklet", "pages", "questions", "figures",
                        "matched", "unreferenced", "missing", "found"])
            for r in rows:
                w.writerow([r["booklet"], r["pages"], r["questions"],
                            r["figures"], r["matched"], r["unreferenced"],
                            r["missing"], " ".join(r["found"])])

        print(f"\n{len(rows)} booklets, "
              f"{sum(r['pages'] for r in rows)} pages, "
              f"{sum(r['figures'] for r in rows)} figures "
              f"({sum(r['matched'] for r in rows)} matched to a marker, "
              f"{sum(r['unreferenced'] for r in rows)} unreferenced)")
        print(f"  {report}")
        return 0

    if not args.booklet:
        raise SystemExit("give --booklet student_01/cie_1, or --all")

    build(args.engine, args.booklet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
