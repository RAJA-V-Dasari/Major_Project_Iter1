"""
Produce SYNTHETIC OCR results in the real output shape.

    04_ocr/input/  (the line crops + manifest)
        -> 04_ocr/output/  ocr.json, lines.csv, transcripts/

WHY SIMULATE AT ALL
-------------------
Choosing and tuning a handwriting recogniser is slow, and everything
downstream - reading order, answer extraction, question association,
scoring, storage - only needs the SHAPE of the output, not its truth.
Emitting the real schema filled with plausible text unblocks all of
that immediately and in parallel, and when a real engine lands it
writes the same records, so nothing downstream has to change.

THIS DATA IS FAKE AND SAYS SO
-----------------------------
Every file carries `simulated: true`, every transcript opens with a
banner, and schema.validate_run() refuses a payload that lacks the
flag. This is not ceremony: a synthetic transcript is indistinguishable
from a real one by eye, and the failure mode - somebody quoting an
accuracy figure, or a student's "answer", that came from a random
number generator - is bad enough to be worth the noise. Nothing here
is a measurement of anything.

WHAT IS MODELLED, AND WHAT IS NOT
---------------------------------
Modelled, because downstream code has to cope with it:
  - text length scaling with crop width (~25px per character, measured
    off real lines), so line lengths are distributed like real ones
  - a confidence distribution with a bad tail, not a constant 0.99
  - the four statuses, including `failed` and `empty`, so error paths
    are exercised rather than discovered later
  - diagram regions returning no text at all

Not modelled, because faking it would mislead rather than help:
  - realistic OCR *error* patterns. The text is clean domain prose. Do
    not use this to measure accuracy, tune a spell-corrector, or judge
    whether post-processing helps - it cannot answer those.

Deterministic: each line's content is seeded from its own line_uid, so
a rerun reproduces the corpus exactly and two machines agree.

Run:
    python simulate.py               # whole corpus
    python simulate.py --limit 50    # a sample
"""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import date
from pathlib import Path

import schema
from schema import LineResult, PageResult


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

CROP_DIR = STAGE_DIR / "input"
MANIFEST_PATH = CROP_DIR / "manifest.csv"

OUT_DIR = STAGE_DIR / "output"
TRANSCRIPT_DIR = OUT_DIR / "transcripts"

ENGINE = "simulated-v1"

# Measured off real crops: a line ~1000px wide holds ~40 characters.
PIXELS_PER_CHAR = 25

# Rates for the statuses that are not `ok`. Chosen to be small but not
# negligible, so downstream error handling is actually exercised by a
# full run rather than never firing.
EMPTY_RATE = 0.02
FAILED_RATE = 0.003

# The corpus is one Computer Networks paper, so the vocabulary is
# drawn from it - real pages mention exactly these. Domain-plausible
# text keeps downstream heuristics (question detection, keyword
# matching) from being tuned against nonsense.
SUBJECTS = [
    "the sender", "the receiver", "the client", "the server",
    "the transport layer", "the application layer", "the data link layer",
    "the DNS server", "the DNS client", "the router", "the host",
]

VERBS = [
    "sends", "receives", "retransmits", "acknowledges", "forwards",
    "encapsulates", "discards", "requests", "resolves", "establishes",
]

OBJECTS = [
    "the packet", "the datagram", "the acknowledgement", "the segment",
    "the sequence number", "the window size", "the header",
    "the IP address", "the URL", "the frame", "the checksum",
]

TAILS = [
    "before the timer expires", "using Go-Back-N protocol",
    "in the sliding window", "after the three-way handshake",
    "if no error is detected", "to avoid congestion",
    "when the timeout occurs", "using selective repeat",
    "so that the data is not lost", "at the receiver end",
]

HEADINGS = [
    "PART-A", "PART-B", "PART-C", "Advantages:", "Disadvantages:",
    "Answer:", "Given:", "Solution:", "Diagram:", "Conclusion:",
]

FRAGMENTS = [
    "Q1)", "Q2)", "Q3)", "2a)", "2b)", "3a)", "3b)", "4a)", "(i)", "(ii)",
    "=", "= 8", "m = 5", "2^m - 1", "= 255 packets", "99 mod 32",
    "N = 8", "*", "->", "5", "6", "7",
]


def _rng(line_uid):
    """A generator seeded by identity, so results are position-independent."""

    digest = hashlib.sha256(line_uid.encode()).digest()

    return random.Random(int.from_bytes(digest[:8], "big"))


def _sentence(rng, target_chars):
    """Domain-plausible text of roughly the requested length."""

    if target_chars <= 12:
        return rng.choice(FRAGMENTS)

    if target_chars <= 26:
        return rng.choice(HEADINGS + FRAGMENTS)

    parts = [f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} "
             f"{rng.choice(OBJECTS)}"]

    while len(" ".join(parts)) < target_chars - 12:
        parts.append(rng.choice(TAILS))

        if len(" ".join(parts)) < target_chars - 30:
            parts.append(f"and {rng.choice(SUBJECTS)} "
                         f"{rng.choice(VERBS)} {rng.choice(OBJECTS)}")

    text = " ".join(parts)

    # trim to length on a word boundary rather than mid-word
    if len(text) > target_chars + 10:
        words = text[:target_chars].rsplit(" ", 1)[0]
        text = words or text[:target_chars]

    return text


def _confidence(rng, status, width):
    """
    0.0-1.0, shaped like a handwriting recogniser's output rather than
    a flat 0.99: mostly high, with a real tail, and worse on short
    crops where there is less context to disambiguate.
    """

    if status in ("failed", "diagram"):
        return 0.0

    if status == "empty":
        return round(rng.uniform(0.05, 0.30), 3)

    base = rng.betavariate(9, 2)          # centred ~0.82, long low tail

    if width < 150:                       # a stub with little context
        base *= rng.uniform(0.70, 0.92)

    return round(min(0.999, max(0.05, base)), 3)


def _status(rng, tall):

    if tall:
        return "diagram"

    draw = rng.random()

    if draw < FAILED_RATE:
        return "failed"

    if draw < FAILED_RATE + EMPTY_RATE:
        return "empty"

    return "ok"


def build_lines(rows):
    """One page's manifest rows -> LineResults in reading order."""

    # manifest is already sorted by (block_id, line_id), which
    # segmentation emitted top-to-bottom; that is the reading order
    ordered = sorted(rows, key=lambda r: (int(r["block_id"]),
                                          int(r["line_id"])))

    lines = []

    for order, row in enumerate(ordered):

        line_uid = (f"{row['page_id']}_b{int(row['block_id']):02d}"
                    f"_l{int(row['line_id']):02d}")

        rng = _rng(line_uid)

        tall = bool(int(row["tall"]))
        width = int(row["width"])

        status = _status(rng, tall)

        if status in ("ok",):
            text = _sentence(rng, max(1, width // PIXELS_PER_CHAR))
        else:
            text = ""

        lines.append(LineResult(
            line_uid=line_uid,
            page_id=row["page_id"],
            student=int(row["student"]),
            cie=int(row["cie"]),
            page=int(row["page"]),
            block_id=int(row["block_id"]),
            line_id=int(row["line_id"]),
            bbox=[int(row["x1"]), int(row["y1"]),
                  int(row["x2"]), int(row["y2"])],
            crop=row["crop"],
            text=text,
            confidence=_confidence(rng, status, width),
            status=status,
            reading_order=order,
            tall=tall,
        ))

    return lines


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="first N pages only")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"{MANIFEST_PATH} not found - run 02_segment/src/crop_lines.py"
        )

    with open(MANIFEST_PATH) as handle:
        rows = list(csv.DictReader(handle))

    # only regions that actually became a crop; the rest were too small
    # to hold a glyph and there is nothing for an engine to read
    croppable = [r for r in rows if r["crop"]]
    skipped = len(rows) - len(croppable)

    by_page = defaultdict(list)

    for row in croppable:
        by_page[row["page_id"]].append(row)

    page_ids = sorted(by_page)

    if args.limit:
        page_ids = page_ids[:args.limit]

    # the prepared-page path and size, so a consumer can find pixels
    segmentation = json.load(
        open(STAGE_DIR.parent / "02_segment" / "output" / "segmentation.json")
    )
    meta = {p["page_id"]: p for p in segmentation}

    pages = []

    for page_id in page_ids:

        lines = build_lines(by_page[page_id])

        info = meta.get(page_id, {})

        pages.append(PageResult(
            page_id=page_id,
            student=lines[0].student,
            cie=lines[0].cie,
            page=lines[0].page,
            source=info.get("source", ""),
            size=info.get("size", []),
            lines=lines,
        ))

    payload = {
        "schema_version": schema.SCHEMA_VERSION,
        "simulated": True,
        "warning": ("SYNTHETIC DATA. Text was generated, not recognised. "
                    "Do not report accuracy or quote content from this "
                    "as if it came from a student."),
        "engine": ENGINE,
        "generated": date.today().isoformat(),
        "pages_total": len(pages),
        "lines_total": sum(len(p.lines) for p in pages),
        "regions_without_crop": skipped,
        "pages": [p.to_dict() for p in pages],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "ocr.json", "w") as handle:
        json.dump(payload, handle, indent=1)

    with open(OUT_DIR / "lines.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line_uid", "page_id", "student", "cie", "page",
                         "block_id", "line_id", "reading_order",
                         "x1", "y1", "x2", "y2", "status", "confidence",
                         "tall", "crop", "text"])
        for page in pages:
            for line in page.lines:
                writer.writerow([
                    line.line_uid, line.page_id, line.student, line.cie,
                    line.page, line.block_id, line.line_id,
                    line.reading_order, *line.bbox, line.status,
                    line.confidence, int(line.tall), line.crop, line.text,
                ])

    # one transcript per booklet, in reading order - the form a human
    # (or a downstream answer extractor) actually wants to read
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    booklets = defaultdict(list)

    for page in pages:
        booklets[(page.student, page.cie)].append(page)

    for (student, cie), booklet_pages in sorted(booklets.items()):

        booklet_pages.sort(key=lambda p: p.page)

        lines_out = [
            "=" * 68,
            "SIMULATED OCR OUTPUT - THIS TEXT WAS GENERATED, NOT READ.",
            "Nothing here is a real student's writing.",
            "=" * 68,
            "",
            f"student_{student:02d}  cie_{cie}",
            "",
        ]

        for page in booklet_pages:
            lines_out.append(f"--- page {page.page:02d} "
                             f"({page.page_id}) ---")
            lines_out.append(page.text)
            lines_out.append("")

        (TRANSCRIPT_DIR / f"student_{student:02d}_cie_{cie}.txt").write_text(
            "\n".join(lines_out)
        )

    # never ship a payload this module's own validator would reject
    checked = schema.validate_run(payload)

    counts = defaultdict(int)

    for page in pages:
        for line in page.lines:
            counts[line.status] += 1

    print(f"Pages       : {len(pages)}")
    print(f"Lines       : {payload['lines_total']}  (validated: {checked})")
    print(f"Status      : " + ", ".join(
        f"{k}={counts[k]}" for k in schema.STATUSES if counts[k]))
    print(f"Booklets    : {len(booklets)} transcripts")
    print(f"Regions with no crop (not OCR'd): {skipped}")
    print()
    print(f"Output      : {OUT_DIR}")
    print(f"Transcripts : {TRANSCRIPT_DIR}")
    print()
    print("SIMULATED DATA - text was generated, not recognised.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
