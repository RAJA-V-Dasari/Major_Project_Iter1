"""
Package the assembled booklets as the contract the marking half reads.

    data/booklets/<booklet>/structure.json   what build_booklet.py decided
    data/booklets/<booklet>/figures/*.png    the crops it cut
    data/pages/                              the prepared page images
        -> data/handoff/index.json
        -> data/handoff/<booklet>/booklet.json   questions -> parts -> answer[]
        -> data/handoff/<booklet>/pages/page_NN.png
        -> data/handoff/<booklet>/regions/*.png

THE CONTRACT
------------
docs/HANDOFF.md is the schema and marking/src/load_handoff.py is the code
that reads it. This writes schema 2.0, the shape part 1 shipped to the
marking half in September, so marking reads a booklet assembled here
exactly as it read those - `load_handoff.py --report` re-derives the
totals in index.json and refuses to go on if they disagree.

AN ANSWER IS A LIST, NOT A STRING
---------------------------------
Each part's `answer` is the ordered list of places it was found, each
carrying its page:

    text       a paragraph the reader transcribed
    diagram    a crop of the page; marking routes items whose evidence is
               a drawing to a person, who sees this image
    crossed    work the student struck out (~~...~~); carried so a marker
               can see it, `excluded: true` so nothing scores it
    gap        a page that was never read; marking treats the part as
               incomplete, not short, and will not zero it

`part.text` is a convenience join of the readable prose and is
redundant by design - the list is the answer.

WHAT IS DROPPED ON THE WAY
--------------------------
Only build_booklet's own notes to a human reader ("Reading failed
partway down this page", "figure not located"), matched by their exact
prefixes. They are scaffolding, not the student's words. The full-page
image that follows a failed reading is kept, flagged `reading_failed`.

Run:
    python reading/handoff/export.py
    python reading/handoff/export.py --check       # report, write nothing
    python reading/handoff/export.py --no-images   # JSON only
"""

import argparse
import json
import re
import shutil
import sys
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "common" / "layout.py").exists())

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reading" / "assemble"))

from common import layout                                  # noqa: E402
from build_booklet import FAILED_NOTE, NOT_LOCATED_NOTE    # noqa: E402

SCHEMA_VERSION = "2.0"

IMAGE_LINE = re.compile(r"^!\[(?P<caption>[^\]]*)\]\((?P<file>figures/[^)\s]+)\)$")
STRUCK = re.compile(r"~~(.+?)~~")
FENCE = re.compile(r"^```")
EMPHASIS = re.compile(r"^(\*{1,2})(?P<text>.+?)\1$")
LABEL = re.compile(r"^(?P<number>[1-9])(?P<part>[a-e])?$")


def split_label(label):
    """'2a' -> ('2', 'a'); '1' -> ('1', None); None -> (None, None)."""

    match = LABEL.match(label or "")
    if not match:
        return None, None
    return match.group("number"), match.group("part")


def items_from_lines(lines, page_id, source, crops, booklet_dir, out_dir,
                     with_images):
    """One run of Markdown lines from one page -> answer items, in order."""

    items = []
    paragraph, struck = [], []

    def flush():
        text = "\n".join(line for line in paragraph).strip()
        if text:
            items.append({"type": "text", "page_id": page_id, "text": text,
                          "excluded": False, "source": source})
        for piece in struck:
            items.append({"type": "crossed", "page_id": page_id,
                          "text": piece, "excluded": True, "source": source})
        paragraph.clear()
        struck.clear()

    for raw in lines:
        line = raw.strip()

        if not line:
            flush()
            continue

        if line.startswith((FAILED_NOTE, NOT_LOCATED_NOTE)) or \
                FENCE.match(line):
            continue

        image = IMAGE_LINE.match(line)
        if image:
            flush()
            rel = image.group("file")
            meta = crops.get(rel, {})
            name = Path(rel).name
            source_image = booklet_dir / rel
            if with_images and source_image.exists():
                (out_dir / "regions").mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, out_dir / "regions" / name)
            flags = {"page": ["reading_failed", "full_page"],
                     "unreferenced": ["unreferenced"]}.get(meta.get("kind"),
                                                           [])
            items.append({
                "type": "diagram",
                "page_id": meta.get("page_id", page_id),
                "image": f"regions/{name}",
                "bbox": meta.get("bbox"),
                "caption": image.group("caption"),
                "excluded": False,
                "flags": flags,
            })
            continue

        emphasis = EMPHASIS.match(line)
        if emphasis:
            line = emphasis.group("text").strip()

        # Whole-line strike-outs become their own item; inline ones are
        # lifted out of the prose so nothing scores a withdrawn claim.
        pieces = STRUCK.findall(line)
        if pieces:
            struck.extend(p.strip() for p in pieces if p.strip())
            line = STRUCK.sub(" ", line).strip()
            if not line:
                continue

        paragraph.append(line)

    flush()
    return items


def build_parts(structure, booklet_dir, out_dir, with_images):
    """The booklet's reading-order stream as questions -> parts -> answer.

    Parts are ordered by first appearance, not sorted: the order a student
    answered in is information, and marking does its own lookup by label.
    """

    source = f"read:{structure['engine']}"
    crops = structure.get("crops", {})
    headings = structure.get("headings", {})

    parts = OrderedDict()

    for entry in structure["stream"]:
        label = entry["label"]
        if label not in parts:
            number, part = split_label(label)
            parts[label] = {
                "number": number,
                "part": part,
                "label": label,
                "raw_marker": headings.get(label),
                "answer": [],
            }
            if entry.get("recovered"):
                parts[label]["recovered"] = True

        if "gap" in entry:
            parts[label]["answer"].append({
                "type": "gap", "page_id": entry["page_id"],
                "reason": entry["gap"]})
            continue

        parts[label]["answer"].extend(items_from_lines(
            entry["lines"], entry["page_id"], source, crops, booklet_dir,
            out_dir, with_images))

    questions = OrderedDict()

    for part in parts.values():
        pages = []
        for item in part["answer"]:
            if item["page_id"] not in pages:
                pages.append(item["page_id"])
        part["pages"] = pages
        part["crosses_page_break"] = len(pages) > 1
        part["text"] = " ".join(
            i["text"] for i in part["answer"]
            if i["type"] == "text" and not i["excluded"]) or None
        part["counts"] = dict(Counter(i["type"] for i in part["answer"]))

        questions.setdefault(part["number"], {"number": part["number"],
                                              "parts": []})
        questions[part["number"]]["parts"].append(part)

    return list(questions.values())


def page_records(structure, out_dir, with_images):
    """Every page of the booklet that exists here, with its role.

    cover    page 1, only present if it was fetched with --with-covers
    answer   something read on it was kept
    blank    read, and nothing on it survived
    unread   never transcribed - the matching gap item says so in the answer
    """

    student, cie = structure["student"], structure["cie"]
    folder = layout.PAGES / f"student_{student:02d}" / f"cie_{cie}"

    kept = {e["page_id"] for e in structure["stream"] if "lines" in e}
    read = {p["page_id"]: p["read"] for p in structure["pages"]}

    records = []
    for path in sorted(folder.glob("page_*.png")):
        number = int(re.search(r"(\d+)", path.stem).group(1))
        pid = layout.page_id(student, cie, number)
        if number == layout.COVER_PAGE:
            role = "cover"
        elif not read.get(pid, False):
            role = "unread"
        elif pid in kept:
            role = "answer"
        else:
            role = "blank"

        image = None
        if with_images:
            (out_dir / "pages").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out_dir / "pages" / path.name)
            image = f"pages/{path.name}"

        records.append({"page_id": pid, "n": number, "role": role,
                        "image": image})
    return records


def export_booklet(structure, booklet_dir, out_root, with_images, write):
    out_dir = out_root / structure["booklet_id"]
    if write:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

    images = with_images and write
    questions = build_parts(structure, booklet_dir, out_dir, images)
    pages = page_records(structure, out_dir, images)

    parts = [p for q in questions for p in q["parts"]]
    items = [i for p in parts for i in p["answer"]]
    kinds = Counter(i["type"] for i in items)
    content_pages = [p for p in pages if p["role"] != "cover"]

    coverage = {
        "questions_total": len(questions),
        "parts_total": len(parts),
        "parts_labelled": sum(1 for p in parts if p["label"]),
        "items_total": len(items),
        "diagrams": kinds.get("diagram", 0),
        "crossed": kinds.get("crossed", 0),
        "gaps": kinds.get("gap", 0),
        "pages_total": len(content_pages),
        "pages_read": sum(1 for p in content_pages
                          if p["role"] in ("answer", "blank")),
        "model": structure["engine"],
        "recovered_question_1": structure.get("recovered_question_1", False),
    }

    booklet = {
        "schema_version": SCHEMA_VERSION,
        "booklet_id": structure["booklet_id"],
        "identity": {
            "extracted": False,
            "source": None,
            "warning": ("Not read. No stage reads the cover; the corpus is "
                        "anonymised on disk as student_NN, and only cie and "
                        "the corpus number are known."),
            "fields": {
                "cie": {"value": structure["cie"], "confidence": 1.0},
                "corpus_student": {"value": structure["student"],
                                   "confidence": 1.0},
            },
        },
        "pages": pages,
        "questions": questions,
        "coverage": coverage,
    }

    if write:
        (out_dir / "booklet.json").write_text(
            json.dumps(booklet, indent=2, ensure_ascii=False),
            encoding="utf-8")

    return booklet


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check", action="store_true",
                        help="report what would be exported, write nothing")
    parser.add_argument("--no-images", action="store_true",
                        help="booklet.json only, no page images or crops")
    args = parser.parse_args()

    layout.require(layout.BOOKLETS, "assembled booklets",
                   "Assemble them first: python pipeline.py assemble")

    sources = sorted(layout.BOOKLETS.glob("*/structure.json"))
    if not sources:
        raise SystemExit(f"no structure.json under {layout.BOOKLETS} - "
                         "assemble the booklets first: "
                         "python pipeline.py assemble")

    structures = [(json.loads(p.read_text(encoding="utf-8")), p.parent)
                  for p in sources]

    engines = Counter(s["engine"] for s, _ in structures)
    if len(engines) > 1:
        raise SystemExit(
            f"the booklets were assembled from more than one read: "
            f"{dict(engines)}. Marking them together would mix readers; "
            "re-run: python pipeline.py assemble --engine <one of them>")

    write = not args.check
    out_root = layout.HANDOFF

    if write:
        out_root.mkdir(parents=True, exist_ok=True)
        keep = {s["booklet_id"] for s, _ in structures}
        for stale in out_root.iterdir():
            if (stale.is_dir() and layout.BOOKLET_ID.match(stale.name)
                    and stale.name not in keep):
                shutil.rmtree(stale)

    entries = []
    for structure, booklet_dir in structures:
        booklet = export_booklet(structure, booklet_dir, out_root,
                                 not args.no_images, write)
        entries.append({
            "booklet_id": booklet["booklet_id"],
            "path": f"{booklet['booklet_id']}/booklet.json",
            **booklet["coverage"],
        })

    totals = {key: sum(e[key] for e in entries) for key in (
        "questions_total", "parts_total", "parts_labelled", "items_total",
        "diagrams", "crossed", "gaps", "pages_read", "pages_total")}

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "reading/handoff/export.py",
        "contract": "docs/HANDOFF.md",
        "engine": next(iter(engines)),
        "booklets_total": len(entries),
        # Named as marking/src/load_handoff.py --report reads them.
        "totals": {
            "questions": totals["questions_total"],
            "parts": totals["parts_total"],
            "parts_labelled": totals["parts_labelled"],
            "items": totals["items_total"],
            "diagrams": totals["diagrams"],
            "crossed": totals["crossed"],
            "gaps": totals["gaps"],
            "pages_read": totals["pages_read"],
            "pages_total": totals["pages_total"],
        },
        "booklets": entries,
    }

    if write:
        (out_root / "index.json").write_text(json.dumps(index, indent=2),
                                             encoding="utf-8")

    t = index["totals"]
    print(f"booklets : {len(entries)}  (read by {index['engine']})")
    print(f"pages    : {t['pages_read']}/{t['pages_total']} content pages read")
    print(f"parts    : {t['parts']} ({t['parts_labelled']} labelled) in "
          f"{t['questions']} questions")
    print(f"items    : {t['items']} - {t['diagrams']} diagrams, "
          f"{t['crossed']} crossed out, {t['gaps']} gaps")

    if not write:
        print("\n(check only - nothing written)")
        return 0

    print(f"\nhandoff  : {out_root / 'index.json'}")
    if layout.handoff_data() != out_root:
        print(f"note     : MPE_HANDOFF is set, so marking will read "
              f"{layout.handoff_data()} - not this export")
    print("next     : python pipeline.py mark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
