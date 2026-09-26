"""
One booklet in, Markdown out. The whole pipeline behind one command.

    python run_booklet.py student_07/cie_2 --reader cached
    python run_booklet.py student_07/cie_2 --reader server --url http://gpu:8000/v1
    python run_booklet.py --all --reader cached

        -> 05_pipeline/output/<student>/<cie>/pages/*.md   per page
        -> 05_pipeline/output/<student>/<cie>/booklet.md   one document
        -> 05_pipeline/output/<student>/<cie>/run.json     what produced it

WHAT THIS IS
------------
The answer to "I hand it a booklet and it works". Everything except
reading a page is plain Python and runs here in seconds; reading is
the only step wanting a GPU, so it is the only step behind an
interface (see readers.py). The pipeline itself does not change when
that backend does.

WHY COVER PAGES ARE DROPPED HERE TOO
-------------------------------------
page_01 is the identity block - name, USN, signature, marks. It is
excluded at every stage that touches the corpus rather than once at
the start, because "the caller already handled it" is exactly the
assumption that leaks student data to a hosted GPU.

THE OUTPUT IS ONE DOCUMENT, NOT A PILE OF PAGES
------------------------------------------------
An answer routinely runs across a page break, so per-page Markdown is
an intermediate, not the deliverable. booklet.md concatenates the
pages in order, and 03_assemble regroups that by question number - the
question headings the recogniser already read are what make that
possible, and are why the marker CNN is no longer needed.

RUN.JSON IS NOT OPTIONAL BOOKKEEPING
-------------------------------------
A CER belongs to a model, not to a pipeline. Markdown that does not
record which reader produced it cannot be interpreted later - 0.099
and 0.229 are different enough to change what you would trust it for.
Every run writes its reader, model and timings next to its output.

Run:
    python run_booklet.py --list
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES = STAGE_DIR.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MODULES / "02_read" / "src"))

import readers as R                       # noqa: E402

PAGES_DIR = MODULES / "01_prepare" / "03_tone" / "output"
OUT_DIR = STAGE_DIR / "output"

COVER_PAGE = 1

DEFAULT_CACHE = MODULES / "04_evaluate" / "predictions" / "qwen7b"


def prompt_text():
    """The same prompt the benchmark used - imported, never retyped.

    A prompt that drifts from the one a score was measured with makes
    the score meaningless, so there is exactly one copy of it.
    """

    import make_colab_notebook as N

    return N.PROMPT


def booklet_pages(relative):
    """Content pages of one booklet, in order."""

    directory = PAGES_DIR / relative

    if not directory.exists():
        raise SystemExit(f"not found: {directory}")

    found = []

    for path in sorted(directory.glob("page_*.png")):

        number = int(re.search(r"(\d+)", path.stem).group(1))

        if number == COVER_PAGE:
            continue                       # identity block

        found.append((number, path))

    return found


def page_id(relative, number):
    student = int(re.search(r"student_(\d+)", relative).group(1))
    cie = int(re.search(r"cie_(\d+)", relative).group(1))
    return f"s{student:02d}_c{cie}_p{number:02d}"


def all_booklets():
    found = []
    for student in sorted(PAGES_DIR.glob("student_*")):
        for cie in sorted(student.glob("cie_*")):
            found.append(f"{student.name}/{cie.name}")
    return found


def run(relative, reader, force=False):

    pages = booklet_pages(relative)

    if not pages:
        return None

    target = OUT_DIR / relative
    page_dir = target / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    read_count = 0
    reused = 0
    missing = []

    bodies = []

    for number, path in pages:

        key = page_id(relative, number)
        out_path = page_dir / f"{key}.md"

        if out_path.exists() and out_path.stat().st_size and not force:
            bodies.append((number, out_path.read_text(encoding="utf-8")))
            reused += 1
            continue

        body = reader.read(path, key)

        if body is None:
            missing.append(key)
            continue

        out_path.write_text(body, encoding="utf-8")
        bodies.append((number, body))
        read_count += 1

    if not bodies:
        return {"booklet": relative, "pages": 0, "missing": missing}

    # one document, in reading order, with a page marker that survives
    # into the assembled answers so a claim can be traced to a page
    document = []
    for number, body in sorted(bodies):
        document.append(f"<!-- page {number:02d} -->")
        document.append(body.strip())
        document.append("")

    (target / "booklet.md").write_text("\n".join(document), encoding="utf-8")

    record = {
        "booklet": relative,
        "pages": len(bodies),
        "read": read_count,
        "reused": reused,
        "missing": missing,
        "seconds": round(time.time() - started, 1),
        **reader.provenance(),
    }

    (target / "run.json").write_text(json.dumps(record, indent=1),
                                     encoding="utf-8")

    return record


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("booklet", nargs="?",
                        help="e.g. student_07/cie_2")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="re-read pages already done")

    parser.add_argument("--reader", default="cached",
                        choices=["cached", "server", "local"])
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--engine", default="qwen7b")
    parser.add_argument("--url")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-path")
    parser.add_argument("--mmproj-path")

    args = parser.parse_args()

    if args.list:
        found = all_booklets()
        print(f"{len(found)} booklet(s) under {PAGES_DIR}")
        for name in found[:20]:
            print(f"  {name}")
        if len(found) > 20:
            print(f"  ... and {len(found) - 20} more")
        return 0

    if not args.booklet and not args.all:
        raise SystemExit("give a booklet, or --all (or --list)")

    options = {"cache_dir": args.cache_dir, "engine": args.engine,
               "url": args.url, "model": args.model,
               "model_path": args.model_path,
               "mmproj_path": args.mmproj_path}

    if args.reader == "server" and not args.url:
        raise SystemExit("--reader server needs --url")

    if args.reader == "local" and not (args.model_path and args.mmproj_path):
        raise SystemExit("--reader local needs --model-path and --mmproj-path")

    reader = R.build(args.reader, prompt_text(), **options)

    targets = all_booklets() if args.all else [args.booklet]

    print(f"Reader   : {reader.provenance()}")
    print(f"Booklets : {len(targets)}\n")

    done, empty = 0, 0

    for relative in targets:

        record = run(relative, reader, force=args.force)

        if record is None or not record.get("pages"):
            print(f"  {relative:<24} no pages read")
            empty += 1
            continue

        note = (f"{record['pages']:>3} pages "
                f"({record['read']} read, {record['reused']} reused) "
                f"{record['seconds']:>6.1f}s")

        if record["missing"]:
            note += f"  MISSING {len(record['missing'])}"

        print(f"  {relative:<24} {note}")
        done += 1

    print(f"\nCompleted : {done}/{len(targets)}"
          + (f", {empty} with nothing to read" if empty else ""))
    print(f"Output    : {OUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
