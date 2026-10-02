"""
Read prepared pages into Markdown, one file per page, through any reader.

    python reading/read/read_pages.py                      # coverage only
    python reading/read/read_pages.py --all --reader server --url http://gpu:8000/v1
    python reading/read/read_pages.py student_07/cie_2 --reader local \
        --model-path qwen2.5-vl-3b.gguf --mmproj-path mmproj.gguf
    python reading/read/read_pages.py --all --reader cached --cache-dir <folder>

    data/pages/student_NN/cie_C/page_PP.png
        -> data/read/<engine>/<page_id>.md      one transcription per page
        -> data/read/<engine>/_runs.jsonl       which reader produced them

WHAT THIS IS
------------
The answer to "I hand it a booklet and it works". Reading a page is the
only step in the whole project that wants a GPU, so it is the only step
behind an interface (readers.py). Everything after it reads
data/read/<engine>/ and neither knows nor cares which reader filled it:

    cached   copy transcriptions that already exist (an unzipped Colab or
             Kaggle download) into data/read/<engine>/. With no
             --cache-dir it reads nothing and reports coverage - which
             pages of which booklets are read and which are not.
    server   any OpenAI-compatible endpoint serving the VLM (vLLM,
             llama.cpp, Modal - see modal_vllm.py)
    local    llama.cpp in-process with a GGUF model: no GPU, no network,
             minutes per page

The batch routes - the Colab notebook and kaggle_run.py - write the same
files into the same folder, so it does not matter which one ran.

WHY COVER PAGES ARE DROPPED HERE TOO
-------------------------------------
page_01 is the identity block - name, USN, signature, marks. It is
excluded at every stage that touches the corpus rather than once at
the start, because "the caller already handled it" is exactly the
assumption that leaks student data to a hosted GPU.

THE RUN LOG IS NOT OPTIONAL BOOKKEEPING
---------------------------------------
A CER belongs to a model, not to a pipeline. Markdown that does not
record which reader produced it cannot be interpreted later - 0.099
and 0.229 are different enough to change what you would trust it for.
Every run appends its reader, model and counts to _runs.jsonl beside
the pages it wrote.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(next(p for p in HERE.parents
                            if (p / "common" / "layout.py").exists())))

import readers as R                       # noqa: E402
from common import layout                 # noqa: E402

PAGES_DIR = layout.PAGES


def prompt_text():
    """The same prompt the benchmark used - imported, never retyped.

    A prompt that drifts from the one a score was measured with makes
    the score meaningless, so there is exactly one copy of it.
    """

    import make_colab_notebook as N

    return N.PROMPT


def booklet_pages(relative):
    """Content pages of one booklet, in order, as (number, path)."""

    directory = PAGES_DIR / relative

    if not directory.exists():
        raise SystemExit(f"not found: {directory}")

    found = []

    for path in sorted(directory.glob("page_*.png")):

        number = int(re.search(r"(\d+)", path.stem).group(1))

        if number == layout.COVER_PAGE:
            continue                       # identity block

        found.append((number, path))

    return found


def page_id(relative, number):
    student = int(re.search(r"student_(\d+)", relative).group(1))
    cie = int(re.search(r"cie_(\d+)", relative).group(1))
    return layout.page_id(student, cie, number)


def all_booklets():
    found = []
    for student in sorted(PAGES_DIR.glob("student_*")):
        for cie in sorted(student.glob("cie_*")):
            found.append(f"{student.name}/{cie.name}")
    return found


def run(relative, reader, out_dir, force=False):
    """Read one booklet's missing pages. Returns a coverage record."""

    pages = booklet_pages(relative)

    if not pages:
        return None

    counts = {"pages": len(pages), "read": 0, "reused": 0}
    missing = []

    for number, path in pages:

        key = page_id(relative, number)
        out_path = out_dir / f"{key}.md"

        if out_path.exists() and out_path.stat().st_size and not force:
            counts["reused"] += 1
            continue

        body = reader.read(path, key) if reader else None

        if body is None:
            missing.append(key)
            continue

        out_path.write_text(body, encoding="utf-8")
        counts["read"] += 1

    return {"booklet": relative, **counts, "missing": missing}


def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("booklet", nargs="?",
                        help="e.g. student_07/cie_2 (default: every booklet)")
    parser.add_argument("--all", action="store_true",
                        help="every booklet under data/pages/")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="re-read pages already done")

    parser.add_argument("--engine", default=layout.DEFAULT_ENGINE,
                        help="the folder under data/read/ to fill "
                             f"(default: {layout.DEFAULT_ENGINE})")
    parser.add_argument("--reader", default="cached",
                        choices=["cached", "server", "local"])
    parser.add_argument("--cache-dir",
                        help="cached reader: a folder of <page_id>.md to "
                             "copy in (default: none - report coverage)")
    parser.add_argument("--url")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-path")
    parser.add_argument("--mmproj-path")

    args = parser.parse_args()

    layout.require(PAGES_DIR, "prepared pages",
                   "Fetch them: python pipeline.py fetch")

    if args.list:
        found = all_booklets()
        print(f"{len(found)} booklet(s) under {PAGES_DIR}")
        for name in found[:20]:
            print(f"  {name}")
        if len(found) > 20:
            print(f"  ... and {len(found) - 20} more")
        return 0

    out_dir = layout.read_dir(args.engine)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = None

    if args.reader == "server":
        if not args.url:
            raise SystemExit("--reader server needs --url")
        reader = R.build("server", prompt_text(), url=args.url,
                         model=args.model)

    elif args.reader == "local":
        if not (args.model_path and args.mmproj_path):
            raise SystemExit("--reader local needs --model-path and "
                             "--mmproj-path")
        reader = R.build("local", prompt_text(), model_path=args.model_path,
                         mmproj_path=args.mmproj_path)

    elif args.cache_dir:
        if Path(args.cache_dir).resolve() != out_dir.resolve():
            reader = R.build("cached", None, cache_dir=args.cache_dir,
                             engine=args.engine)

    targets = [args.booklet] if args.booklet else all_booklets()

    if not targets:
        raise SystemExit(f"no booklets under {PAGES_DIR}")

    what = reader.provenance() if reader else {"reader": "none (coverage)"}
    print(f"Reader   : {what}")
    print(f"Pages to : {out_dir}")
    print(f"Booklets : {len(targets)}\n")

    started = time.time()
    totals = {"pages": 0, "read": 0, "reused": 0, "missing": 0}
    complete = 0

    for relative in targets:

        record = run(relative, reader, out_dir, force=args.force)

        if record is None:
            continue

        for key in ("pages", "read", "reused"):
            totals[key] += record[key]
        totals["missing"] += len(record["missing"])
        complete += not record["missing"]

        note = (f"{record['pages']:>3} pages  {record['read']:>3} read  "
                f"{record['reused']:>3} already there")
        if record["missing"]:
            note += f"  {len(record['missing'])} NOT READ"
        print(f"  {relative:<24} {note}")

    print(f"\n{totals['reused'] + totals['read']}/{totals['pages']} content "
          f"pages read ({totals['read']} this run), {complete}/{len(targets)} "
          f"booklets complete")

    if totals["missing"] and reader is None:
        print("\nTo read the rest, either:\n"
              "  - run the Colab notebook (reading/read/read_pages_colab.ipynb)"
              " and unzip\n    its download into "
              f"{out_dir}\n"
              "  - or point this at a model: --reader server --url ...")

    if reader is not None and totals["read"]:
        with open(out_dir / "_runs.jsonl", "a", encoding="utf-8") as log:
            log.write(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **reader.provenance(),
                "booklets": len(targets),
                "pages_read": totals["read"],
                "seconds": round(time.time() - started, 1),
            }) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
