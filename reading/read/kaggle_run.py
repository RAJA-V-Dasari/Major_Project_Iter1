"""
The corpus pass, driven from this laptop, on somebody else's free GPU.

    python reading/read/kaggle_run.py --user <kaggle-name> --all

Uploads the pages as a PRIVATE Kaggle dataset, builds a kernel from the
same cells `make_colab_notebook.py` already generates, pushes it, polls
until it finishes and drops the markdown into `data/read/<ENGINE>/`.
Then the rest of the pipeline runs against it with no GPU at all.

No browser, no Drive, no session to babysit. Kaggle gives 30 GPU hours a
week on a P100 (16GB) or 2xT4 (32GB total) and asks for no card, ever.

WHY THIS SHAPE AND NOT A SERVER
--------------------------------
Because free means batch. A hosted endpoint costs money the moment it is
warm, and nothing free will hold a 7B VLM resident for you. Kaggle rents
whole sessions instead, so the unit of work is a job, not a request.

That is not a compromise here, it is the design the repo already has.
`readers.py` says it plainly: a corpus pass is hours of GPU time and the
result does not change, so `CachedReader` is not a toy. This script fills
that cache. `--reader server` remains the answer for one new booklet on
demand; this is the answer for 1,231 pages at a cost of nothing.

WHY IT SPLICES RATHER THAN RESTATES
------------------------------------
The prompt and the read logic are imported from `make_colab_notebook`,
never copied. That module is the single home of the prompt, and a second
copy that drifts would silently invalidate every CER in DONE.md - the
same failure the `read_pages.py` import already guards against.

The splice point is a contract, not a line number: the Colab cells stage
pages and define `IN`, `OUT_ROOT` and `imgs`; every cell from the model
load onward consumes those three and knows nothing about where it runs.
This replaces the head, keeps the tail, and drops the one cell that calls
`google.colab`. If that contract breaks, this exits rather than shipping
a kernel that reads nothing - see `splice()`.

THE SESSION WALL
----------------
Kaggle stops a kernel somewhere around 9-12 hours depending on session
type, and the full corpus takes 8-15, so assume a single run does NOT
finish it. `--resume USER/KERNEL` mounts the
previous run's output as an input and the staging cell seeds `OUT_ROOT`
from it, so the notebook's own skip-what-exists logic picks up where it
stopped. Two runs, no lost work, still free.

PRIVACY
-------
The dataset is created private and this refuses to upload if a single
`_p01` is present. That is the third such check in the pipeline and it is
deliberate: a cover page carries a name, a USN and a signature, and
"the caller already handled it" is the assumption that leaks it.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(next(p for p in HERE.parents
                            if (p / "common" / "layout.py").exists())))

import make_colab_notebook as N            # noqa: E402  - the one prompt
from common import layout                  # noqa: E402

PAGES = layout.PAGES
# Staging holds a full copy of the corpus on its way to Kaggle, so it
# lives under the data root with everything else that is student work.
WORK = layout.READ / "kaggle"
DATASET_DIR = WORK / "dataset"
KERNEL_DIR = WORK / "kernel"

# The head the tail depends on. Defines exactly the three names every
# cell from the model load onward consumes: IN, OUT_ROOT and imgs.
STAGE = '''
import pathlib, shutil

found = sorted(pathlib.Path("/kaggle/input").glob("*/pages"))
if not found:
    raise SystemExit("no pages/ under /kaggle/input - attach the dataset")
IN = found[0]

OUT_ROOT = pathlib.Path("/kaggle/working")

# --resume mounts the previous run's output; seed OUT_ROOT from it so
# the notebook's skip-what-exists logic resumes instead of restarting.
for prior in pathlib.Path("/kaggle/input").glob("*/qwen*"):
    if prior.is_dir():
        dest = OUT_ROOT / prior.name
        dest.mkdir(parents=True, exist_ok=True)
        carried = 0
        for page in prior.glob("*.md"):
            if not (dest / page.name).exists():
                shutil.copy2(page, dest / page.name)
                carried += 1
        print("resumed", carried, "page(s) from", prior)

imgs = sorted(p for p in IN.rglob("*")
              if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
print(len(imgs), "image(s) staged")

# Third check, same reason as the other two.
covers = [p for p in imgs if p.stem.endswith("_p01")]
if covers:
    raise SystemExit("COVER PAGES PRESENT (%d) - remove them first: %s"
                     % (len(covers), [p.name for p in covers[:5]]))
print("no cover pages present")
'''


def run(command, **kwargs):
    """Shell out to the kaggle CLI, loudly."""

    print("  $ " + " ".join(command))
    return subprocess.run(command, check=True, text=True, **kwargs)


def splice():
    """Colab's cells, with the staging head replaced by Kaggle's.

    The tail is found by content, not by index, so reordering cells in
    `make_colab_notebook.py` does not silently truncate the kernel.
    """

    cells = N.CELLS

    head = [i for i, c in enumerate(cells)
            if c["cell_type"] == "code" and "RUN = " in "".join(c["source"])]

    if len(head) != 1:
        raise SystemExit(
            "expected exactly one cell defining RUN, found %d. The splice "
            "contract in make_colab_notebook.py has changed - read the "
            "docstring in this file before editing it." % len(head))

    tail = [c for c in cells[head[0]:]
            if "google.colab" not in "".join(c["source"])]

    if not any("OUT_ROOT" in "".join(c["source"]) for c in tail):
        raise SystemExit("tail never uses OUT_ROOT - splice point is wrong")

    return [
        N.md("# Reading answer scripts with a VLM - Kaggle batch\n\n"
             "Generated by `kaggle_run.py`. Do not edit here; edit "
             "`make_colab_notebook.py` and push again."),
        N.code('!pip -q install "transformers>=4.49" accelerate '
               'qwen-vl-utils bitsandbytes'),
        N.code(STAGE),
    ] + tail


def build_kernel(user, slug, dataset, resume_from=None):
    """Write the kernel directory the CLI pushes."""

    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    notebook = {
        "cells": splice(),
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    # Same guard make_colab_notebook.py uses: refuse to ship a kernel
    # whose cells will not compile, because the failure otherwise shows
    # up twenty minutes into a queued run.
    for index, cell in enumerate(notebook["cells"]):

        source = "".join(cell["source"])

        if cell["cell_type"] != "code" or source.lstrip().startswith("!"):
            continue

        try:
            compile(source, "<cell %d>" % index, "exec")
        except SyntaxError as error:
            raise SystemExit("BROKEN cell %d, line %s: %s"
                             % (index, error.lineno, error.msg))

    (KERNEL_DIR / "read_pages.ipynb").write_text(
        json.dumps(notebook, indent=1), encoding="utf-8")

    metadata = {
        "id": "%s/%s" % (user, slug),
        "title": "Answer script reading",
        "code_file": "read_pages.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,          # the model downloads from HF
        "dataset_sources": [dataset],
        "competition_sources": [],
        "kernel_sources": [resume_from] if resume_from else [],
    }

    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=1), encoding="utf-8")

    print("  kernel -> %s (%d cells)" % (KERNEL_DIR, len(notebook["cells"])))


def stage_dataset(user, slug, batch=None):
    """Copy pages into a dataset directory, covers refused.

    Pages are renamed to the manifest key on the way in - `s01_c2_p03`,
    not `page_03`. Everything downstream keys on that, and a cache keyed
    on the filename matches nothing and reports an empty booklet.
    """

    if not PAGES.exists():
        raise SystemExit("pages not found: %s" % PAGES)

    target = DATASET_DIR / "pages"

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    staged = covers = 0

    for page in sorted(PAGES.rglob("*.png")):

        booklet = page.parent.name             # cie_2
        student = page.parent.parent.name      # student_07

        key = "s%s_c%s_p%s" % (student.replace("student_", ""),
                               booklet.replace("cie_", ""),
                               page.stem.replace("page_", ""))

        if key.endswith("_p01"):
            covers += 1
            continue

        if batch and batch not in str(page):
            continue

        shutil.copy2(page, target / (key + ".png"))
        staged += 1

    print("  staged %d page(s), excluded %d cover(s)" % (staged, covers))

    if not staged:
        raise SystemExit("nothing staged - check the corpus junction")

    (DATASET_DIR / "dataset-metadata.json").write_text(json.dumps(
        {"title": "Answer script pages", "id": "%s/%s" % (user, slug),
         "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")

    return staged


def upload_dataset(user, slug):
    """Create the private dataset, or add a version if it exists."""

    probe = subprocess.run(["kaggle", "datasets", "status",
                            "%s/%s" % (user, slug)],
                           capture_output=True, text=True)

    exists = probe.returncode == 0 and "404" not in (probe.stdout or "")

    if exists:
        run(["kaggle", "datasets", "version", "-p", str(DATASET_DIR),
             "-m", "refresh", "--dir-mode", "zip"])
    else:
        run(["kaggle", "datasets", "create", "-p", str(DATASET_DIR),
             "--dir-mode", "zip"])

    print("  waiting for Kaggle to ingest the dataset...")
    time.sleep(30)


def poll(user, slug, every=60):
    """Block until the kernel stops, printing what it is doing."""

    while True:

        result = subprocess.run(
            ["kaggle", "kernels", "status", "%s/%s" % (user, slug)],
            capture_output=True, text=True)

        status = (result.stdout or result.stderr).strip()
        print("  %s  %s" % (time.strftime("%H:%M:%S"), status))

        lowered = status.lower()

        if "complete" in lowered:
            return True
        if "error" in lowered or "cancel" in lowered:
            return False

        time.sleep(every)


def fetch(user, slug, engine):
    """Pull the markdown into the cache every other stage reads."""

    destination = layout.read_dir(engine)
    destination.mkdir(parents=True, exist_ok=True)

    run(["kaggle", "kernels", "output", "%s/%s" % (user, slug),
         "-p", str(WORK / "output")])

    pulled = 0

    for page in (WORK / "output").rglob("*.md"):
        shutil.copy2(page, destination / page.name)
        pulled += 1

    print("  %d page(s) -> %s" % (pulled, destination))

    if not pulled:
        print("  NOTHING PULLED - check the kernel log on kaggle.com")

    return pulled


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Kaggle username")
    parser.add_argument("--dataset-slug", default="answer-script-pages")
    parser.add_argument("--kernel-slug", default="answer-script-reading")
    parser.add_argument("--engine", default=layout.DEFAULT_ENGINE,
                        help="folder under data/read/ to fetch into")
    parser.add_argument("--batch", help="substring filter, e.g. cie_2")
    parser.add_argument("--resume", metavar="USER/KERNEL",
                        help="mount a previous run's output and continue")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="stage and build, push nothing")

    args = parser.parse_args()

    if not any([args.upload, args.push, args.fetch, args.all, args.dry_run]):
        parser.error("pick one of --upload / --push / --fetch / --all")

    dataset = "%s/%s" % (args.user, args.dataset_slug)

    if args.upload or args.all or args.dry_run:
        print("staging pages")
        stage_dataset(args.user, args.dataset_slug, args.batch)

        if not args.dry_run:
            print("uploading dataset")
            upload_dataset(args.user, args.dataset_slug)

    if args.push or args.all or args.dry_run:
        print("building kernel")
        build_kernel(args.user, args.kernel_slug, dataset, args.resume)

        if args.dry_run:
            print("dry run - nothing pushed")
            return 0

        print("pushing kernel")
        run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)])

        print("running (session wall ~9-12h; --resume continues a cut-off run)")

        if not poll(args.user, args.kernel_slug):
            raise SystemExit("kernel did not complete - see kaggle.com")

    if args.fetch or args.all:
        print("fetching markdown")
        fetch(args.user, args.kernel_slug, args.engine)

    print("\ndone. Now, with no GPU:")
    print("  python pipeline.py run --engine %s" % args.engine)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
