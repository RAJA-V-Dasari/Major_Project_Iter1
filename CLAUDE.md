# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Scanned handwritten exam booklets become one Markdown document per
booklet. See `README.md` for orientation, `plan.md` for the map,
`DONE.md` for what is measured, `TODO.md` for what is next.

## Commands

```bash
# score an OCR engine against the hand-transcribed pages (the arbiter)
.venv/Scripts/python modules/04_evaluate/src/ocr_bench.py --engine qwen7b --verbose
.venv/Scripts/python modules/04_evaluate/src/ocr_bench.py --list   # transcription progress

# run one booklet end to end
.venv/Scripts/python modules/05_pipeline/src/run_booklet.py student_07/cie_2 --reader cached
.venv/Scripts/python modules/05_pipeline/src/run_booklet.py --list

# group page markdown into one answer per question
.venv/Scripts/python modules/03_assemble/src/assemble.py --engine qwen7b

# package the corpus for a GPU run (writes 5 zips to 02_read/upload/)
.venv/Scripts/python modules/02_read/src/prepare_corpus_batches.py

# regenerate the Colab notebook after editing the prompt
.venv/Scripts/python modules/02_read/src/make_colab_notebook.py
```

**There is no test framework.** Verification is `ocr_bench` plus each
script's own `--check` / `--list` / `--report` flags.
`make_colab_notebook.py` compiles every generated cell and refuses to
write on a syntax error; that is the closest thing to a test here.

## Architecture

Five stages, numbered in flow order under `modules/`:

```text
01_prepare -> 02_read -> 03_assemble -> 04_evaluate
                                     -> 05_pipeline (driver)
```

**The load is asymmetric, and that is the central design fact.** Only
`02_read` needs a GPU; it runs Qwen2.5-VL-7B over a whole page and gets
0.099 CER zero-shot. Everything else is opencv/numpy and runs on a
laptop in seconds. That asymmetry is why reading sits behind
`modules/05_pipeline/src/readers.py` — `cached`, `server`, `local` — and
why the pipeline does not change when the GPU does.

**Structure comes from the reader, not from geometry.** The model emits
`### 2a)` headings, inline LaTeX, and `![diagram](x1,y1,x2,y2)` boxes in
original page pixels. `03_assemble` groups on those headings. This
replaced a whole line-segmentation stack — read "Deleted" below before
proposing any of it again.

**Box coordinates need rescaling.** The model returns them in the
resized image it was shown; the notebook scales them back to page pixels
before writing. Anything consuming boxes assumes page pixels.

## Data

Pages are **not in the repo**. `modules/01_prepare/03_tone/output` is a
Windows **directory junction** to a corpus elsewhere on disk
(`MP_Dataset/cleaned`), holding `student_NN/cie_C/page_PP.png` at
1598x2177 greyscale.

Recursive deletion through a junction can destroy the **target's**
contents rather than just the link. Delete the reparse point itself
(`cmd /c rmdir` without `/s`), never a recursive delete of a path that
contains one, and verify the page count afterwards.

There is no ingestion script; the step that built that tree was deleted
with `preprocessing/` and lives only in git history.

## Privacy — non-negotiable

`page_01` of every booklet is the identity block: name, USN, signature,
marks. It is excluded **at every stage that touches the corpus**, not
once at the start — `prepare_corpus_batches.py` by construction, again
inside the notebook on arrival, again in `run_booklet.py`. That
redundancy is deliberate: "the caller already handled it" is the
assumption that leaks student data to a hosted GPU.

Page images and transcriptions are gitignored (`upload/`,
`predictions/`, `ground_truth/`, `input/`, `output/`, `markers/`). A
transcription *is* the student's answer. Before any broad `git add`,
check what is staged — 322MB of page zips were staged once because
`modules/*/upload` was missing from `.gitignore`.

## Measurement discipline

`modules/04_evaluate/src/ocr_bench.py` is the arbiter, and comparisons
only mean something on the same pages with the same scorer. Two rules:

- **A benchmark page must never become training data.** The pages in
  `bench_pages.json` are the measurement; using one to train inflates
  the score it is measured against.
- **A lower CER is not automatically better.** If a model starts
  declining a table it used to invent, CER may barely move while the
  output becomes far more trustworthy. Read the diff, not just the
  number. The open risk today is that the reader emits zero `[?]` and
  zero `![table]` across all 15 pages — it never declines.

Prompt changes invalidate prior scores. The prompt lives once, in
`make_colab_notebook.py`, and `run_booklet.py` imports it rather than
restating it.

## Environment

Windows, PowerShell, Python 3.11 in `.venv` (created with `uv`). Local
dependencies are **opencv-python-headless and numpy only** —
deliberately no torch, no transformers; those left with the deleted
TrOCR pipeline. Keep it that way, and when adding an import check it is
not pulling a GPU stack back into the laptop path.

When writing files from PowerShell, use
`[System.IO.File]::WriteAllText(path, text, (New-Object System.Text.UTF8Encoding $false))`.
The `>` redirect and `Out-File` add a BOM, which makes Python files fail
with `invalid non-printable character U+FEFF`.

PowerShell flattens a single-element array, so a table of
`[[old, new]]` replacement pairs degrades to a bare string and
`Replace(pair[0], pair[1])` becomes a single-character replacement. Do
multi-file text edits in Python, and compile-check afterwards.

## Deleted, and why — do not resurrect without new evidence

Reading whole pages replaced these; all are in git history.

- `02_segment` line/block geometry, `03_router`, `05_math` — the reader
  needs no crops and emits LaTeX inline.
- `07_reconstruct` — found question markers geometrically then had to
  identify them. TrOCR read 2 of 8 correctly; a CNN over 576
  hand-labelled crops reached 78.4% against a 71.2% majority baseline.
  The reader gets them right from page context.
- `annotation/` — CVAT to YOLO layout labelling, abandoned at epoch 49.
- TrOCR fine-tuning — measured at ~19 usable lines per page and ~57s per
  sample on this CPU, so ~1,000 lines is ~53 pages of transcription and
  ~126 hours of training.
