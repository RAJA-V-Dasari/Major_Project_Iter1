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

# mark booklets against the answer keys (~2 min for all 122)
.venv/Scripts/python modules/04_evaluate/src/grade.py --all

# --- figures: the diagram pass ---------------------------------------
# package every read page for the pass (147 MB, upload via Drive)
.venv/Scripts/python experiments/diagram_pass/build_batch.py
.venv/Scripts/python experiments/diagram_pass/make_notebook.py

# rebuild the booklets with the pass as the figure source
.venv/Scripts/python modules/03_assemble/src/build_booklet.py \
    --engine all_read --diagrams experiments/diagram_pass/output/diagrams.json --all

# contact sheet for the human figure check, then rebuild with --rejects
.venv/Scripts/python modules/03_assemble/src/review_figures.py
.venv/Scripts/python modules/04_evaluate/src/grade.py --booklet student_07_cie_2 --verbose
.venv/Scripts/python modules/04_evaluate/src/grade.py --all --no-semantic   # no torch
.venv/Scripts/python modules/04_evaluate/src/grade.py --apply verdicts.jsonl

# regenerate the LLM-tier Colab notebook
.venv/Scripts/python modules/04_evaluate/src/make_grader_notebook.py
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

**Marking is a ladder, not a model.** `04_evaluate/keys/cie*.json` holds
the three official schemes broken into rubric items that each carry
their own marks, extracted by eye from the scanned PDFs in
`answer_keys/`. `grade.py` decides each item with the cheapest tier that
can honestly decide it — keyword coverage, then sentence similarity,
then an LLM on Colab, then a person — and queues rather than guesses
what none of them can settle.

Two rules in that file are load-bearing and were both written against a
measured failure; read the module docstring before loosening either.
Similarity may only *confirm* keyword evidence, never supply it, and the
LLM tier may not award marks it cannot quote the student's own words to
support.

**Thirteen questions are marked on method, not on final values.**
`add_method_marks.py` tags the ones whose steps form a chain — the five
subnets of CIE-2 3a, CIE-3 4a's delay cascade, CIE-3 4b's CRC — and
attaches the procedure plus a note on what may be carried forward. Those
110 of 180 marks would otherwise charge one early slip once per rubric
item, five times over for a single wrong block size. There is no second
model for this and there does not need to be one: the keys already hold
every exact value, so a string comparison settles final answers more
reliably than any model would, and the LLM is asked only for the part
exact matching cannot do. Re-run `add_method_marks.py` after editing a
key; it is idempotent and it owns which questions carry the tag.

**The load is asymmetric, and that is the central design fact.** Only
`02_read` needs a GPU; it runs Qwen2.5-VL-7B over a whole page and gets
0.099 CER zero-shot. Everything else is opencv/numpy and runs on a
laptop in seconds. That asymmetry is why reading sits behind
`modules/05_pipeline/src/readers.py` — `cached`, `server`, `local` — and
why the pipeline does not change when the GPU does.

**Structure comes from the reader, not from geometry.** The model emits
`### 2a)` headings and inline LaTeX, and `03_assemble` groups on those
headings. This replaced a whole line-segmentation stack — read "Deleted"
below before proposing any of it again.

**Never ask a VLM for pixel coordinates.** It was tried twice. The
original prompt asked for `![diagram](x1,y1,x2,y2)`: of six boxes drawn
back onto their pages, one enclosed its figure. The diagram pass then
asked for fractions of page height instead, and got spans snapped to
multiples of 0.05 at about half their true height. The model reads far
better than it measures, so **position comes from words it read**: each
drawing reports an `anchor`, the last few words of handwriting above it,
which `build_booklet.py` fuzzy-matches into the transcription.

**Figures: three inputs, each doing only what it is good at.**

| source | supplies | why not the others |
|---|---|---|
| diagram pass | which pages have a drawing, and its caption | the only signal that knows a drawing *is* one |
| `anchor` text | where it goes in the answer | reading order, not a guessed number |
| `figure_regions()` | the pixels, when it has a region | precise edges; blind to sparse art, fires on 37% of pages |

Where geometry has no region — the sparse node-and-edge case it cannot
see — the box comes from the reported span, padded, and trimmed to the
width of its own ink by `ink_columns()`. **Over-crop deliberately**: a
crop carrying a spare line of writing reads fine, a crop that clips an
arrow does not.

Without `--diagrams`, `build_booklet.py` keeps its old behaviour, which
appends every unclaimed geometry region at the *end* of the page. That
is where the misplaced figures came from; do not restore it.

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
  number. Measured over batch00's 250 pages, the reader emitted zero
  `![table]` and nine real `[?]`, and instead invented: empty-cell loops
  that ran to the token ceiling, a three-column table that is not on the
  page, and one `example.com` URL standing in for four legible routing
  tables. The prompt was rewritten against those findings, so scores
  taken before that rewrite do not carry over.

Prompt changes invalidate prior scores. The prompt lives once, in
`make_colab_notebook.py`, and `run_booklet.py` imports it rather than
restating it.

## Environment

Windows, PowerShell, Python 3.11 in `.venv` (created with `uv`).

`01_prepare`, `02_read`'s local half, `03_assemble` and `05_pipeline`
depend on **opencv-python-headless and numpy only** — deliberately no
torch, no transformers. Keep that path clean: when adding an import
there, check it is not pulling a GPU stack back into it.

**`04_evaluate` is the one exception**, and only since the marking stage
landed. It may use torch and sentence-transformers, because the semantic
tier needs an embedding model and MiniLM on CPU costs about two minutes
for the whole corpus. That licence does not extend outward — reading
still runs on a hosted GPU, and `grade.py --no-semantic` must keep
working so the ladder degrades rather than breaks when torch is absent.

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
