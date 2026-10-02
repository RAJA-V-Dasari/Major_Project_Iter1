# Reading: scanned booklets to structured answers

The first half of the pipeline. A prepared page goes to a vision model,
comes back as Markdown, and is grouped by question into one booklet per
student per CIE. The booklets are then packaged as the handoff the
marking half reads.

```
data/pages ──read──> data/read/<engine> ──assemble──> data/booklets ──handoff──> data/handoff
            [GPU]                          (+ data/figures from the diagram pass)
```

Measured at **0.099 character error rate** on 15 hand-transcribed pages,
zero-shot, with no training and no labelled data
([`benchmark/RESULTS.md`](benchmark/RESULTS.md)).

---

## What is where

| folder | stage | scripts |
|---|---|---|
| [`prepare/`](prepare/) | raw scan → prepared page (only to rebuild `data/pages`) | `deskew.py`, `measure.py`, `crop.py`, `tone.py`; `tools/` holds two unused groundwork tools |
| [`read/`](read/) | page → Markdown, behind a swappable reader | `make_colab_notebook.py` (**the prompt**), `read_pages_colab.ipynb`, `read_pages.py`, `readers.py`, `prepare_corpus_batches.py`, `prepare_test_batch.py`, `kaggle_run.py`, `modal_vllm.py` |
| [`read/qa/`](read/qa/) | checking a read | `check_batch.py`, `audit_pages.py`, `compare_runs.py`, `show_page.py` |
| [`figures/`](figures/) | the diagram pass | `build_batch.py`, `make_notebook.py` → `find_diagrams.ipynb`, `labels.py` (its hand-labelled benchmark) |
| [`assemble/`](assemble/) | pages → booklets, with figures cropped | `build_booklet.py`, `segment.py` (geometry), `review_figures.py` |
| [`handoff/`](handoff/) | booklets → the marking contract | `export.py` |
| [`benchmark/`](benchmark/) | the reader's CER against hand transcription | `ocr_bench.py`, `bench_pages.json`, `RESULTS.md` |
| [`docs/`](docs/) | what was measured and decided | `DONE.md`, `TODO.md`, `BATCH00_REVIEW.md`, `SEGMENT.md`, `DATASET_CARD.md` |

Run the stages through `python pipeline.py <stage>` from the repo root
([`../docs/PIPELINE.md`](../docs/PIPELINE.md)), or any script directly:
each says how in its docstring.

---

## The design facts that shape everything here

**The load is asymmetric.** Only reading a page needs a GPU. It runs
Qwen2.5-VL-7B over a whole page, on Colab, Kaggle or a server.
Everything else is opencv and numpy on a laptop, in seconds. That is why
reading sits behind `read/readers.py`, and why nothing downstream
changes when the GPU does.

**Whole pages beat segmented lines, by 4.7×.** The original pipeline cut
pages into lines and read each with TrOCR: 0.463 CER, and the errors
were fabrication rather than misreading. A page-level reader sees the
sentence and gets it right ([`docs/DONE.md`](docs/DONE.md)).

**Structure comes from the reader, not from geometry.** The model emits
`### 2a)` headings, and assembly groups on them. A heading opens a new
question where it is written; a page that opens mid-answer continues
the open question.

**Never ask a vision model for pixel coordinates.** It was tried twice,
and of six boxes drawn back onto their pages one enclosed its figure.
The model reads far better than it measures, so a drawing's position
comes from words it read (its anchor), and its pixels come from
`assemble/segment.py`.

**Figures come from three inputs, each doing only what it is good at.**

| source | supplies | why not the others |
|---|---|---|
| the diagram pass | which pages have a drawing, and its caption | the only signal that knows a drawing *is* one |
| `anchor` text | where it goes in the answer | reading order, not a guessed number |
| `segment.py` regions | the pixels, when it has a region | precise edges, but blind to sparse drawings |

**Privacy is enforced at every stage, not once.** `page_01` is the
identity cover. It never leaves this machine: batches exclude it by
construction, the notebook refuses it on arrival, and `read_pages.py`
skips it before any reader sees a page.

**A lower CER is not automatically better.** If the reader starts
declining a table it used to invent, CER may barely move while the
output becomes far more trustworthy. Read the diff, not just the number.
A benchmark page must never become training data.

---

## Status

1,000 of 1,231 content pages were read (batch_04, students 46-61, was
deliberately cut). Those transcriptions exist only on the machine that
read them, so they need copying into `data/read/` before anything here
can run on real text. The open work, in order, is in
[`docs/TODO.md`](docs/TODO.md).
