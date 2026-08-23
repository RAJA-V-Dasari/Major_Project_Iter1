# Project plan

Turning scanned handwritten exam answer scripts into one clean Markdown
document per booklet, ready for marking. 61 students x up to 3 CIEs,
1,231 content pages of computer-networks answer booklets.

*State as of 2026-08-23.*

---

## 1. The pipeline

```
MP_Dataset/cleaned/              raw HF copy, normalised   [gitignored]
    |
    v
modules/01_prepare/              deskew -> crop -> tone
    |   01_deskew   Hough on printed rules -> page angle
    |   02_crop     one fixed size, anchored on the paper edge
    |   03_tone     flatten illumination, stretch ink to black
    v
modules/02_read/                 page image -> Markdown          [GPU]
    |   Qwen2.5-VL-7B, zero-shot, one .md per page
    |   question numbers, LaTeX maths, ![diagram](box), ![table](box)
    v
modules/03_assemble/             group by question number
    |   headings the reader already produced -> one answer per question
    v
modules/04_evaluate/             CER/WER against hand transcription
    |
    v
modules/05_pipeline/             one booklet in, one document out
```

`05_pipeline/run_booklet.py` is the entry point. Reading is the only
stage that wants a GPU, so it sits behind a swappable reader
(`cached` / `server` / `local`); everything else is plain Python and
runs on a laptop in seconds.

---

## 2. Where it stands

| Module | Status |
|---|---|
| `01_prepare` | **working** - deskew, crop, tone, all measured |
| `02_read` | **working** - 0.099 CER on 15 hand-transcribed pages |
| `03_assemble` | **working** - groups pages into per-question answers |
| `04_evaluate` | **working** - CER/WER harness plus 15 ground-truth pages |
| `05_pipeline` | **working** - one command per booklet |

### The measurement that decided the architecture

Scored on the same 15 pages, same scorer:

| engine | char-weighted CER |
|---|---|
| `trocr_lines` - segment into lines, TrOCR each | 0.463 |
| `qwen3b` - whole page, zero-shot | 0.229 |
| **`qwen7b`** - whole page, zero-shot | **0.099** |

Full detail in `modules/04_evaluate/RESULTS.md`.

---

## 3. What was removed, and why

The repo used to segment each page into line crops and recognise them
one at a time. Reading the whole page replaced all of it:

- **`02_segment`** - line/block geometry. The reader needs no crops.
- **`03_router`** - routed line crops to text or maths OCR. A page-level
  reader emits LaTeX inline; there is nothing to route.
- **`05_math`** - maths OCR over routed crops. Same reason.
- **`07_reconstruct`** - found question markers geometrically, then had
  to identify them. That stalled: TrOCR read 2 of 8 marks correctly, and
  a CNN over 576 hand-labelled crops reached 78.4% against a 71.2%
  majority baseline. The reader gets them right because it sees the mark
  in the context of the page, so the whole problem disappeared.
- **`annotation/`** - CVAT layout labelling feeding a YOLO detector,
  abandoned mid-run at epoch 49. The reader locates figures itself and
  returns a box for each.
- **TrOCR fine-tuning** (`04_ocr/finetune`, `render_numbered`,
  `build_finetune_set`) - measured at ~19 usable lines a page and ~57s
  a sample on this CPU, so ~1,000 lines would be ~53 pages transcribed
  and ~126 hours of training. Abandoned in favour of a zero-shot reader
  that needed neither.

All recoverable from git history.

---

## 4. Next

**Embed the figures.** The reader returns `![diagram](x1,y1,x2,y2)` in
page pixels. Crop those regions and reference them from the Markdown,
so a booklet document carries its own diagrams.

**One document per booklet.** `03_assemble` groups by question; the
remaining work is ordering and cleaning that into a single file per
booklet rather than a CSV of answers.

**Then evaluation** - keyword, then semantic similarity, then an LLM for
what those cannot settle. Waiting on real answer keys.

---

## 5. Known gaps

- **1,216 of 1,231 pages are unread.** `02_read/upload/` holds the five
  batches; the notebook resumes, so this is GPU time, not new work.
- **Tables are the weak spot.** The 3B fabricated a Dijkstra table
  outright. The prompt now offers per-cell `[?]` and a whole-table
  decline; whether the 7B takes it is unverified beyond 15 pages.
- **15 ground-truth pages is thin**, and only one carries a table.
- **No automated tests.** Verification is CLI flags and the CER harness.
- **Root `README.md` is stale** - it documents the removed pipeline.

---

## 6. Privacy

**Everything generated from the corpus is student work.** `page_01` of
every booklet is the identity block - name, USN, signature, marks - and
is excluded at every stage that touches the corpus, including a second
check inside the notebook. That exclusion is the basis on which pages
may go to a hosted GPU at all.

Gitignored: `input/`, `output/`, `upload/`, `predictions/`,
`ground_truth/`, `markers/`, `crops/`, `dataset/`, `.venv/`, `.env`.
Tracked: source, READMEs, page manifests, the benchmark selection.