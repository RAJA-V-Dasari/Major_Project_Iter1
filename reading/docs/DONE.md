# What is done

Every number here was measured on this corpus, on this machine. Where a
claim rests on 15 pages rather than 1,231, it says so.

*As of 2026-08-23.* Written when this half lived under `modules/`. The
stage names below map onto today's folders: `01_prepare` is
`reading/prepare`, `02_read` is `reading/read`, `03_assemble` is
`reading/assemble`, `04_evaluate` is `reading/benchmark`, and
`05_pipeline` became `reading/read/read_pages.py` plus the root
`pipeline.py`.

---

## 1. The headline

**A page of handwritten answers becomes Markdown at 0.099 character
error rate, with no training and no labelled data.**

Scored on 15 hand-transcribed pages, same scorer, same pages:

| engine | char-weighted CER | WER |
|---|---|---|
| `trocr_lines` - segment into lines, TrOCR each | 0.463 | 0.796 |
| `qwen3b` - whole page, zero-shot | 0.229 | 0.447 |
| **`qwen7b`** - whole page, zero-shot | **0.099** | **0.281** |

4.7x better than the line pipeline this repo was originally built
around. Detail in [`reading/benchmark/RESULTS.md`](../benchmark/RESULTS.md).

---

## 2. What works, stage by stage

### 01_prepare (`reading/prepare`) - deskew, crop, tone
Unchanged and working. Rotation from the printed rule angle via Hough;
every page cropped to one size anchored on the detected paper edge;
illumination flattened by dividing by a local background estimate,
which removes bleed-through by exploiting sharpness rather than
brightness. Kept greyscale, never binarised.

### 02_read (`reading/read`) - page to Markdown
Qwen2.5-VL-7B in 4-bit on a free Colab T4, zero-shot. Emits question
numbers as headings, maths as inline LaTeX, struck-out text as `~~ ~~`,
and figures as `![diagram](x1,y1,x2,y2)` in original page pixels.

All 1,231 pages are packaged as five ~65MB batches ready to run, and
the notebook resumes after a dropped session.

### 03_assemble (`reading/assemble`) - group by question
Parses the headings the reader produced and groups pages into one
answer per question per booklet. Handles answers that span a page
break, and flags a page that opens mid-answer rather than guessing.

### 04_evaluate (`reading/benchmark`) - the measurement
CER/WER against hand transcription, with markdown scaffolding
normalised away so an engine is scored on reading rather than
formatting. 15 pages transcribed, stratified across neat, medium and
messy handwriting, covering all three CIEs.

### 05_pipeline (`pipeline.py`) - one command
`run_booklet.py <booklet>` runs the whole thing. Reading sits behind a
swappable reader so the pipeline does not change when the GPU does, and
`run.json` records which model produced every output.

---

## 3. The decisions, and what settled them

### Reading whole pages beat reading segmented lines
The original pipeline hit 99.8% ink coverage and still produced garbage,
because coverage measures whether ink landed in *a* box, not whether the
boxes were right. Measured on the neatest page in the benchmark, TrOCR
returned:

| on the page | TrOCR read |
|---|---|
| `a. The source port number is the first four` | `# Quinthouses through a number in` |
| `b. The destination port number is the next` | `Although the situation had not yet yet been` |
| `is directed from client to server.` | `in directed from Grosses , to seven` |

Not misreading - **fabrication**. A page-level reader sees the sentence
and gets it right.

### Question numbers solved themselves
Identifying `4)a)` from a 40x40 crop defeated two approaches: TrOCR read
2 of 8 correctly, and a CNN trained on 576 hand-labelled crops reached
78.4% against a 71.2% majority baseline. The page-level reader gets them
right because it sees the mark in context - after "Part - C", at the
start of a line. The whole sub-problem disappeared.

### Segmentation over-merging was fixed, then made moot
An earlier pass cut region fragmentation per hand-drawn box from 5.60 to
3.96 and regions per page from 30.0 to 21.8, with ink coverage flat at
99.8%. Real work, correctly measured - and then superseded, because the
reader needs no crops at all. Recorded here because the finding stands
even though the code is gone.

### Fine-tuning TrOCR was costed and rejected
~19 usable lines per page, so ~1,000 in-domain lines is ~53 pages of
hand transcription. Training measured at ~57s per sample on this CPU
with the encoder frozen: ~126 hours. A zero-shot reader needed neither.

---

## 4. Bugs found and fixed

Each of these was silent, and each would have corrupted a result.

**The Colab OOM.** `min_pixels`/`max_pixels` were set inside the chat
message while the raw image went to `processor(images=...)`, which
ignores it. No resize happened, a page became ~4,437 visual tokens, and
attention asked a T4 for 18.85 GiB. The budget now sits on the
processor and `process_vision_info` does the resize - ~1,024 tokens,
5.3% of the memory.

**Box coordinates in the wrong space.** The model returns boxes in the
*resized* image; they are rescaled to original page pixels before being
written, or nothing downstream could crop with them.

**A benchmark page in the training set.** `build_finetune_set` refused
it - the first page transcribed for training was a benchmark page, and
using it would have quietly inflated the CER it was measured against.

**Cache keyed on the wrong name.** `CachedReader` looked up `page_03`
while every manifest calls that page `s01_c2_p03`. It matched nothing
and reported an empty booklet.

**322MB of student pages staged for commit.** `modules/*/upload` was
missing from `.gitignore`. Caught before committing.

**Two files corrupted during cleanup.** PowerShell flattens a
single-element array, so a `[[old, new]]` replacement table degraded to
a string and `Replace(pair[0], pair[1])` became `Replace('p','a')`.
Caught by a compile check, restored from git, re-patched in Python.

---

## 5. What was deleted

Reading the whole page replaced six modules. All recoverable from git
history.

| removed | why |
|---|---|
| `02_segment` | line/block geometry - no crops needed |
| `03_router` | routed crops to text vs maths - nothing to route |
| `05_math` | maths OCR over crops - reader emits LaTeX inline |
| `07_reconstruct` | geometric question markers - reader reads them |
| `annotation/` | CVAT to YOLO layout labelling, abandoned at epoch 49 |
| TrOCR fine-tune | costed above; not worth 53 pages and 126 hours |

23 tracked files remain, down from about 70.

---

## 6. What these numbers do NOT cover

- **15 pages, not 1,231.** The CER is a sample. Only one of those pages
  carries a table.
- **0.099 is the 7B on a rented T4**, not anything this laptop can run.
- **The reader never declines.** Across all 15 pages it emitted zero
  `[?]` and zero `![table]`, including on a page where the 3B invented a
  whole Dijkstra table. See [TODO.md](TODO.md) - this is the open risk.