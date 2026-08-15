# Module 1b (v2) — Structural Layout Analysis

Fresh approach to the same problem as `modules/scan_doc/seg`: classify
each region of a scanned handwritten answer page as one of
`paragraph`, `math`, `table`, `figure`, `crossed_out`.

**This module does not do OCR.** It does not even try to find precise
text boxes. It only needs to say, coarsely, "this area is probably X" —
that's the routing decision the downstream pipeline (Module 2) needs,
nothing finer.

---

## Why a v2, and what's different

`scan_doc/seg` took a detector-fusion route: docTR for text geometry, a
custom ink-diff pass for non-text, DocLayout-YOLO for printed tables,
merged into CVAT pre-annotations for a human to correct into training
data for a fine-tuned YOLO-seg model. That works, but it is heavy
(torch, transformers, a training loop, a labeling pass) for a task that
turns out to have a lot of *free structure* to exploit first.

**The structure worth exploiting:** these are ruled notebook pages with
a fixed line pitch and a printed margin. That gives a strong geometric
prior before any content analysis happens:

- The **margin line** anchors the left edge of normal text; a table's
  vertical divider or a figure's freeform ink crossing it is itself a
  classification signal, not just noise to crop out.
- The **rule pitch** (fixed spacing between horizontal lines) means text
  lines are *periodic* — a measurable, predictable signal — while a
  figure's ink is not periodic at all.

**The core technique: projection profiling** (classical document layout
analysis, sometimes paired with RLSA / XY-cut). Sum ink along rows and
columns to get 1-D histograms, then read structure off their shape:

| Signal | What it indicates |
|---|---|
| Regular horizontal peaks locked to rule pitch | `paragraph` |
| Regular horizontal peaks + regular *vertical* gaps (columns) | `table` |
| Horizontal peaks, but irregular pitch (fractions, superscripts) | `math` |
| No periodicity, arbitrary blob shapes | `figure` |
| A thin, dense, near-continuous horizontal stroke inside a line | `crossed_out` |

This is cheap (no GPU, no training loop, runs on the system's default
Python), interpretable (every classification traces to a specific
measurement, not a black-box score), and needs no labeled dataset to
start — thresholds are tuned against real pages, not learned.

**The honest tradeoff:** hand-tuned thresholds may plateau below what a
trained model reaches, especially on ambiguous figure-vs-math cases, and
they're specific to this ruled-paper format (a different notebook
ruling needs re-tuning). If this ceiling proves too low, the fallback is
the `scan_doc/seg` route — but that's a fallback, not the starting
assumption.

---

## Planned pipeline

```
        input/pages/*.png
                │
                ▼
      1. normalize_page.py
   detect margin + rule lines
   → deskew, get pitch + margin x-position
                │
                ▼
      2. segment_blocks.py
   horizontal projection profile
   (locked to measured pitch)
   → candidate region boxes
                │
                ▼
      3. classify_blocks.py
   per-box projection-profile features
   → periodicity, density, margin adherence
   → assign one of the 5 classes
                │
                ▼
      4. visualize.py
   colored boxes per class, for review
                │
                ▼
        output/regions.json
        output/annotated/*.png
```

Each stage is a separate script with its own inputs/outputs so any one
of them can be swapped out (e.g. classify_blocks.py replaced by a
trained classifier later) without touching the rest.

---

## Status

**Step 1 (`normalize_page.py`) — done and verified** on all 36 sample
pages:

- Pitch converges to **88–90 px on 33 of 36 pages** — the ruling really
  is uniform across all three booklets in the sample.
- The 3 pages that fail the ruling test (`pitch_strength` 0.17–0.21 vs
  0.45–0.67) are **exactly the 3 printed cover sheets** (pages 1, 9,
  23). Cover-page separation therefore comes free from this measurement,
  with no extra classifier.
- Margin line found on 29/36 pages; it sits at a different x per booklet
  (≈295 vs ≈500), so it must be measured per page, never hardcoded.
- Residual skew is small (±1.5°); the leftover distortion is page
  *curvature*, which rotation cannot fix — hence the profile-based
  methods below rather than line-fitting.

A first attempt detected rule lines directly with a straight
morphological kernel and found only 2 of ~25 rules on a curved page.
That approach was replaced by autocorrelation, which measures the whole
page's periodicity at once. Worth remembering before anyone
reintroduces line-fitting here.

**Step 2 (`segment_blocks.py`) — done and verified**: 841 lines and 277
blocks across the 36 pages. Blocks land on real answer units (a part
marker plus its prose, a worked calculation, etc.).

The decisive detail: **the printed ruling must be subtracted before
profiling**. With it left in, every ruled row carries ink, the profile
has no valleys, and a whole page collapses to one block (first attempt:
146 blocks, most of them page-sized). `split_horizontal_strokes()` in
`normalize_page.py` removes it.

That function deliberately *keeps* the short horizontal strokes it
separates out. A printed rule spans the page; a strikethrough is a short
dense horizontal stroke inside a line of writing. Same detector, and the
short-stroke output is the primary `crossed_out` feature for step 3.

Line count cross-checks well against the independent docTR measurement
from `scan_doc/seg` (841 vs 825), which is reassuring for two methods
that share no code.

**Steps 3-4 (`classify_blocks.py`, `visualize.py`) — built, and the
results are mixed.** Read this before trusting the output.

### What works

| Capability | Evidence |
|---|---|
| **Cover-page detection** | `pitch_strength` splits the 3 printed cover sheets from the 33 written pages with a wide gap (0.17-0.21 vs 0.45-0.67). No misses. |
| **Paragraph segmentation** | Blocks land on real answer units. 863 lines here vs 825 from docTR in `scan_doc/seg` - two unrelated methods agreeing. |
| **Table detection** | Grid crossings are decisive: 77 inside the cover sheet's table, 0-1 anywhere on a prose page. |
| **Line-level cancellation** | On page 5 the struck `= 16x16³ + 16x16²` is boxed exactly, and the prose around it stays `paragraph`. |

### What does not work yet

- **`math` vs `paragraph` is weak.** It rests on mean glyph-component
  width (maths 24-29 px, cursive prose 30-48 px) because cursive joins
  letters into wide blobs while maths is isolated symbols. The margin is
  a few pixels wide, and labels flip when block boundaries shift. This
  is the first thing to distrust.
- **`figure` produces false positives.** Ordinary prose on page 12 was
  labelled `figure`. And the sample contains no real diagrams, so the
  class has never been tested against a true positive - only against
  things it should have rejected.
- **Word-level cancellation is missed.** Page 12 has ~5 single struck
  words (`elean`, `sa`, `Sot`); one was caught. A word is too small a
  share of its line's ink to move the ratio.

### Design conclusion: `crossed_out` is a line-level thing

Judging cancellation per block was tried and abandoned. A student strikes
a phrase inside an otherwise-good answer, so a block-level verdict either
condemns the surrounding valid text or gets diluted into silence. Tuning
the threshold just traded one failure for the other:

- strict rule-width (0.25) removed a real strikethrough *as if it were
  printed ruling*, hiding the cancellation
- a block-span limit then rejected the same region for being too tall

Both went away once lines were tested individually and struck lines were
emitted as their own regions. The same argument now applies one level
further down - to reach single struck **words**, detection has to move to
the word level too.

### Recommended next move

Do not keep hand-tuning step 3. The split is clean:

- **Steps 1-2 (structure) are solid** and worth keeping. They need no
  training data and no GPU.
- **Step 3 (semantics) should become a small trained classifier** over
  the crops steps 1-2 already produce.

That turns the hard problem (detection over a full page) into an easy one
(classify a cropped region), which needs a far smaller model and far less
labelled data - and the labels can be produced by correcting this
pipeline's own output rather than annotating from scratch. It keeps the
"lightweight execution" goal intact: a small classifier over
classically-segmented regions is cheaper than a full detector.

---

## Setup

```bash
pip install -r requirements.txt
```

No venv pinning needed — this module has no compiled-wheel dependencies
tying it to a specific Python version, unlike `scan_doc/seg`.
