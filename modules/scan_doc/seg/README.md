# Module 1b — Layout Segmentation (`seg`)

Segments scanned handwritten answer booklets into labelled regions
(text, equation, diagram, table, crossed-out, …) and hands them to
Module 2 for routing to OCR / Math-OCR / diagram parsing.

Input is page images from `ptoi`; output is `regions.json` plus crops.

---

## The core problem

Every pretrained document-layout model we tried is trained on **printed**
documents. Handwriting on ruled paper is far enough out of distribution
that their *labels* are useless, even when their *boxes* are not.

Measured on the 36-page sample booklet:

| Approach | Result | Verdict |
|---|---|---|
| **DocLayout-YOLO** (DocStructBench) | 439 regions, **67% labelled `figure`**; whole pages boxed as one figure | Boxes usable, labels not |
| **Grounding DINO** tiny & base (zero-shot, open-vocab) | 2–9 weak detections/page, scores 0.20–0.35, tokenizer-merged labels like `"table diagram mathematical equation"`, `"##written sentence"` | **Failed** |
| **docTR DBNet** (`db_resnet50`) | ~110–130 word boxes/page, near-perfect on cursive handwriting | **Works** |

The conclusion that drives this module's design:

> **Geometry is solvable off-the-shelf. Semantics is not.**
> Use docTR for *where* the content is; train a small model for *what* it is.

Grounding DINO was tried and rejected — kept in `experiments/` so the
negative result isn't re-litigated later.

---

## Pipeline

```
                  input/pages/*.png
                          │
     ┌────────────────────┼────────────────────┐
     │                    │                    │
detect_lines.py    detect_nontext.py      segment.py
 (docTR DBNet)     (classical CV)      (DocLayout-YOLO)
 words → lines     ink − rules − text   printed tables only
     │                    │                    │
     └────────────────────┼────────────────────┘
                          │  complete page coverage
                          ▼
         export_preannotations.py  ──► CVAT (COCO 1.0)
                │
                │   ⬅ HUMAN: assign a class to each pre-drawn box
                │
         corrected dataset
                │
                ▼
          train_yolo.py  ──► runs/answer_sheet_seg/weights/best.pt
                │
                ▼
           segment.py   (production inference, lightweight)
                │
                ▼
        export_module2.py  ──► Module 2 router
```

Three sources are needed because each is good at exactly one thing:

- **docTR only finds text.** Arrows in a worked long division, a box a
  student drew around a final answer, seals and scanner watermarks are
  all invisible to it.
- **`detect_nontext.py`** covers that residue: it isolates ink, subtracts
  the printed rules and everything docTR already claimed, and reports
  what is left. Tuned for **precision over recall** — a false box costs
  you a deletion in CVAT, a missed one costs a quick manual draw.
- **DocLayout-YOLO still earns its place for printed tables.** Its
  handwriting labels are worthless, but it found all 6 tables on the
  cover sheet at 0.65–0.96, and table grid lines are exactly what
  `detect_nontext.py` strips as rules. So it contributes `table` only.

Actual yield on the 36-page sample: **921 pre-annotations** = 825 text
lines + 79 non-text + 17 tables.

---

## Scripts

| Script | Purpose |
|---|---|
| `detect_lines.py` | docTR word detection → grouped lines (`--level word\|line\|block`) |
| `detect_nontext.py` | Non-text regions: diagrams, arrows, seals, drawn boxes |
| `export_preannotations.py` | Merge detector outputs → COCO JSON for CVAT import |
| `train_yolo.py` | Fine-tune `yolo11n-seg` on the corrected dataset |
| `segment.py` | Production inference; auto-selects fine-tuned weights |
| `export_module2.py` | Convert to Module 2's format + write crops |
| `visualize.py` | Draw `regions.json` boxes onto pages |
| `preview.py` | Scroll through annotated pages (`n`/`p`/`q`) |

---

## Setup

The system Python (3.14) has no compatible torch/torchvision wheels, so
this module pins **Python 3.11** in a local venv:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Then run everything with `.venv/bin/python`.

---

## Workflow

**1 — Detect (no training needed):**

```bash
.venv/bin/python detect_lines.py --visualize     # text lines
.venv/bin/python detect_nontext.py --visualize   # diagrams, arrows, seals
.venv/bin/python segment.py --baseline           # printed tables
```

Check `output/lines_annotated/` and `output/nontext_annotated/`.

**2 — Export pre-annotations** (merges all three; the `:table` suffix
takes only that label from the unreliable baseline):

```bash
.venv/bin/python export_preannotations.py
# equivalent to:
#   export_preannotations.py output/lines.json output/nontext.json \
#                            'output/regions.json:table'
```

**3 — Label in CVAT** (the only manual step):

- Create a task with the same page images from `input/pages/`.
- `Actions → Upload annotations → COCO 1.0` → `output/cvat_preannotations.json`.
- Boxes arrive pre-drawn. Assign each the correct class; delete false
  positives; draw the few regions both detectors missed.
- Every box's original detector guess is preserved in the
  `source_label` / `source_confidence` attributes.
- Export as **YOLO segmentation** format into `dataset/`.

**4 — Train:**

```bash
.venv/bin/python train_yolo.py
```

`yolo11n-seg` (nano) is the default — it is the lightweight model meant
for deployment. Move up to `yolo11s-seg` only if accuracy demands it.

**5 — Infer and hand off:**

```bash
.venv/bin/python segment.py
.venv/bin/python export_module2.py output/regions.json
cp output/segmentation_output.json ../../module2_router/input/
```

---

## Label set

Names match Module 2's `config.ROUTING_TABLE` so no translation layer is
needed:

| Label | Module 2 route |
|---|---|
| `text` | `ocr` |
| `question` | `ocr` |
| `equation` | `math_ocr` |
| `diagram` | `diagram_parser` |
| `table` | `table_parser` |
| `crossed_out` | `ignore` |
| `unknown` | `manual_review` |
| `code` | ⚠️ **not in ROUTING_TABLE yet** |
| `page_furniture` | ⚠️ **not in ROUTING_TABLE yet** |

---

## ⚠️ Open issues

**1. Module 1 → Module 2 format mismatch.**
`segment.py` writes nested `{"pages":[{"regions":[…]}]}`; Module 2's
`router.load()` reads a flat `{"regions":[…]}` where each region carries
its own `page` and `crop_path`. They could not be connected at all.
`export_module2.py` bridges this and the full chain is **verified
working** — 825 regions loaded, validated (0 rejected) and routed to
`ocr` with reading order assigned. But the two modules should agree on
one contract rather than depending on a converter.

**2. Two labels are unroutable.** `code` and `page_furniture` will fail
Module 2's `VALID_LABELS` check until added to `ROUTING_TABLE`
(`page_furniture` → `ignore`; `code` → its own parser or `ocr`).

**3. The dataset is one booklet.** `initial_dataset/CIE_Dataset_RV.pdf`
is 36 pages containing ~3 booklets from a single course, one
handwriting, two paper rulings (green and red). A model trained only on
this will not generalise. Collect more booklets across courses,
students and paper types before trusting any metric.

**4. No diagrams in the sample.** The booklet has prose, equations,
worked calculations and crossed-out text — but no real diagrams or code.
Those classes will have near-zero training examples until pages
containing them are sourced.

**5. `crossed_out` is the highest-stakes class.** Struck-through content
must never reach the grader. It is also visually subtle — a thin line
over otherwise normal handwriting. Worth over-sampling when labeling.
