# Project plan

What this repo is, what is actually built, and what to build next.

Turning scanned handwritten exam answer scripts into readable, placeable
text. The corpus is 61 students × up to 3 CIEs, 1385 pages of
computer-networks answer booklets.

This file is the current source of truth for structure and direction.
The root `README.md` is **stale** — it documents a `preprocessing/`
pipeline that no longer exists on disk (see §5). Per-stage READMEs under
`modules/*/` are accurate and are the place for detail; this file does not
duplicate them.

*State as of 2026-08-18.*

---

## 1. The pipeline

```
dataset/                        raw HF copy, normalised   (gitignored, 6.9G)
    |
    |   ??  no ingestion step exists — see §5
    v
modules/01_prepare/
    01_deskew    Hough on printed rules -> page angle
    02_crop      one fixed size, anchored on the paper edge
    03_tone      flatten illumination, stretch ink to black
    v
modules/02_segment/             blocks -> lines, geometry only, no labels
    |                           33,577 line crops + manifest.csv
    v
modules/03_router/  (pass 1)    text_ocr | diagram        [math_ocr = 0, by design]
    v
modules/04_ocr/                 TrOCR line recognition -> ocr.json
    v
modules/03_router/  (pass 2)    reroute_by_content(): text with '=' -> math_ocr
    v
modules/05_math/                prepare -> engine -> LaTeX + page coordinates
    v
    ??  06_evaluation           DOES NOT EXIST — see §5, §6
```

Data moves through symlinks, not copies: each stage's `input/` points at
the previous stage's output. Verified on disk:

```
01_deskew/input     real directory, currently EMPTY
02_crop/input    -> ../01_deskew/output
03_tone/input    -> ../02_crop/output
02_segment/input -> ../01_prepare/03_tone/output
03_router/input  -> ../02_segment/crops
04_ocr/input     -> ../02_segment/crops
05_math/input/routed -> ../../03_router/output/after_ocr
```

**The router runs twice, and that is the one thing to understand here.**
Nothing is sent to `math_ocr` before recognition. Separating equations
from prose by looking at the crop was measured on 600 real regions and
rejected — ink density, gap/height, component-size CV and aspect ratio
are all unimodal, so any threshold would be an invented boundary. After
recognition the question is easy, because `=` is a character you can read
rather than a property inferred from ink statistics. Full argument in
[`modules/03_router/README.md`](modules/03_router/README.md).

Running alongside, not wired in: **`annotation/`** — a CVAT layout
labelling effort feeding a YOLO11n detector (§3).

---

## 2. Module status

| Module | Status | Where it stands |
|---|---|---|
| `01_prepare/01_deskew` | **working** | Hough over horizontally-opened ink; chosen over connected components and a rotation sweep after measuring all three. `output/` holds 62 entries from a prior run; `input/` is empty (§5). |
| `01_prepare/02_crop` | **working** | Every page to one size — the 5th-percentile paper size across the corpus — anchored per page off the detected edge. |
| `01_prepare/03_tone` | **working** | Divide by local background, then stretch. Removes bleed-through by exploiting sharpness, not brightness. Deliberately greyscale, never binarised. |
| `02_segment` | **working, stable** | Block/line hierarchy, 37,966 regions → 33,577 crops (4,389 too small to hold a glyph, recorded with a reason rather than dropped). Emits **no labels**, deliberately. No stage README. |
| `03_router` | **working, math rules provisional** | Both passes run; 1.6s for the corpus; reading order re-derived and cross-checked against `02_segment` (0 of 1,231 pages disagree). `config.MATH_RULES_ARE_PROVISIONAL = True` — thresholds validated against 12 hand-written cases, never against real OCR output. |
| `04_ocr` | **built; never run for real at scale** | `recognise.py` (TrOCR base, greedy, CPU) and `simulate.py` both satisfy `schema.py`'s contract. **The `ocr.json` currently on disk is `simulated: true`** (§4). A real full-corpus run is ~4.4 s/line ≈ 40 hours on this CPU. |
| `05_math` | **exploratory** | Plumbing, rule-erasure front-end, expression splitting and page-coordinate geometry all verified. Two engines measured head to head; neither is good enough to ship. `output/` is currently absent — not run in this tree. |
| `annotation/` | **labelled; training abandoned mid-run** | 112 of 120 sampled pages annotated, 409 boxes, 6 classes. A 150-epoch run stopped at **epoch 49** on 2026-08-15 (mAP50 0.478, mAP50-95 0.306 on 19 val images). Not consumed by any pipeline stage. |
| `06_evaluation` | **does not exist** | Nothing anywhere measures correctness. §6. |

### What the numbers in the READMEs are

Worth being blunt about, because it shapes the roadmap: every figure
quoted in a stage README today is a **throughput or reconciliation**
count — how many regions, how many seconds, whether the totals add up.
`33,577 + 4,389 = 37,966` proves nothing was silently dropped. It does
not say a single region is correct.

The two places accuracy has been looked at are both by eye and both
small: `05_math/README.md` §3 compares 19 expressions across two engines
against ground truth read off a review sheet, and `annotation/`'s YOLO
run reports mAP against 19 val images. Everything else is unmeasured.

---

## 3. The annotation track

`annotation/` is ground truth for **layout** — where the paragraphs,
maths, figures, tables, code and crossed-out spans are on a page.

- 120 pages sampled from 1385 with a fixed seed, **split by student, not
  by page**, fixed in the manifest before any measurement was taken.
- 112 pages annotated in CVAT and exported
  (`labels/instances_default.json`, and a cleaned variant).
- Pre-annotation emits geometry only, no classes. The old `scan_doc_v2`
  classifier was measured over all 1385 pages and rejected: it over-calls
  `table` ~3×, barely finds `figure`, and of 18 sampled `crossed_out`
  detections **zero** were genuine strikethroughs. Annotators anchor on
  whatever is already on screen, so a confident wrong label is worse than
  a blank one.
- Training is verified end to end but the real run is **incomplete** —
  stopped at epoch 49 of 150, three days ago, no process running.

Two loose ends here:

1. `evaluate.py` and `pseudo_label.py` default to
   `runs/layout/weights/best.pt`. The actual run wrote
   `runs/seg/weights/best.pt` (`name: seg`). The default path is wrong
   and evaluation will not find the weights without `--weights`.
2. **Nothing decides what this model is for.** `02_segment` emits
   geometry only, on purpose; a trained layout detector is exactly the
   thing that could supply the labels it refuses to invent, and could
   route tables and figures away before `05_math` ever sees them. That
   connection is unbuilt and undecided.

---

## 4. Read this before trusting any output on disk

**`modules/04_ocr/output/ocr.json` is `simulated: true`** — 1,231 pages
of text from a seeded random generator, not from a recogniser. The file
carries its own `warning` field saying so.

Everything derived from it inherits that:

- `03_router/output/after_ocr/` — 153 booklets of post-recognition
  routing decisions, made from synthetic text.
- therefore every `math_ocr` decision currently recorded is a decision
  about invented prose.

The simulator exists so downstream stages could be built against the real
contract without waiting 40 hours on a CPU, and it was the right call.
But it means **no accuracy claim can be made from anything currently in
the tree**, and the provisional maths thresholds have still never met a
real character. Check the `simulated` flag before quoting any of it.

---

## 5. Known gaps

**Documentation is actively misleading.** Root `README.md` documents
`convert_dataset.py`, `clean.py` and a `preprocessing/cleaner/` U-Net
under a `preprocessing/` directory. That directory does not exist.
`annotation/README.md` references `preprocessing/output/` throughout, for
the same reason. A new contributor following either one gets nowhere.

**There is no way to get data into the pipeline.** `01_deskew/input/` is
empty, and nothing in `modules/` writes to it. The step that populated it
was `convert_dataset.py` — normalising a mix of per-page PDFs, multi-page
PDFs and loose PNGs into `student_NN/cie_C/page_PP.png` — and it went
away with `preprocessing/`. `dataset/` already holds that normalised tree
locally, so the missing piece is small, but until it exists the pipeline
cannot be re-run from scratch by anyone.

**Nothing is measured.** See §6.

**No automated tests, anywhere.** Verification is ad hoc CLI flags —
`--preview`, `--verify`, `--check`, `--dry-run` — plus
`schema.validate_run()` and the router's reconciliation counts. All of it
requires a human to run it and look.

**No dependency manifest except one.** `modules/05_math/requirements.txt`
is the only one in the repo. `cv2`, `numpy`, `torch`, `transformers` and
`ultralytics` are imported across the other stages and declared nowhere.
Environment is `.venv` at the repo root: Python 3.11 via `uv`, CPU-only
torch (no GPU on this machine; system Python is 3.14, which has no torch
wheels).

**Inconsistent stage documentation.** `03_router`, `04_ocr` and `05_math`
have thorough READMEs. `01_prepare` and `02_segment` have module
docstrings only.

**Unpushed work.** Local `main` is 1 commit ahead of `origin/main`.

---

## 6. Evaluation — the missing module

There is no `06_evaluation`, and nothing else fills the role.
`annotation/evaluate.py` scores the YOLO layout detector against its own
val split; that is the only scoring code in the repo, and it measures a
model that no pipeline stage uses.

So: no measurement of segmentation quality, routing correctness, OCR
accuracy, maths accuracy, or end-to-end output. Every improvement
proposed in §7 is currently unfalsifiable — there is no number that would
move.

Proposed `modules/06_evaluation/`, in build order. Each reuses ground
truth that exists or is already planned, rather than opening a new
labelling effort:

**1. Layout and routing.** Score `02_segment` regions and `03_router`
decisions against `annotation/`'s 112 annotated pages, honouring the
manifest's student-level split. This ground truth already exists and is
currently used by nothing in the pipeline. Report region IoU/recall, and
routing precision/recall per destination — in particular, how much of
what a human called `math` the post-recognition reroute actually catches.

**2. OCR accuracy.** Needs a hand-transcribed set: a few hundred line
crops with their true text. CER and WER against `recognise.py` output,
and — separately — whether TrOCR's confidence ranks good readings above
bad ones, since triage depends on that and sumen's confidence provably
does not. **This is the same dataset the TrOCR fine-tune needs
(§7 P1.1).** Transcribe once, use twice.

**3. Maths accuracy.** Expression-level scoring over the same
transcriptions, turning `05_math/README.md` §3's by-eye 19-expression
comparison into something repeatable across engines and re-runnable after
a fine-tune.

**4. End-to-end reconciliation, as an assertion.** The counts each stage
prints today become checks that fail loudly. Cheap, and it turns the
existing reconciliation discipline into a regression test.

Build 1 first: it needs no new labelling and it scores two stages.

---

## 7. Roadmap

### P0 — make the repo runnable and honest

- Rewrite root `README.md` for the real `modules/` pipeline, or reduce it
  to a pointer at this file and the stage READMEs.
- Fix `annotation/README.md`'s `preprocessing/output/` references.
- **Build the ingestion step**: `dataset/` → `01_deskew/input/`. Recover
  the normalisation logic from `convert_dataset.py` in git history
  (pre-`preprocessing/` removal) rather than rewriting it.
- Fix `evaluate.py` / `pseudo_label.py`'s default weights path
  (`runs/layout` vs the actual `runs/seg`).

### P1 — measure, then improve

- **`06_evaluation` step 1** (layout + routing vs. existing annotations).
  Nothing else on this list can be judged without it.
- **A real `04_ocr` run.** Even a few booklets end to end with
  `simulated: false` would replace the synthetic chain in §4 and let the
  provisional maths thresholds meet real text for the first time.
- **Re-measure `MATH_NON_ALPHA_RATIO`** against that output. The known
  near-miss (`Sequence no. of 100th Packet = (100-1) mod(2^5)` at 42%
  against a 40% threshold) survives only because the `mod` rule catches
  it too.
- **Hand-transcribe the OCR ground truth** (§6 step 2) — the shared
  dependency of evaluation and the fine-tune.

### P1 — maths OCR (from `05_math/README.md` §6, unchanged)

1. **Fine-tune TrOCR on this handwriting, not a formula model.** The
   corpus is prose-with-numbers; TrOCR already reads whole lines nearly
   correctly and its one systematic failure — flattening superscripts
   (`2^{m-1}` → `2m-`) — is exactly what fine-tuning fixes. It also beat
   sumen ~10 of 19 head to head and runs 5× faster.
2. Sharpen maths detection against real OCR output (above).
3. Find a signal separating a good reading from a fabricated one. Sumen's
   confidence does not: it scored 0.94 on `\frac{\varepsilon\delta 9}{…}`
   for a crop reading `= 69`, against 0.96 for a perfect read. TrOCR's
   does separate. Round-tripping a reading back to an image would be
   stronger than either.
4. Route tables and diagrams away first — most surviving false positives
   are boxed table rows.

### P2 — decide the annotation track's purpose

Either finish it and wire it in — complete the 150-epoch run, label the
remaining 8 pages, bootstrap the other ~1265 pages with `pseudo_label.py`,
and use the detector to supply `02_segment`'s missing labels and route
tables/figures away before `05_math` — or shelve it explicitly and say so
in its README. Right now it is a substantial, half-finished effort that
no other stage depends on, and the repo does not say which it is meant to
be.

### P2 — engineering hygiene

- Root dependency manifest (`pyproject.toml`), consolidating what
  `modules/*/src` and `annotation/` actually import.
- Stage READMEs for `01_prepare` and `02_segment`.
- Automated tests. `04_ocr/src/schema.py::validate_run()`, the router's
  rule table, and the reconciliation counts are the natural first three;
  §6 step 4 turns the last into a real check.

### P3 — corpus scale

A full real `04_ocr` pass is ~40 hours on this CPU. Before attempting it,
decide between a GPU box and a smaller model — note
`trocr-small-handwritten` is **not** a drop-in (it needs `sentencepiece`,
which the base model does not).

---

## 8. Layout

```
Major_Project_Iter1/
    plan.md                 this file
    README.md               STALE — see §5
    DATASET.md              HF card for the cleaned stage
    dataset/                raw normalised corpus            [gitignored, 6.9G]
    .venv/                  py3.11 via uv, CPU-only torch    [gitignored]
    modules/
        01_prepare/
            01_deskew/      src/deskew.py
            02_crop/        src/{crop,grid,measure,cover_template}.py
            03_tone/        src/tone.py
            publish_dataset.py
        02_segment/         src/{segment,crop_lines}.py
        03_router/          README.md  src/{config,rules,route,reroute}.py
        04_ocr/             README.md  src/{schema,recognise,simulate}.py
        05_math/            README.md  requirements.txt
                            src/{prepare,math_ocr}.py  src/engines/{null,sumen,trocr}.py
    annotation/             README.md  LABELING_GUIDE.md  manifest.csv
                            sample/preannotate/validate/build/train/evaluate/pseudo_label
                            labels/                          [tracked — geometry only]
                            images/ dataset/ runs/           [gitignored]
```

Tracked: `src/`, READMEs, manifests, label geometry. Gitignored:
`input/`, `output/`, `preview/`, `crops/`, `annotated/`, `dataset/`,
`runs/`, `.venv/`, `.env`.

**Everything generated from the corpus is student work** — real names,
USNs, signatures and marks. It stays out of git, off hosted services, and
out of issues and pastes. The source lives in a private Hugging Face repo
and `HF_TOKEN` comes from `.env`. Keep it that way.
