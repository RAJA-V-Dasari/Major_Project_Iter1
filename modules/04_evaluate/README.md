# 06_evaluation

Scores pipeline stages against the ground truth that already exists.

Before this module, nothing in the repo measured correctness — every
figure in every stage README was a throughput or reconciliation count.
`plan.md` §6 asks for this first, because nothing else on the roadmap can
be judged without it.

Step 1 of that plan is built: **layout**. `annotation/`'s hand-drawn boxes
are moved into the coordinate space the pipeline actually runs in, and
`02_segment` is scored against them. Steps 2–4 (OCR accuracy, maths
accuracy, end-to-end reconciliation) are not built — they need a
hand-transcribed set that does not exist yet.

| File | What it does |
|---|---|
| `src/register.py` | Maps `annotation/` boxes into prepared-page coordinates |
| `src/score_layout.py` | Scores `02_segment`'s regions against them |

```bash
cd modules/06_evaluation/src
python register.py --check 6      # map the boxes, draw 6 pages to verify
python score_layout.py            # score the segmenter
python score_layout.py --baseline ../baseline_score.json
```

---

## 1. Why registration is needed at all

`annotation/` is the only ground truth in the repo and **no stage could
read it**, because it is in the wrong coordinate space.

The boxes were drawn on the raw pages (1700×2338). `01_prepare` then
rotated every page by its own measured angle and cropped it to 1598×2177
anchored on the detected paper edge. Both are per-page, so there is no
constant offset to subtract — each box is stale by a couple of degrees
and a few dozen pixels, by a different amount on every page.

`annotation/remap_annotations.py` did this for the *previous* generation
of the cleaning code, reading a `transforms.json` that stage wrote. The
current `01_prepare` does not write one and its output is a different
size, so `labels/instances_cleaned.json` is stale too — it is still
1700×2338. `register.py` recovers the transform instead of requiring it to
have been recorded.

**How.** Not by generic image registration — that was tried first and
fails. ECC over the ink masks converged to nonsense (aligned ink IoU
0.03–0.07): handwriting is sparse and high-frequency, with no gradient
basin for a gradient method to descend. Instead the pipeline's own two
operations are reproduced in order:

1. **Rotation**, re-derived with `01_prepare/01_deskew`'s own
   `rule_angle()` — imported, not reimplemented, so this cannot drift
   from what the stage did.
2. **Translation** — the crop — is then a pure 2-D shift, found by FFT
   cross-correlation of the two ink masks. A shift is the one thing
   correlation is reliable at, and with the rotation removed there is
   nothing else left to solve.

Measured over the 40 pages that have both a raw and a prepared copy on
disk: **aligned ink IoU median 0.941, minimum 0.666, none rejected.** The
ceiling is well under 1.0 by construction — `03_tone` changes stroke
weight, so identical geometry still leaves ink disagreeing at the edges.

A page registering below `MIN_INK_IOU` is **dropped, not warned about**.
Ground truth that is quietly 40px out is worse than none, because every
score computed from it looks plausible.

---

## 2. What the ground truth can and cannot support

Read this before quoting any number from here.

**The annotation is incomplete on purpose.** 409 boxes over 112 pages —
about 3.6 a page, where a page holds ~24 lines of writing. Annotators
boxed the regions the guide asks for, not every line. Run
`register.py --check` and look: most of the writing on any page has no box
on it.

That **rules out precision**. A segment region outside every annotated box
is not a false positive; it is almost always correct content nobody boxed.
Any "spurious region" count here would measure the annotation effort, not
the segmenter, and would look like a defect that cannot be fixed. So it is
not reported — only counted, and labelled as such.

What the annotation does support:

| Metric | What it answers | Why it survives incomplete annotation |
|---|---|---|
| **Fragments per region** | A human drew one box round one paragraph. How many pieces does the segmenter cut that area into? | Only ever looks *inside* a box that exists |
| **Ink recall** | Of the ink inside an annotated box, how much lands inside some region? | Same — confined to boxed areas |
| **Spill** | How far a region assigned to a box extends beyond it | Weak signal, see below |

Spill is reported honestly as weak: an unannotated neighbour is
indistinguishable here from a genuine over-merge, so it is expected to be
non-zero and only a large move in it means anything.

**Cover sheets are excluded** (`--covers` to include them). `02_segment`
skips `page_01` of every booklet, so scoring it measures code that does
not run. This matters more than it sounds: almost every annotated `table`
is on a cover, because the printed marks grid *is* a table. Leaving covers
in reported "tables shatter into 25 pieces" for pages the segmenter is
never asked about. Excluding them leaves exactly **one** annotated table
on a content page — which is the honest sample size for that class.

---

## 3. Result: 02_segment, before and after

33 annotated content pages, 126 boxes. `baseline_score.json` (tracked,
beside this file) is commit `18b4276`, before the regrouping work; the
second column is now.

| Class | Boxes | Fragments/box before | after | Ink recall |
|---|---|---|---|---|
| paragraph | 68 | 4.56 | **3.85** | 99.9% |
| math | 38 | 6.79 | **4.61** | 99.8% |
| figure | 13 | 8.31 | **3.23** | 99.9% |
| code | 5 | 4.00 | **3.80** | 99.7% |
| table | 1 | 10.00 | **1.00** | 100.0% |
| **overall** | 126 | **5.60** | **3.96** | |

Regions emitted: **30.0 → 21.8 per page.** Ink coverage unchanged at
99.8%, which is the point — this is regrouping, not discarding.

A paragraph box spans several written lines, so 3.85 fragments for one is
close to correct and not the number to drive to 1. Figures and tables are
single objects and 1 is the right answer for them.

---

## 4. Limits of this measurement

**Only 40 of 112 annotated pages can be scored.** Registration needs the
raw page as well as the prepared one, and only students 01–20 of the raw
corpus are on disk locally (535 of 1385 pages). Syncing the rest of the
raw tree would take this from 33 content pages / 126 boxes to ~105 / ~380
at no annotation cost — the single cheapest improvement available to this
module.

**13 figures and 1 table is a thin sample.** Enough to show an 8.3→3.2
move, not enough to tune against; thresholds swept in `02_segment` were
deliberately left at the knee of the curve rather than its minimum for
this reason.

**Splits are not honoured yet.** `--split` works, but with 6 val and 3
test pages registerable the numbers above are over everything. Once the
raw tree is complete, report `val` and keep `test` unopened.

**Nothing here scores the router, OCR, or maths.** Those are `plan.md` §6
steps 2–4 and all three need a hand-transcribed set that does not exist.

---

## 5. What this module says about figure detection

Worth recording, because it settles a question `plan.md` §P2 leaves open.

Every classical signal that might separate a hand-drawn figure from prose
was measured over the annotated boxes. **None separates them:**

| Signal | figure | paragraph |
|---|---|---|
| ink mass whose baseline snaps to a printed rule | 0.98 | 1.00 |
| ink mass in components taller than a line | 0.00 | 0.00 |
| ink density inside the box | 0.031 | 0.066 |
| long thin drawn strokes present | 15% of boxes | 1% of boxes |

Students draw diagrams on ruled paper and rest them on the rules exactly
as they rest their writing, so the geometry genuinely does not distinguish
the two. This is the same wall `annotation/README.md` records
`scan_doc_v2` hitting from the other side — it "barely finds figure",
predicting 4.3% where reality is nearer 1 page in 3.

`02_segment` therefore does not try to *classify* figures. It merges drawn
structure, which needs only the local question "are these two pieces part
of one drawn object" — and a stroke joining them answers that. That is
enough to take figures from 8.31 fragments to 3.23 without touching prose.

**A diagram with no straight strokes in it — a free-hand curve, a sketch —
is still missed, and no threshold will catch it.** That case needs a
learned detector, and it is the one place in this pipeline where more
labels are genuinely the answer.

---

## 6. Getting those labels without annotating 1,300 pages

The segmenter's geometry is 99.8% right, so a human does not need to
*draw* anything for a figure detector — only to *name* what is already
drawn. That makes the unit of work one keypress per region instead of
minutes per page in CVAT, and it means what you need is enough *regions*,
not enough pages.

The regions can also be pre-selected, and this module measures how well.
`02_segment` flags what it merged as drawn structure; those flagged
regions are **9.5× enriched for `figure`**:

| Annotated as | Share of grid-flagged regions | Share of the rest | Enrichment |
|---|---|---|---|
| figure | 41.4% | 4.3% | **9.5×** |
| math | 34.5% | 24.7% | 1.4× |
| paragraph | 6.9% | 38.4% | 0.2× |

Flagged regions are 4.0% of all regions. Labelling them first buys figure
examples an order of magnitude faster than sampling pages at random —
which is how the existing set ended up with 27 figures across 112 pages.

Roughly: 200 pages × ~22 regions is ~4,400 regions, the flagged ~180 of
them first. That is hours of keypresses over more pages than the current
112, and `annotation/pseudo_label.py` already exists to bootstrap the
remainder. No tool for this is built yet.
