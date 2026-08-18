# 02_segment

Finds where the content is on every prepared page, and cuts each region
out as its own image. **Geometry only — no labels, deliberately.**

```
02_segment/input/ -> ../01_prepare/03_tone/output
    |
    |  segment.py      page -> block -> line hierarchy
    v
output/segmentation.json, pages.csv, blocks.csv, lines.csv
annotated/            the same pages with boxes drawn, for a human
    |
    |  crop_lines.py   one PNG per region
    v
crops/ + crops/manifest.csv     consumed by 03_router and 04_ocr
```

```bash
cd modules/02_segment/src
python segment.py                  # every content page
python segment.py --limit 40       # a sample
python segment.py --no-images      # geometry only, much faster
python crop_lines.py               # cut the regions out
```

The `src/*.py` module docstrings carry the reasoning for every threshold
and are the place to look before changing one. This file is the stage-level
view: what it produces, how good it is, and what it refuses to do.

---

## 1. What it does

**Rules come out first.** The printed rules put ink on every ruled row;
left in, a horizontal projection has no valleys and the whole page smears
into one region. They are removed from the *mask* — the image on disk is
untouched — and told from handwriting by **length**, not darkness. Short
horizontal strokes are kept on purpose, because a strikethrough is exactly
a short dense horizontal stroke.

**Run-length smoothing, not projection profiling.** Profiling is blind to
anything not laid out in rows: measured on this corpus it put only 42–95%
of the handwriting inside a box, missing isolated question numbers and
every arrow and vertical stroke of a diagram. Smearing ink then taking
connected components means every stroke joins *some* region by
construction.

**Every threshold is a multiple of the measured rule pitch**, never a pixel
count. The previous generation of this code was tuned at 300 DPI and
silently mis-segmented — no error — when the corpus moved to 200 DPI.
Pitch is measured per page from the rules that were just removed.

**Regions are units of content, not connected components.** Lines are
assembled by snapping components to the printed rule their baseline rests
on; anything that did not snap is then absorbed into the line it belongs
to, and hand-drawn structure is merged into one region rather than sliced
into rows. §3 is about why.

**Cover pages are skipped.** `page_01` of every booklet is the printed
identity block — name, USN, signature, marks. Not answer content, and
pipelines that do not need identity data should not handle it.

---

## 2. How good it is

Scored by `modules/06_evaluation` against `annotation/`'s hand-drawn boxes,
mapped into this stage's coordinate space — 33 annotated content pages,
126 boxes. Read that module's README §2 first for what this ground truth
can and cannot support.

| Class | Boxes | Fragments per box | Ink recall |
|---|---|---|---|
| paragraph | 68 | 3.85 | 99.9% |
| math | 38 | 4.61 | 99.8% |
| figure | 13 | 3.23 | 99.9% |
| code | 5 | 3.80 | 99.7% |
| table | 1 | 1.00 | 100.0% |

**Ink coverage: 99.8% median** across the corpus — the number that says
nothing is being silently dropped, since ink no region covers is content
that never reaches OCR.

A paragraph box spans several written lines, so ~3.9 regions for one is
close to correct and not a number to drive to 1. Figures and tables are
single objects, and 1 is the right answer for them.

---

## 3. The regrouping fix

The stage used to put 99.8% of the ink inside a box and still hand the next
stage 30 boxes a page, because the pieces of one thing arrived as several
regions. Two causes, both measured:

**Descenders.** The line snap is a *baseline* test, and a descender does
not have the baseline of its own word. Over 50 pages, 390 components
failed the snap and **351 of them — 90% — were descenders hanging below a
line that snapped perfectly well.** Emitted alone they were 25% of every
region the stage produced: the loop of a `g`, the tail of a `y`, each
cropped as its own "line" and queued for OCR. A further 2.8% were
superscripts sitting *above* their line — worse than cosmetic, since the
superscript of `2^{m-1}` carries the meaning of the expression it was cut
from.

`absorb()` puts them back: a component in a line's band — ascender height
above the printed rule to descender depth below — that also sits
horizontally over writing which snapped to that rule is part of that line.
Both conditions are needed; the band alone would pull in a diagram label
level with a paragraph across the page.

**Drawn structure.** A table's cell borders lie along the printed rules, so
every row snapped like a line of writing and the table came out shredded —
one annotated table became 10 regions, and on cover sheets (which this
stage never processes) the marks grids reach 25–32.

`find_grids()` / `merge_grids()` find the strokes that draw a grid — long,
straight and thin, unlike any letter — and collapse what they enclose into
one region. Rule removal has already taken the *printed* horizontals, so a
horizontal stroke still this long was drawn by hand. Measured over the
annotated boxes: **13 of 14 tables contain 6 or more such strokes and 93%
contain at least two, against 1% of paragraphs** — 61 of 68 paragraphs
contain none at all.

Growth is capped vertically at 2.5 pitch past the strokes that started it.
Unbounded, the box crept upward a line at a time and took three lines of
ordinary prose into a diagram on `s57_c1_p07`; content pulled into a
drawing never reaches the recogniser, which is the same failure as
dropping it.

**Result: 30.0 → 21.8 regions per page, fragments per hand-drawn region
5.60 → 3.96, with ink coverage unchanged.** Coverage staying flat is the
point — this is regrouping, not discarding.

---

## 4. What it deliberately does not do

**It does not classify.** Calling a region "maths" or "figure" or "crossed
out" needs training data this corpus does not have — the labelling pass
found 27 figure, 10 code and 1 `crossed_out` examples in total. A wrong
label is a claim the rest of the pipeline has to un-learn. Where content
sits is a much easier question, and the one OCR actually needs answered.
`03_router` decides destinations, and does it *after* recognition, for the
same reason.

**Merging drawn structure is not classification**, and the distinction is
where the line is drawn. `merge_grids` never asks "is this a figure". It
asks "are these two pieces part of one drawn object", which a stroke
joining them answers locally. That it helps figures is a consequence, not
a claim about them.

**It cannot find a figure that has no straight strokes.** A free-hand
curve or sketch is missed, and no threshold here will catch it — every
classical signal separating a diagram from prose was measured and none
does (`06_evaluation/README.md` §5). That case needs a learned detector.

---

## 5. Known limits

- **Two-column layout would over-merge.** All components snapping to one
  rule become one region spanning min to max x. The corpus is
  single-column, so this has never fired; it would need splitting on a
  large horizontal gap if that changed.
- **Stacked fractions split.** `= d/V` written with the numerator on one
  rule and the denominator on the next is two regions, because they are
  two lines. Reassembling them is `05_math`'s problem, and it needs the
  coordinates this stage already emits.
- **Pitch is unreliable on sparse pages.** Where a student writes on
  alternate rules the measurement can lock onto the 2× harmonic. Every
  threshold here is in pitch units and inherits that error.
- **A page with no detectable rules** falls back to one region per
  component rather than inventing a grid.
- **`crossed_out` is invisible to this stage** and stays that way — a
  strikethrough is kept as ink, not recorded as an event. The one
  annotated example is not a basis for anything.

---

## 6. Reconciliation

`crop_lines.py` writes every region segment.py found to `manifest.csv`
either way, with the reason it was skipped if it was — so the counts
always reconcile and a mistake is visible rather than invisible. A region
too small to hold a glyph is recorded and not written; a 28×1px sliver of
leftover rule is not a line and an OCR queue full of them helps nobody.

Crops come out at **native resolution, greyscale, unresized**. Every
recogniser wants a different input height, and downscaling is the one step
that cannot be undone.
