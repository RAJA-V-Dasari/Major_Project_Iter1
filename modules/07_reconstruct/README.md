# 07_reconstruct

One PNG per booklet, cut at question boundaries — for a human to
review, not for the OCR pipeline.

```
07_reconstruct/input/          (deskewed, cropped, toned pages)
07_reconstruct/segmentation/   (02_segment's geometry - pitch only)
    -> 07_reconstruct/output/student_NN/cie_C.png
    -> 07_reconstruct/output/chunks.csv
```

Run:

```bash
python src/reconstruct.py --student 1            # one student, all CIEs
python src/reconstruct.py --student 1 --cie 1     # one booklet
python src/reconstruct.py                         # whole corpus
```

## Why this exists

`02_segment`'s line and block regions are built for OCR: every crop
needs a baseline, so a diagram that crosses several rule rows still
gets cut into per-rule-row fragments internally (see
`02_segment/README.md`'s note on regrouping — that fixed table grids
and descenders, not free-form diagrams). Reviewing a booklet by paging
through those fragments does not work; a client-server handshake
diagram spread across seven line crops looks like nothing.

Reviewing one page at a time does not work either, because an answer
routinely spans a page break, and Part A / Part B / Part C are not the
same length as a page.

So this module never touches line or block geometry for the crop
itself. It finds where each **question starts**, and crops the raw
Y-band between one start and the next straight off the source page —
whatever is in that band, diagram included, comes out whole because it
was never segmented into pieces to begin with.

## How a question start is found

Checked page by page against `student_01/cie_1` — 6 pages, cross-
referenced by eye. The one reliable signal is where the student writes
it: **a mark whose box crosses the printed left margin**, the same rule
`02_segment` already removes as `_vertical_rules`. "①", "(2c)", "(3b)"
all do this. A sub-part continuing the question already open —
"ⓑ" for the second part of a question first marked "(2a)" — does not;
it is written just to the right of the margin, same as the prose
around it. A numbered list *inside* an answer ("① client, ② server")
does not either, for the same reason. No OCR is involved — this is
pure geometry, and it is what separates "new question" from
"formatting inside the current one" without reading a single word.

## The cut is the gap above the mark, not the mark itself

A student writes the circle beside whatever line it lands next to,
which is routinely a line or two below the actual start of the
question — measured on `student_01`: "(2a)" sits 2.6 pitch below the
blank line before "PART-B", "①" sits 3.0 pitch below the one before
"PART-A". Cutting at the mark's own y put both headings on the wrong
side. `gap_above_marker` walks up from the mark to the nearest run of
blank rows and cuts there instead — the blank line is what the student
actually used to separate the two answers, the mark is just where the
pen happened to be.

Bounded to `GAP_MAX_SEARCH_PITCH` (4 pitch), because "nearest blank
band" breaks the moment the content above the mark is a diagram rather
than prose. `student_01/cie_3/page_04`'s "(2b)" sits directly under an
ARP diagram with no blank-line convention anywhere in it; searching
unbounded walked 13 pitch back up through the whole diagram before
finding blank paper above its header row. Past the bound, a mark keeps
its own position rather than jumping somewhere clearly wrong.

## Two things that look like a marker and are not

Both were caught by running against real pages, not assumed:

**A hand-drawn box hard against the margin.** `student_01/cie_1`'s
request/response tables sit flush against the rule; their own border
can bleed a 1-2px sliver past the cutoff column. Left alone, closing
that sliver into a marker nearby inflated it past a real mark's height
by 4×. Fixed by opening the gutter mask with a small kernel before the
merge that joins a circle to its digit — a real stroke survives; a
1px sliver does not.

**Bleed-through from the facing page.** `student_01/cie_3/page_02` has
a strip of the previous page's text down its extreme left edge — the
known binding-seam residue `DATASET.md` documents for `02_crop`. Every
one of those text fragments passed every other check and produced 11
spurious markers on one page. What told them apart from real ones:
every confirmed marker on this booklet sits within a third of a pitch
of the margin's right edge; the bleed-through sits 3+ pitches away, at
the physical page edge. `MARKER_MAX_GAP_PITCH` is that boundary.

## What is NOT solved

- **Margin position is per page, not per booklet.** Each page is
  deskewed and cropped independently, anchored off its own detected
  paper edge (`01_prepare/02_crop`), so the margin's pixel x can move
  by 100px+ between consecutive pages of the same booklet — confirmed,
  not assumed, on `student_01/cie_1` (x = 123, 251, 299, 172, 289, 187
  across its six pages). Detection runs fresh per page for this
  reason; a page with no detectable candidate borrows the booklet's
  median over the pages that do.
- **Validated on one student, three booklets.** The marker convention
  — circle a compound "2c", leave a bare letter uncircled or
  uncrossing — is this student's habit, not a rule every student in a
  61-student corpus is guaranteed to follow. Before running corpus-
  wide, spot-check a handful of other students' booklets the same way
  this one was checked: by eye, page by page.
- **A missed marker merges two questions rather than losing content.**
  If a student's mark does not cross the margin clearly enough to
  detect, the chunk it should have started keeps growing as part of
  the question before it. Nothing is dropped — the content is still in
  the PNG — but the cut line is in the wrong place.
- **No text labels.** A chunk's own marker, still visible at its top
  because the crop starts exactly there, is the only indication of
  which question it is. Reading it still takes a human eye; nothing
  here does OCR.
