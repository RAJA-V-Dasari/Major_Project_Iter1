# The examiner's marks — a reference, not a ground truth

`gold_marks.csv` holds the marks the faculty wrote on each booklet's
cover. It is the best external comparison this project has, and it is
**not an arbiter**. The examiner made mistakes: their arithmetic does
not always add up, and on four booklets they marked Part C in a way the
paper's own instructions do not allow.

So the right question at the end of this project is never "did we match
the examiner?" but "**where do we differ, and which of us is right?**" —
and sometimes the answer will be us.

The directory is still called `gold/` because renaming it is churn, but
treat the name as a filename, not a claim.

Read by: Claude, in session, from `data/marking/covers/*.png`
Checked by: `src/gold_check.py`
Status: **45 of 50 reconcile, 4 documented, 1 excluded — 49 usable**

---

## Why it is still worth having

Every cover carries two independently written sums — a TOTAL per row and
a grand total, repeated again in a separate box at the foot of the page.
A single misread digit breaks at least one of them.

That is what makes reading fifty grids affordable. Of the 7 rows the
checker rejected on the first pass, **2 were my transcription errors**:

| booklet | what the checker said | what it was |
|---|---|---|
| `student_03_cie_3` | q3 parts 6, total 8; grand total off by 2 | **my misread** — the total is 06 |
| `student_50_cie_3` | q2 parts sum 13, total 11 | **my misread** — q2b is 04, not 06 |

Two errors in ~400 cells, both caught mechanically rather than by
staring harder. The other five rejections were the examiner's, below.

---

## The marking policy for Part C

**Part C is answered by internal choice. The pair is worth the better
half — `max(3a, 3b)` and `max(4a, 4b)` — never the sum.** That is the
paper's instruction and it is what our grader implements.

Five booklets have both halves of a choice pair marked. On four of them
the examiner recorded something other than the better half:

| booklet | parts | policy | examiner | difference |
|---|---|---|---|---|
| `student_01_cie_2` | 10 + 7 | 10 | 10 | — agrees |
| `student_45_cie_1` | 6 + 1 | 6 | 7 | +1 |
| `student_22_cie_3` | 6 + 4 | 6 | 10 | +4 |
| `student_38_cie_3` | 4 + 2 | 4 | 6 | +2 |
| `student_27_cie_3` | 8 + 4 | 8 | **12** | +4, and over the 10-mark cap |

The examiner appears to have added both halves on these. On
`student_27_cie_3` that pushed the row to 12 marks on a question worth
10, which no reading of the paper permits.

**Our grader will disagree with the examiner on these four booklets by
design.** That disagreement must not be tuned away, and `agreement.py`
reports the comparison both with and without them.

### The Part C columns are not reliable anyway

`student_38_cie_3` records its Q3 marks in columns **b and c** — but
CIE-3's Q3 has only parts a and b. The examiner is using the grid's
columns loosely on Part C, probably to track the sub-parts inside 3b
rather than the choice halves.

So for Part C, compare at the **row** level (what the question scored),
not the column level (which part scored what). The column split is
trustworthy in Parts A and B, where the question structure matches the
grid.

---

## The four that do not reconcile

### `student_07_cie_3` — `accepted_no_totals`

Every part cell filled (q1a=4; q2 a=5 b=4 c=3; q3a=6; q4a=3); the entire
TOTAL column, the Total Mark Obtained box and the Marks Obtained box all
left blank. The marking was done, the adding up was not.

The parts sum to 25, but that is our number, not theirs, so `total_grid`
stays empty. Per-part agreement can use this booklet; booklet-total
agreement must skip it.

### `student_15_cie_2` — `accepted_edge_clipped`

The right-hand TOTAL column runs off the edge of the scanned page — a
defect in the scan, not the crop, and there is no other copy. q1 (02),
q2's parts (05, 03), q4 (10) and both grand totals (19) are legible;
q2's row total is not.

The legible parts imply 20. The examiner wrote 19, twice. One mark, on
one booklet, unresolvable from this image. Recorded, not guessed.

### `student_27_cie_3` — `accepted_over_cap`

q3 totalled 12 on a question worth 10 (see the choice table above). Not
a misread: the grand total of 31 is consistent with the 12 and the
Marks Obtained box repeats 31.

### `student_39_cie_3` — `accepted_examiner_sum`

Row totals read 02, 01, 02, 08 — summing to **13**. The examiner wrote
**14** in the grand total and again in the Marks Obtained box. All four
row totals are unambiguous at full resolution. A one-mark slip in the
student's favour, recorded as written.

---

## Excluded

### `student_26_cie_1` — `no_gold`

The grid is effectively unmarked: a faint `1` in q1a, a `4` in q2b,
`5+4` in q4b, no row totals, no grand total, no Maximum Marks, nothing
in the Marks Obtained box, no FIC signature. Too faint and too sparse to
attribute.

Blanked and **excluded from agreement entirely**. The comparison set is
therefore **49 booklets, not 50**, and every agreement figure is
reported over 49.

---

## What this data cannot tell you

- **The examiner is not an oracle.** Seven of the fifty covers carry a
  demonstrable defect — four choice-policy deviations, one arithmetic
  slip, one unfinished grid, one unmarked grid. That is a **14% defect
  rate on the thing we are comparing against**, and it sets a floor on
  how close any grader should be expected to get. A grader that matched
  this reference perfectly would be reproducing its errors.
- **Compensating errors survive the check.** Reading `3, 5` as `5, 3`
  preserves both sums. Nothing here catches that.
- **A blank cell is ambiguous** — unattempted, awarded nothing, or a
  cell the examiner skipped. The covers do not distinguish these.
- **Part C column splits are unreliable** (see above). Row totals only.
