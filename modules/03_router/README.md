# 03_router

Decides which processor should read each segmented region.

```
01_prepare -> 02_segment -> 03_router -> { text_ocr | math_ocr | diagram }
```

## Layout

```
03_router/
    input/  -> symlink to 02_segment/crops   (manifest.csv + 33,577 line images)
    src/
        config.py   routes and thresholds (plain data, no side effects)
        rules.py    the decisions, each with the reason it fired
        route.py    load -> validate -> order -> route -> save, plus CLI
    output/
        routed.json   full payload, pages -> regions
        routed.csv    same flat, for joins
```

Run: `cd src && python route.py` (1.6s for the corpus).

## Current result

| route | regions |
|---|---|
| `text_ocr` | 33,229 |
| `diagram` | 348 |
| `math_ocr` | **0 — by design, see below** |

33,577 routed + 4,389 with no crop = 37,966, reconciling exactly with
what `02_segment` found.

## Read this before wiring up a maths engine

**Nothing is routed to `math_ocr` at this stage, and that is deliberate.**

The obvious design is to look at a crop and send equations one way and
prose the other. That was measured on 600 real crops and rejected.
Four candidate features are all **unimodal** across the corpus — there
is no second peak to cut between:

```
ink density        6  49 139 185 123  47  17  10   7   4   8   5
gap / height     592   5   1   0   0   1   0   0   0   0   0   1
component CV     176  19  27  64  66  68  49  56  26  28  13   8
aspect w/h       169  96  79 144  74  25   7   2   2   0   1   1
```

Any threshold there would be an invented boundary, and the corpus has
no labelled maths to validate one against. That is the same reason
`02_segment` does not classify regions at all: a wrong label is a claim
the rest of the pipeline then has to un-learn.

**So maths is identified after recognition, from the text**, where `=`
is directly observable rather than inferred from ink statistics:

```python
import sys; sys.path.insert(0, "modules/03_router/src")
import rules

routed = {"route": "text_ocr", "reason": "default"}

route, reason = rules.reroute_by_content(routed, "= 99 mod 32")
# -> ("math_ocr", "re-routed after recognition: contains the mathematical word 'mod'")
```

The intended flow is therefore:

1. `route.py` sends everything readable to `text_ocr`, diagrams aside
2. the text engine recognises it
3. `reroute_by_content()` moves the equations to `math_ocr`
4. the maths engine re-reads those crops (the crop path travels with
   the region, so nothing needs re-cropping)

### The maths rules are provisional

`config.MATH_RULES_ARE_PROVISIONAL = True`. A line is maths if it
carries an operator **and** is mostly non-alphabetic, or contains a
word like `mod`. Both conditions together, because prose mentions "="
occasionally and short labels are non-alphabetic without being
equations.

Verified against 12 hand-written cases, including the near-misses:

| text | verdict | why |
|---|---|---|
| `= 256-1` | maths | operator, 100% non-alphabetic |
| `N = 8` | maths | operator, 67% non-alphabetic |
| `Go-Back-N protocol is a protocol` | prose | has a hyphen but 93% letters |
| `the window size = 8 packets…` | prose | has `=` but 95% letters |

**One is close to the line:** `Sequence no. of 100th Packet = (100-1) mod(2^5)`
lands at 42% non-alphabetic against a 40% threshold. It is caught by
the `mod` rule too, so it survives either way — but the ratio alone is
not a comfortable margin, and `MATH_NON_ALPHA_RATIO` should be
re-measured against real OCR output rather than trusted.

## What routes on geometry

Only what segmentation already established:

- `tall` → `diagram`. A figure, brace or long division, not a row of
  writing; a line recogniser should not be handed it.
- everything else → `text_ocr`, the default. A handwriting recogniser
  reads a short `2b)` or a stray `=` perfectly well, so sending it
  there costs nothing while guessing costs accuracy.

`tags` (`short`, `full-line`) are descriptive only and never affect
routing — they exist so a consumer can batch small fragments
separately if it wants to.

## Reading order is re-derived, not trusted

Regions arrive already ordered by `02_segment`. `route.py` re-sorts by
`(y1, x1)` anyway and reports any page where the two disagree — a cheap
independent check on the upstream stage. **Currently 0 of 1,231 pages
disagree.**

## Why this replaced `module2_router`

The previous router keyed everything on a per-region `label`
(paragraph / equation / diagram / table) and dropped regions below a
confidence threshold. Neither exists in this pipeline: `02_segment`
emits geometry only and is deterministic, so there is no label and no
probability. Using it would have meant inventing both for all 33,577
regions.

What was kept is the shape — load → validate → order → route → save —
which was right. What was dropped: the label table, the confidence
filter, an `int` page id that cannot address 61 students × 3 CIEs, and
a `config.py` that created directories at import time.
