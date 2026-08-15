# Labeling guide — handwritten answer script layout

Read this before drawing a single box. Consistency between annotators
matters more than any individual judgement call: a model trained on two
people's conflicting definitions learns the conflict.

**What this task is.** Mark *where each kind of content is*, so the
downstream pipeline can route each region to the right handler. It is
**not** OCR and **not** text detection. You are not outlining words,
lines or sentences. You are outlining **regions**.

**Corollary:** a box does not need to hug the ink. Getting the region
and its class right matters; a few millimetres of slack does not.

---

## The subject matter

Every booklet in this corpus is **Computer Networks (CIE, 5th sem)**.
That is why diagrams, binary arithmetic and protocol traces are so
common. If a future corpus covers a different subject, re-survey before
reusing these class definitions.

---

## Classes

Six classes. They were chosen after surveying the 120-page sample, not
picked in advance — each one below actually occurs, with a real example
named.

| Class | What it is | Roughly how often |
|---|---|---|
| `paragraph` | Ordinary prose, bullet and numbered lists | Nearly every page |
| `math` | Worked calculation | Very common |
| `figure` | Any drawing | ~1 page in 3 |
| `table` | Grid with ruled cells | ~1 page in 10 |
| `code` | Pseudocode, algorithms, protocol messages | Uncommon |
| `crossed_out` | Cancelled content — **overlapping layer, see below** | ~1 page in 3 |

### `paragraph`

Prose written along the printed rules, one thought per line, flowing
left to right at a steady line pitch. Numbered points (`1)`, `a.`) and
bulleted lists are still `paragraph` — the numbering is not a table.

One box per *contiguous run* of prose. If prose is interrupted by a
figure and then resumes, that is two `paragraph` boxes, not one box
swallowing the figure.

*Example:* `s18_c1_p09.png` — advantages/disadvantages list, whole page.

### `math`

Worked calculation: the thing being marked is the derivation, not the
sentence. Signs to look for — the line pitch stops matching the printed
rules, lines are short and drift right, `=` signs stack, and symbols
sit isolated rather than joining into cursive.

Includes long division (the CRC work is everywhere in this corpus),
unit conversions, hex/binary expansions, and boxed intermediate
results.

*Examples:* `s36_c3_p03.png` — CRC long division; `s55_c1_p07.png` —
hex-to-decimal port number expansion.

**Prose that merely mentions a number is not `math`.** "The size of
data = 88 - 8 = 80 bytes" written as one line inside an explanation
stays `paragraph`. Only break `math` out when the calculation occupies
its own visual block.

This is the least clear-cut boundary in the schema. When genuinely
torn, prefer `paragraph` — see *Ties* below.

### `figure`

Any drawing: network topologies, block/flow diagrams, sequence
diagrams, signal waveforms, weighted graphs, sketches.

**Figures usually contain text, and the text stays inside the figure.**
Node names, box captions, axis labels and arrow annotations are *part
of the figure*. Do not carve them out as `paragraph`. This is the single
most common way annotations go wrong on this dataset.

The box covers the whole drawing, **plus its caption or title** if one
sits directly under or over it.

*Examples:* `s17_c3_p08.png` — NRZ-I / Manchester waveforms;
`s34_c3_p06.png` — ARP diagram, boxes and arrows with labels
throughout; `s17_c2_p08.png` — weighted graph, nodes A–E.

**Boxed text is not automatically a figure.** A student boxing an answer
for emphasis (`s19_c3_p05.png`) is still `math` or `paragraph`. Ask
whether removing the box would destroy meaning — in a diagram it would,
in an emphasis box it would not.

### `table`

A grid: two or more columns separated by *ruled* dividers, with
corresponding rows. Hand-drawn counts.

**Mark the whole table as one region, including its caption/title and
any header row.** Internal structure — individual cells, rows, columns
— is explicitly *not* wanted here.

*Examples:* `s32_c2_p05.png` — hand-drawn "Routing Table of F";
`s16_c1_p07.png` — Go-back-N vs Selective-Repeat comparison; every
`[cover]` page — the printed marks table.

**Aligned columns without ruled dividers are not a table.** Values
listed in neat columns with no lines between them stay `math` or
`paragraph`.

### `code`

Pseudocode, algorithm listings, and verbatim protocol text. Signs —
indentation that is deliberate rather than incidental, control keywords
(`for`, `if`, `else`, `while`), bracket/brace pairs, array subscripts
like `D[y]`, or a literal request line such as `GET /usr/users/doc
HTTP/1.1`.

*Examples:* `s17_c2_p08.png` — Dijkstra listing beneath the graph;
`s03_c1_p03.png` — HTTP request and response lines.

`code` is uncommon. Do not stretch it to cover any maths that happens to
use a bracket.

### `crossed_out` — read this carefully

**`crossed_out` is a separate overlapping layer, not a sixth mutually
exclusive class.**

Students cancel a word or a phrase far more often than a whole answer,
so cancelled content nearly always sits *inside* a region that is
otherwise ordinary. So:

1. Label the region normally, ignoring the cancellation. A paragraph
   containing two struck words is still one `paragraph` box.
2. **Additionally** draw a `crossed_out` box around each cancelled
   span. It will overlap the region box. That is intended and correct.

Mark it at the granularity that was actually cancelled: a struck word
gets a word-sized box; a struck line gets a line-sized box; a page
cancelled with a large X gets one box over the whole cancelled area.

*Examples:* `s54_c1_p02.png` — several struck words mid-sentence;
`s58_c3_p03.png` — struck words in prose.

**Underlines are not strikethroughs.** The distinction is where the
stroke sits: a strikethrough passes *through* the letter bodies, an
underline sits *below* them. Emphasis underlining is common here — do
not label it.

---

## What NOT to label

- **Printed furniture** — the ruled lines, the margin rule, the printed
  page number, the coloured invigilator dot at the page foot. Not
  content.
- **Bleed-through** — faint mirrored writing showing from the reverse
  side. Visible on many pages (`s35_c3_p09.png`, `s34_c3_p06.png`).
  Ignore it completely; it is not on this page.
- **Blank space.** Large empty areas are common — part-used final pages,
  unattempted questions. Leave them unlabelled. A page with three
  written lines gets one small box, not a page-sized one.
- **Anything on a page you cannot interpret.** Flag it (below) instead
  of guessing.

---

## Cover pages

`page_kind = cover` in `manifest.csv`. These are a fixed printed form.
Label:

- the marks grid as one `table`
- the identity block (name / USN / department fields) as one `table` —
  it is a ruled grid and behaves like one

Do not label the college header, the printed instructions, or the
signature lines.

Covers are ~11% of the corpus but nearly identical to each other, which
is why only 15 are in the sample.

---

## Decision order

Work down this list and take the first that fits. It is ordered by how
reliable the evidence is, so an earlier match beats a later one.

1. Ruled grid with 2+ columns → **`table`**
2. A drawing → **`figure`** (and everything inside it stays inside it)
3. Indentation + control keywords → **`code`**
4. Its own block of worked calculation → **`math`**
5. Otherwise → **`paragraph`**

Then, as a second pass over the whole page, add `crossed_out` boxes
wherever content is cancelled.

### Ties

When two classes fit equally well, pick the one **later** in that list —
the more general one. Over-calling a rare class is worse than
under-calling it: `figure` and `table` are precise claims that a model
learns to distrust if the training data applies them loosely, whereas
`paragraph` is the honest default.

---

## Geometry

- **Axis-aligned rectangles.** Not polygons. The downstream consumer
  routes regions to handlers and does not need pixel-accurate outlines;
  polygons cost roughly three times the effort for a gain this pipeline
  will not use.
- Scans are slightly rotated and page-warped. A box around tilted
  content will include some slack — that is fine, do not attempt to
  compensate.
- Regions may overlap where content genuinely overlaps (most often a
  diagram drawn across a table, as in `s34_c2_p06.png`). Do not force a
  clean partition of the page.
- Minimum useful size: about one text line tall. Do not box individual
  words except for `crossed_out`.

---

## When you are unsure

Do not guess and move on. Add the tag `review` to the page (CVAT: page
level tag) and carry on. Ambiguous pages resolved inconsistently are
worse than ambiguous pages flagged for a second look.

---

## The split — do not annotate around it

`manifest.csv` assigns every page to `train`, `val` or `test`, **split
by student**, so no student's handwriting appears in more than one
split. Annotate all three identically and do not look at `test`
performance while tuning. The split is in the manifest so it is fixed
before any measurement is taken, which is the only point at which it
can be chosen honestly.
