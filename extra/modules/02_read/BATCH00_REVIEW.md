# Batch 00 review — all 250 pages

Read with Qwen2.5-VL-7B, 1024 patches, greedy, `max_new_tokens=1536`.
Students 01–10, CIEs 1–3, page_01 excluded throughout.

Every page was checked two ways that do not share assumptions:

- `check_batch.py` — structure of the Markdown alone
- `audit_pages.py` — the Markdown against **the ink on its source
  image**, which is the only way to see a page that came back short

Then pages from every failure class, plus a random sample of pages both
methods called clean, were compared against the scan by eye.

```bash
python modules/02_read/src/check_batch.py modules/02_read/output/batch00
python modules/02_read/src/audit_pages.py modules/02_read/output/batch00
python modules/02_read/src/render_boxes.py modules/02_read/output/batch00 --out DIR
```

## Headline

| | pages |
|---|---|
| read, none empty | 250 |
| structurally clean | 234 |
| looped and cut off at the token ceiling | 16 |
| less text than the ink supports (THIN) | 17 |
| more text than the ink supports (DENSE) | 7 |

Distinct pages implicated, after overlap: **31 of 250 (12.4%)**.

The other ~219 are good, and on clean prose and mathematics they are
very good. `s10_c2_p09` returned thirteen subnet addresses with two
digit slips; `s08_c2_p04` returned a seven-row routing table exactly.

## What is actually wrong

### 1. The reader never declines, and inventing is its fallback

This is the root cause, and four separate symptoms come out of it.

`![table]` boxes across all 250 pages: **zero**. Real `[?]` marks:
**9, on 3 pages**. When the reader cannot read something it does not
say so — it produces something anyway.

- **Empty-cell loops.** Nine pages emit `| | |` five hundred times
  until the token ceiling. The table it could not read became an empty
  table instead of `![table](box)`.
- **Invented structure.** `s08_c2_p07` holds Dijkstra working sets
  `{A,-,0} {B,A,2} {C,A,5} {E,B,8}`. The distance table below it came
  back correct; the working table came back as
  `| A | A2 | B |` — a three-column shape that is not on the page.
- **An invented source.** `s08_c2_p03` carries a network graph, a line
  of prose and **four complete routing tables, 28 legible rows**. The
  whole page came back as:

  ```markdown
  ### 2b)
  ![](https://example.com/image.png)
  ```

  That page is the clearest statement of the problem in the batch.
- **Prompt text as content.** `s07_c2_p07` transcribed
  `THE ONE RULE THAT MATTERS` from its own instructions. One page.

### 2. Figure boxes are guesses, and often missing entirely

Coordinates come back as round numbers in the model's own resized
space — `105, 210, 630, 840, 1268, 1479` are `50, 100, 300, 400, 609,
711` at 768x1046. That is a guess, not a measurement, and rendering
them onto the pages confirms it. Of six examined:

| page | box |
|---|---|
| `s07_c2_p09` | correct — page really is four graphs |
| `s04_c3_p03` | roughly right, clips the ARP table |
| `s02_c3_p03` | overlaps, clips the diagram's bottom half |
| `s06_c3_p06` | too large, swallows prose, cuts a text line |
| `s09_c2_p07` | too large, includes a table, clips second graph |
| `s01_c1_p04` | **wrong** — boxes prose, misses the diagram entirely |

**But the placement in reading order is right even when the
coordinates are wrong.** On `s01_c1_p04` the box sits immediately after
`* Eg:`, exactly where the handshake diagram belongs, while pointing at
the top of the page. The reader knows a figure exists and where it goes
in the text; it cannot locate it in pixels.

Worse, **17 pages flatten a diagram into label soup with no box at
all** — `s03_c1_p07` has two substantial diagrams and returns
`web client / Server 1 / Request / Response / host / DNS Client`. Of
roughly 39 pages carrying figures, **44% give the pipeline nothing to
crop.**

### 3. Enumerated lists become question headings

`s04_c1_p08` returns headings `0, 1, 2, 3, 4`; `s07_c2_p08` returns
`#### 2` for *"Shortest path from A to B"*. These are list items inside
one answer. Assembly groups on headings, so each one splits a single
answer into fragments filed under invented question numbers.

### 4. Plain lines are rendered as tables

`s10_c2_p09` is seventeen ordinary lines of subnet arithmetic. Every
one came back wrapped as `| ... |`. The content is right and the
structure is fiction.

### 5. Fences are two different things

157 pages are a whole-page ` ```markdown ` wrapper — scaffolding to
strip. **6 pages fence the student's own block content**, including
`s07_c2_p08`'s Dijkstra working. Stripping every fence deletes real
answers.

## What this changes

**The prompt is the fix for 1–4.** It says what to produce and never
what to do when the page cannot be read, so the model fills the gap by
inventing. A rule it can apply while decoding left to right — *if you
are about to write an empty cell, emit the box instead* — addresses the
loops, the invented tables and the fake URL together.

**P0.1 cannot trust box coordinates.** It should treat a box as *"a
figure belongs here in the reading order"* and find the actual crop
geometrically. That is what the deleted segmentation code was good at:
the original note on it was that it was *"great at geometry"* and bad
at identifying what it had found. This is the opposite half of the same
problem, and the two fit together.

## Re-read list

The 16 looped pages, plus `s08_c2_p03`, `s07_c2_p07`, `s08_c2_p07`,
`s07_c3_p07`, `s10_c2_p03`, `s03_c1_p06`, `s03_c3_p08` — **23 pages**.
