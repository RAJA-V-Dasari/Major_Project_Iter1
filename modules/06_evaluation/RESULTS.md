# OCR benchmark results

Scored with `src/ocr_bench.py` over `bench_pages.json`, 4 of 15 pages
hand-transcribed. Markdown scaffolding is normalised away, diagram
placeholders excluded, and a short table of rendering equivalences
folded (see `EQUIVALENT`), so this measures reading and not formatting.

## Headline

| engine | char-weighted CER | neat | messy |
|---|---|---|---|
| `trocr_lines` — 02_segment + TrOCR base, per line | 0.573 | 0.455 | 0.898 |
| `qwen3b` — Qwen2.5-VL-3B, whole page, zero-shot | 0.141 | 0.080 | 0.318 |
| `qwen7b` — Qwen2.5-VL-7B 4-bit, revised prompt | **0.101** | 0.093 | 0.124 |

5.7x better than the line pipeline, with no training and no labels.

## Where 7B actually won

| page | content | 3B | 7B |
|---|---|---|---|
| s06_c1_p05 | prose | 0.063 | 0.089 |
| s01_c3_p10 | prose + long division | 0.097 | 0.097 |
| s03_c1_p03 | sparse prose | 0.109 | 0.062 |
| s10_c2_p10 | table + 5 diagrams | **0.527** | **0.124** |

Almost the whole gain is the one structured page. On plain prose 3B is
already at the ceiling and 7B is marginally worse on the best page.
**If the corpus were pure prose, the 3B would be the right model.** It
is not: tables and diagrams are common, and that is where the 3B
collapses.

## The table it used to invent

Ground truth row 1 is `2,A | 5,A | inf | inf`.

- 3B: `5 | 6 | 7 | 8` — a Dijkstra-shaped table of invented values
- 7B: `2 | 5 | ∞ | ∞` — correct numbers, predecessor labels dropped

Rows 2-4 are still wrong in 7B (`6 | 8 | 7` where the page says
`5,A | 6,B | 8,B`). So it is reading rather than confabulating the
shape, but it is still not reading the cells reliably.

## Two things the revised prompt did NOT fix

**It never declines.** `[?]` appears zero times across all 15 pages in
both models, including on cells it demonstrably got wrong. Both the
original and the strengthened anti-fabrication clause failed. A
confident wrong cell is indistinguishable from a right one in the
output, which is the failure mode that matters most for grading.

**Boxes are bands, not boxes.** The 7B did emit
`![diagram](x1,y1,x2,y2)` and every box is in valid page coordinates —
but all four are full page width and tile the page vertically
(0-806, 806-1588, 1588-2151). It is partitioning the page into strips,
not localising figures. Usable as coarse regions; not tight enough to
crop a figure without dragging in the text around it.

Asking for grounding inside a transcription prompt appears to be the
problem — Qwen2.5-VL grounds well when that is the whole request.
Options, cheapest first: keep the band as a hint and take the tight box
from `02_segment.find_grids`, which is geometric and already built; or
make a second grounding-only pass per page that has a figure.