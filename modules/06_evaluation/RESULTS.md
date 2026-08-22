# OCR benchmark results

Scored with `src/ocr_bench.py` over `bench_pages.json`, 4 of 15 pages
hand-transcribed so far. Markdown scaffolding is normalised away and
diagram placeholders excluded, so this measures reading, not formatting.

## Headline

| engine | char-weighted CER | neat | messy |
|---|---|---|---|
| `trocr_lines` (02_segment + TrOCR base, per line) | 0.573 | 0.455 | 0.898 |
| `qwen3b` (Qwen2.5-VL-3B, whole page, zero-shot) | **0.141** | 0.080 | 0.318 |

A 4x reduction, with no training and no labels.

## The split is prose vs structure, not neat vs messy

| page | GT tables | GT diagrams | CER |
|---|---|---|---|
| s01_c3_p10 | 0 | 0 | 0.097 |
| s03_c1_p03 | 0 | 0 | 0.109 |
| s06_c1_p05 | 0 | 0 | 0.063 |
| s10_c2_p10 | 7 | 5 | **0.527** |

Every prose page lands near 0.1. The single page carrying a table and
five diagrams is 5x worse than the rest and drags the average up on its
own. The `neat`/`messy` buckets in `bench_pages.json` are keyed to
spurious-marker count, which turns out to track handwriting rather than
difficulty - `s03_c1_p03` is bucketed messy and scores 0.109.

## What the failures actually are

**The 3B model reads prose honestly.** On `s06_c1_p05` the errors are
misreadings, not invention: "is the first four" -> "in the first four",
"88 bits = 11 bytes" -> "1 byte". It correctly dropped both struck-out
spans without being asked twice.

**On the table it fabricated.** Ground truth row 1 is
`2,A | 5,A | inf | inf`; it emitted `5 | 6 | 7 | 8`. It recognised a
Dijkstra table and generated a plausibly-shaped one with invented
values, dropped the graph entirely, and gave 2 iterations where the page
has 4.

**It never declined once.** The prompt offers `[?]` for anything
unreadable and there are zero `[?]` marks across all four pages,
including the page where it invented a table. So the anti-fabrication
instruction did not take, and CER alone would not have revealed this -
0.527 reads as "poor recognition" until you look at the output and see
it is confident fiction.

## What this implies

Route tables and diagrams away from the recogniser rather than asking it
to transcribe them. `02_segment.find_grids` already locates hand-drawn
structure and was built for exactly this; on this evidence it earns its
place in an OCR-first pipeline rather than being deleted with the rest
of the line-cropping stack.

Worth testing next, cheaply: whether the 7B model fabricates the same
table, and whether a prompt that names tables explicitly as a decline
case ("if you cannot read every cell, emit ![table]") stops it.