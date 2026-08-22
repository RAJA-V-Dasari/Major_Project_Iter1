# OCR benchmark results

All 15 pages of `bench_pages.json` hand-transcribed. Scored with
`src/ocr_bench.py`: Markdown scaffolding normalised away, diagram
placeholders excluded, and a short table of rendering equivalences
folded (`EQUIVALENT`), so this measures reading and not formatting.

## Headline (15 pages)

| engine | char-weighted CER | overall CER | WER |
|---|---|---|---|
| `qwen3b` — Qwen2.5-VL-3B zero-shot | 0.229 | 0.250 | 0.447 |
| `qwen7b` — Qwen2.5-VL-7B 4-bit, revised prompt | **0.099** | **0.122** | **0.281** |

(`trocr_lines`, the 02_segment + TrOCR-base pipeline, scored 0.573 on
the first 4 pages; a 15-page re-run is in progress.)

## The 4-page sample was flattering the 3B

Measured on the first 4 pages only, the two models looked close - 0.141
against 0.101. On all 15 they are not:

| sample | qwen3b | qwen7b | gap |
|---|---|---|---|
| first 4 pages | 0.141 | 0.101 | 1.4x |
| all 15 pages | 0.229 | 0.099 | **2.3x** |

The 7B's number barely moved (0.101 -> 0.099) while the 3B's got 60%
worse. Four pages was too few to rank two models, and would have been
enough to pick the wrong one on cost grounds.

## The difficulty buckets are meaningless

| bucket | qwen3b | qwen7b |
|---|---|---|
| neat | 0.176 | 0.131 |
| medium | 0.247 | 0.136 |
| messy | 0.326 | **0.099** |

For the 7B, "messy" is its BEST bucket. The buckets are keyed to
spurious-marker count from `07_reconstruct`, which tracks how much junk
sits in the margin - a property of the handwriting, not of what makes a
page hard to read. What actually predicts difficulty is dense numeric
content, and that cuts across all three buckets.

## Where the 7B still fails

| page | CER | what is on it |
|---|---|---|
| s12_c2_p06 | 0.521 | four routing tables, ~56 numeric cells |
| s51_c3_p04 | 0.260 | two waveform diagrams |
| s29_c3_p08 | 0.159 | binary checksum working |

One page carries the failure. s12_c2_p06 is four 7-row routing tables
of two-digit numbers and nothing else; at 219 characters of ground
truth it is also the shortest page in the set, so its errors weigh
heavily per character. Every one of the 7B's worst pages is dense
numeric working, and its best are prose.

## Still unfixed

**Neither model ever declines.** `[?]` appears zero times across all 15
pages in both, including on cells demonstrably read wrong. Two rounds
of prompt strengthening changed nothing.

**Boxes are bands.** The 7B emits `![diagram](x1,y1,x2,y2)` in valid
page coordinates, but they span the full page width and tile it
vertically rather than enclosing figures. Usable as coarse regions, not
tight enough to crop from.