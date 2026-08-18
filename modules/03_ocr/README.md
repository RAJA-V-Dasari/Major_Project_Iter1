# 03_ocr

> ## ⚠️ The output in `output/` is SIMULATED
>
> The text was **generated, not recognised**. No handwriting has been
> read yet. Every payload carries `"simulated": true`.
>
> **Do not** quote content from it as a student's answer, and **do not**
> measure OCR accuracy, spell-correction benefit, or anything else about
> recognition quality against it — it is domain-plausible prose from a
> seeded random generator, so it can only tell you what you put in.
>
> It exists so that everything downstream can be built now, against the
> real contract, without waiting for an engine to be chosen.

Pipeline position:

```
01_prepare -> 02_segment -> 03_ocr -> (downstream: answer extraction, ...)
```

## Layout

```
03_ocr/
    input/   -> symlink to 02_segment/crops   (33,577 line images + manifest.csv)
    src/
        schema.py     the contract: dataclasses + validate_run()
        simulate.py   fills the contract with synthetic text
    output/
        ocr.json          full nested payload (pages -> lines)
        lines.csv         same data flat, for joins and quick greps
        transcripts/      one .txt per booklet, in reading order
```

## Quick start

```python
import json, sys
sys.path.insert(0, "modules/03_ocr/src")
import schema

payload = json.load(open("modules/03_ocr/output/ocr.json"))

schema.validate_run(payload)        # raises with a specific reason if malformed

if payload["simulated"]:
    print("WARNING: synthetic text, not recognition results")

for page in payload["pages"]:
    for line in page["lines"]:      # already in reading order
        if line["status"] == "ok":
            print(line["line_uid"], line["confidence"], line["text"])
```

Or just `output/lines.csv` if you only need a table.

## The record

One `LineResult` per crop. `line_uid` is unique corpus-wide and is the
key to join on.

| field | meaning |
|---|---|
| `line_uid` | `s01_c1_p05_b00_l02` — unique across the corpus |
| `page_id`, `student`, `cie`, `page` | which booklet page |
| `block_id`, `line_id` | position within the page's block/line hierarchy |
| `reading_order` | 0-based within the page, no gaps (validated) |
| `bbox` | `[x1,y1,x2,y2]` in the **prepared page** frame (1598×2177) |
| `crop` | path relative to `02_segment/crops/` |
| `text` | Unicode, stripped. Empty unless `status == "ok"` |
| `confidence` | 0.0–1.0 for the whole line, not per character |
| `status` | see below |
| `tall` | segmentation flagged it as not-a-text-line |
| `extra` | engine-specific; treat as optional, never require a key |

`bbox` is the region's own extent, **not** the padded crop — the image
on disk is slightly larger on each side (see
`02_segment/src/crop_lines.py`).

### `status`

| value | meaning | handle it by |
|---|---|---|
| `ok` | text was read | use `text` |
| `empty` | the region held no readable text — a stray mark or smudge. **Not an error** | skip; the page really is blank there |
| `diagram` | `tall` region: a figure, brace or long division. A line recogniser should never be asked to read it | route to a diagram path, or ignore |
| `failed` | the engine errored on this crop | needs fixing — do not silently treat as blank |

`empty` and `failed` are deliberately distinct: one means the paper is
blank, the other means the pipeline is broken. Collapsing them hides
real faults.

## Current numbers

| | |
|---|---|
| Pages | 1,231 |
| Lines | 33,577 |
| `ok` / `empty` / `diagram` / `failed` | 32,472 / 646 / 348 / 111 |
| Booklets (transcripts) | 153 |
| Regions with no crop | 4,389 |

That last row is the reconciliation: 33,577 + 4,389 = 37,966, exactly
the regions `02_segment` found. Those 4,389 were too small to hold a
glyph (leftover rule fragments), so there is nothing for an engine to
read — they are recorded in `02_segment/crops/manifest.csv` with a
reason rather than dropped.

## What the simulation does and does not model

**Does** — because downstream code has to survive it:

- text length scaling with crop width (~25px/char, measured off real lines)
- a confidence distribution with a genuine low tail, not a flat 0.99
- all four statuses, so error paths are exercised now rather than discovered later
- diagram regions returning no text at all

**Does not** — because faking it would mislead:

- realistic OCR *error* patterns. The text is clean. Anything that
  depends on how recognition fails (post-correction, confidence
  thresholds tuned to real errors, accuracy figures) cannot be
  answered with this data.

Deterministic: each line is seeded from its own `line_uid`, so reruns
are byte-identical and two machines agree.

## Swapping in a real engine

Write a module that emits the same `LineResult` records and:

1. set `simulated: false` and `engine` to the real model id
2. keep `line_uid` derived the same way, so existing joins keep working
3. run `schema.validate_run()` before writing — `simulate.py` does, and
   it is what stops a malformed payload reaching consumers

Nothing downstream should need to change. If it does, the contract in
`schema.py` was wrong and should be fixed there, not worked around in
a consumer.

Engine still undecided — a trained line recogniser (TrOCR / PaddleOCR /
CRNN) and a vision-LLM call need different things from the crops
(the former wants a fixed input height, the latter does not), which is
why crops are emitted at native resolution and left unresized.
