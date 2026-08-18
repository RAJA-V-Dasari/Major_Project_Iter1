# 03_ocr

Two things can write `output/ocr.json`, and **you must check which one
did** before trusting a word of it. The `simulated` flag says so:

| producer | `simulated` | what the text is |
|---|---|---|
| `recognise.py` | `false` | **real** handwriting, read by TrOCR |
| `simulate.py` | `true` | **invented** prose from a seeded RNG |

> ### ⚠️ If `simulated: true`
>
> The text was **generated, not recognised**. **Do not** quote it as a
> student's answer, and **do not** measure OCR accuracy,
> spell-correction benefit, or anything else about recognition quality
> against it — it is domain-plausible prose from a seeded random
> generator, so it can only tell you what you put in.
>
> It exists so downstream work can be built against the real contract
> without waiting on an engine.

> ### ⚠️ If `simulated: false`
>
> The output is **real student handwriting** — personal data. `output/`
> is gitignored and must stay that way: never commit it, publish it, or
> paste it into an issue. The text is the engine's raw output and
> contains recognition errors, so it is evidence of what the model read,
> not of what the student wrote.

Pipeline position:

```
01_prepare -> 02_segment -> module2_router -> 03_ocr -> (answer extraction, ...)
```

03_ocr does not read the corpus directly. It reads what **Module 2
routed to the `ocr` processor** — equations go to `math_ocr`, figures to
`diagram_parser`, and this stage deliberately leaves them alone.

## Layout

```
03_ocr/
    input/   -> symlink to 02_segment/crops   (33,577 line images + manifest.csv)
    src/
        schema.py            the contract: dataclasses + validate_run()
        simulate_router.py   stands in for Module 2 -> routed_regions.json
        recognise.py         REAL recognition (TrOCR) -> ocr.json
        simulate.py          fills the contract with synthetic text
    output/
        router_input.json    what Module 2 was handed
        routed_regions.json  what Module 2 emitted; the input to recognise.py
        ocr.json             full nested payload (pages -> lines)
        lines.csv            same data flat, for joins and quick greps
        transcripts/         one .txt per booklet, in reading order
```

`output/` holds generated artefacts only, and is gitignored by
`modules/*/output`. The two routed files live there rather than under
`input/` because they are produced here, not received — `input/` is the
symlink to 02_segment's crops, which `simulate.py` also reads.

## Running it

```bash
cd modules/03_ocr/src

python simulate_router.py            # booklet s01_c1 -> routed_regions.json
python simulate_router.py --student 5 --cie 2

python recognise.py --limit 20       # a quick look
python recognise.py                  # every ocr-routed line in that booklet
```

`recognise.py` runs on CPU at roughly **4.4 s/line** (TrOCR base,
greedy, 8 threads), so a booklet is minutes and the full 33,577-line
corpus would be ~40 hours. Use `--limit` unless you mean it; a GPU or a
smaller model is the answer for corpus scale, not patience.

Note `microsoft/trocr-small-handwritten` is **not** a drop-in
alternative here: it ships a `sentencepiece.bpe.model` and would need
`sentencepiece` installed, which the base model does not.

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

## Current numbers — `simulate.py`, whole corpus

These are the *simulator's* figures (it runs over every crop directly,
not through the router). `recognise.py`'s own counts are printed at the
end of each run and stored in `ocr.json`.

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

## Where the input comes from — `simulate_router.py`

Module 2 has never been run against this corpus: its input is a
four-region hand-written sample, and the classifier that would label
real regions does not exist. `simulate_router.py` bridges that.

It is **not** a reimplementation. It builds Module 2's *input* from real
02_segment geometry, then imports and runs the actual `RegionRouter` —
the real `validate()`, `sort_regions()` and `route()` against the real
`ROUTING_TABLE`. Ordering, the discard threshold and the
label→processor mapping are therefore Module 2's behaviour, and they
change when it changes.

One field is synthetic: `label`, and the `confidence` attached to it.
Segmentation is geometry-only, so a region's class has to be guessed
from its shape until the classifier lands. A region typed `equation`
here is not really an equation — it is a line whose shape made the
heuristic say so.

**So: do not measure classification or routing accuracy against this.**
What it is good for is the plumbing — proving the recogniser takes only
what was routed to it and leaves math_ocr's and diagram_parser's
regions alone. A typical booklet routes to `ocr=134, math_ocr=12,
diagram_parser=2, manual_review=3, ignore=1`, with a few regions
dropped under the confidence threshold, so every branch is exercised.

## The real engine — `recognise.py`

`microsoft/trocr-base-handwritten`, greedy, on CPU. It emits the same
`LineResult` records as `simulate.py` with `simulated: false`, so
anything already built against the simulated output keeps working.

Two things worth knowing:

- **Confidence is measured**, not invented: `exp(mean log-prob)` over
  the tokens actually generated. Real — an unsure line scores lower —
  but it is a fluency measure, not a calibrated probability of
  correctness. Threshold and rank with it; don't read it as "87% right".
- **Text is raw.** TrOCR was trained on IAM, so it spaces punctuation
  (`" ."`, `" ,"`) and renders a circled question number as `O.O`.
  Normalising here would bake one engine's quirks into the contract;
  post-processing belongs in its own downstream stage where it can be
  measured.

`extra` carries the provenance — `router_label`, `router_processor`,
`router_confidence`, `router_reading_order`, `router_region_id` — so a
line can always be traced back to the routing decision that sent it here.

Identity (`line_uid`) is parsed out of `crop_path`, because Module 2's
`Region` has a `metadata` dict but never fills it. The naming contract
is `02_segment/src/crop_lines.py:crop_name()`; `recognise.py` raises
rather than guessing if a path doesn't match it.

### Writing another engine

Emit the same `LineResult` records and:

1. set `simulated: false` and `engine` to the real model id
2. keep `line_uid` derived the same way, so existing joins keep working
3. run `schema.validate_run()` before writing — both producers here do,
   and it is what stops a malformed payload reaching consumers

Nothing downstream should need to change. If it does, the contract in
`schema.py` was wrong and should be fixed there, not worked around in
a consumer.

Crops are emitted by 02_segment at native resolution and left
unresized, because a trained line recogniser and a vision-LLM want
different things from them.
