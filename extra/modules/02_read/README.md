# 04_ocr

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
01_prepare -> 02_segment -> 03_router -> 04_ocr -> (answer extraction, ...)
```

04_ocr does not read the corpus directly. It reads what **Module 2
routed to the `ocr` processor** — equations go to `math_ocr`, figures to
`diagram_parser`, and this stage deliberately leaves them alone.

## Layout

```
04_ocr/
    input/   -> symlink to 02_segment/crops   (33,577 line images + manifest.csv)
    src/
        schema.py            the contract: dataclasses + validate_run()
        recognise.py         REAL recognition (TrOCR) -> ocr.json
        simulate.py          fills the contract with synthetic text
    output/
        ocr.json             full nested payload (pages -> lines)
        lines.csv            same data flat, for joins and quick greps
        transcripts/         one .txt per booklet, in reading order
```

`output/` holds generated artefacts only, and is gitignored by
`modules/*/output`. The routed input is **not** produced here — it is
read from `03_router/output/routed_regions.json`, the real router's
output. `input/` is the symlink to 02_segment's crops, which supplies
the images.

## Running it

```bash
cd modules/04_ocr/src

cd ../../03_router/src && python route.py   # the real router, 1.6s
cd -

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
sys.path.insert(0, "modules/04_ocr/src")
import schema

payload = json.load(open("modules/04_ocr/output/ocr.json"))

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

## Where the input comes from — `03_router`

`recognise.py` reads `03_router/output/routed_regions.json`: a **real**
routing decision over the whole corpus, not a simulation. Run
`03_router/src/route.py` first (1.6s).

Of 33,577 regions it routes 33,229 to `ocr` and 348 to
`diagram_parser`. **Nothing is routed to `math_ocr` before
recognition**, deliberately: separating equations from prose by
geometry was measured on 600 crops and rejected — ink density,
inter-symbol gap, component-size variation and aspect ratio are all
unimodal, so any threshold would be invented. Maths is identified
*after* recognition, from the text, via
`03_router/src/rules.py::reroute_by_content()`.

This replaced an earlier `simulate_router.py`, which invented a
`label` and `confidence` per region to drive the old label-based
`module2_router`. That router has been removed: it keyed on labels
this pipeline does not produce and dropped regions below a confidence
that does not exist. See `03_router/README.md`.
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
