# What is left

Ordered by what unblocks the most. Each item says what "done" means, so
it can be checked rather than argued about.

*As of 2026-08-23.*

---

## The goal this is all pointing at

**One clean Markdown document per booklet, with its diagrams embedded,
ready to be marked.** Everything below is a step toward that, and
evaluation begins after it.

```
page .md files  ->  crop and embed figures  ->  order and clean
                ->  ONE booklet.md          ->  evaluation
   [done]              [next]                    [after]
```

---

## P0 - reach the booklet document

### 1. Crop the figures and embed them
The reader already returns `![diagram](x1,y1,x2,y2)` in page pixels, and
those coordinates are verified against the page. Nothing consumes them
yet.

- crop each box out of the prepared page
- write it beside the booklet document
- rewrite the placeholder to point at the file

**Done when:** a booklet document opens in any Markdown viewer with its
diagrams visible in the right places.

**Watch for:** a box that is clipped or merged with a neighbour. Draw
the boxes back onto a few pages and look before trusting 1,231 of them.

### 2. Assemble one document per booklet
`03_assemble` groups answers into a CSV. The deliverable is a document.

- order by question, using the schema (1, 2a-2c, 3a/3b, 4a/4b)
- carry sub-parts (i, ii, iii / a, b, c) under their question
- stitch answers that span a page break
- strip the reader's artefacts: stray code fences, repeated headings,
  the page-marker comments once they have served their purpose

**Done when:** `run_booklet.py student_07/cie_2` writes one `.md` a
human would accept as that student's script.

### 3. Read the rest of the corpus
15 of 1,231 pages are read. The batches are packaged and the notebook
resumes; this is GPU time, not new work. Roughly 8-15 hours across
Colab sessions.

**Done when:** every booklet has a document, and the pages that failed
are listed rather than silently missing.

---

## P1 - trust the output

### 4. The reader never says "I cannot read this"
**This is the most important open risk.** Across 15 pages the 7B emitted
**zero** `[?]` and **zero** `![table]`, despite a prompt that asks for
both explicitly and despite the 3B having invented an entire Dijkstra
table on one of those pages.

A model that never declines is not a model that is always right. It is a
model whose errors are invisible - and for marking, an invented answer is
worse than a blank one.

Worth trying, cheapest first:
- check the 7B's tables against the pages by eye; if they are right, the
  risk is smaller than it looks
- make declining concrete: "if you are not certain of a cell, write [?]"
  as an instruction the model must follow per cell rather than a
  principle to weigh
- ask for a confidence line per page and see whether it separates good
  pages from bad

**Done when:** either the tables are verified correct on a sample, or
the reader marks what it cannot read.

### 5. More ground truth
15 pages is thin, and only one has a table - the exact case that fails.
Transcribe another 15-20, weighted toward tables and dense numeric
working.

**Done when:** the CER rests on 30+ pages with at least 5 carrying
tables.

### 6. Spot-check at corpus scale
CER on 15 pages says nothing about page 900. Once the corpus is read,
sample 20 pages at random, compare against the image, and confirm the
error rate has not drifted.

---

## P2 - evaluation

Blocked on **answer keys**, which do not exist yet. The ladder is
already agreed:

1. **keyword** - are the terms present? exact and explainable
2. **semantic similarity** - right answer, different words
3. **LLM** - only for what the first two cannot settle

Two things to decide before building it:

- **Per-CIE keys.** CIE 1, 2 and 3 are different papers, so `2a` names a
  different question in each. A previous attempt scored every booklet
  against one paper's key and looked plausible while being wrong.
- **Where the LLM runs.** These are real student answers. Either
  self-host that tier or de-identify before it leaves the machine.

---

## P3 - engineering

- **Root `README.md` is stale** - it documents the deleted pipeline.
  Reduce it to a pointer at `plan.md`.
- **No dependency manifest.** `cv2`, `numpy`, `torch`, `transformers`
  are imported and declared nowhere. One `pyproject.toml`.
- **No tests.** The natural first three: `ocr_bench.normalise`,
  `assemble.normalise_question`, and `readers.CachedReader` key lookup -
  the last of which had a bug that reported an empty booklet.
- **Local inference is unproven.** `LocalReader` is written but has
  never been run. If a laptop demo matters, try Qwen2.5-VL-3B at Q4
  (~2GB) and measure a page.

---

## Explicitly not doing

- **Fine-tuning TrOCR.** Costed: ~53 pages of transcription and ~126
  hours of CPU training, to beat a zero-shot reader that already works.
- **Reviving the layout detector.** The reader returns figure boxes
  itself.
- **Hand-transcribing the corpus.** Transcription is ground truth for
  measuring the reader, not a substitute for it.