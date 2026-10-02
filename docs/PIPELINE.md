# The pipeline, stage by stage

What to run, in what order, what each stage reads and writes, and what
re-running one does to the others. Every command runs from the repo root.
Every stage is one script that runs on its own; `python pipeline.py
<stage>` runs it and passes any options through.

```
 HF cleaned ──fetch──> data/pages/ ──read [GPU]──> data/read/<engine>/
 (HF raw ──prepare──> data/pages/, only to rebuild)        │
                                                           │  figures [GPU, optional]
                                                           v  -> data/figures/diagrams.json
                                                   assemble
                                                           │  -> data/booklets/
                                                           v
                                                   handoff
                                                           │  -> data/handoff/   (docs/HANDOFF.md)
                                                           v
                                                   mark
                                                           │  -> data/marking/
                                                           v
                                                   review  (http://127.0.0.1:8000)
```

**Only two steps need a GPU:** reading pages, and the optional diagram
pass. Both run elsewhere (Colab, Kaggle or a server) and drop files into
`data/`. Everything else is plain Python on a laptop, and
`python pipeline.py run` does all of it in order.

---

## 0. Setup, once

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows; source .venv/bin/activate elsewhere
pip install -e .                   # numpy, opencv, pillow, pymupdf, huggingface_hub
pip install -e ".[semantic]"       # optional: the embedding tier (~2 GB with torch)
copy .env.example .env             # then put HF_TOKEN=... in it
python pipeline.py check           # every check that needs no data, no model, no GPU
python pipeline.py status          # what exists, and what to run next
```

Python 3.11 or newer. `.env` is read by `common/layout.py`, so every
script sees the same settings ([`DATA.md`](DATA.md) lists them).

---

## 1. fetch: the prepared pages

```bash
python pipeline.py fetch                          # data/pages/, covers skipped
python pipeline.py fetch --students 1-5 --cie 2   # a subset
```

**Reads** the `cleaned` Hugging Face repo. **Writes** `data/pages/`. See
[`DATA.md`](DATA.md).

### prepare: only to rebuild those pages from the raw scans

```bash
python pipeline.py fetch --repo raw               # data/raw/
python pipeline.py prepare                        # deskew -> crop -> tone -> data/pages/
python pipeline.py prepare --preview              # before/after pairs only, in data/prepare/preview/
```

Three scripts in `reading/prepare/`. Deskew finds the page angle from the
printed rules (Hough). Crop cuts one fixed 1598×2177 sheet anchored on
the paper edge. Tone divides by a local background estimate, which
removes bleed-through by its blur rather than its brightness.
Intermediates go to `data/prepare/`. The rebuild does not yet match the
published pages exactly; see [`reading/docs/TODO.md`](../reading/docs/TODO.md)
item 9.

---

## 2. read: pages to Markdown  `[GPU]`

The reader is Qwen2.5-VL-7B, zero-shot: **0.099 CER** on 15
hand-transcribed pages. It writes one Markdown file per page, with
question numbers as headings (`### 2a)`), maths in `$...$`, struck-out
text as `~~...~~` and each drawing as a `![diagram: ...]` placeholder.
The prompt lives once, in `reading/read/make_colab_notebook.py`, and every
route below uses it.

Every route fills the same folder, `data/read/<engine>/`. `<engine>` is
just a name for the run (`qwen7b` by default; set `MP_ENGINE` or pass
`--engine`), and everything downstream takes the same `--engine`.

| route | for | how |
|---|---|---|
| **Colab** | a corpus pass, free | `python pipeline.py batches` writes `data/read/batches/batch_NN.zip`; upload them, run `reading/read/read_pages_colab.ipynb` on a T4 (it resumes after a dropped session), unzip its download into `data/read/qwen7b/` |
| **Kaggle** | a corpus pass, free, no browser | `python reading/read/kaggle_run.py --user <you> --all` (needs `pip install -e ".[kaggle]"`) |
| **server** | new booklets on demand | `python pipeline.py read --reader server --url http://gpu:8000/v1`; any OpenAI-compatible endpoint (vLLM, llama.cpp, or `reading/read/modal_vllm.py`) |
| **local** | one booklet, no network | `python pipeline.py read --reader local --model-path m.gguf --mmproj-path mmproj.gguf` (minutes a page) |

```bash
python pipeline.py read                           # coverage: what is read, what is not
python pipeline.py read --reader cached --cache-dir <folder>   # import a download
```

**Checking a read** before trusting it:

```bash
python reading/read/qa/check_batch.py data/read/qwen7b    # loops, malformed markers, list-as-headings
python reading/read/qa/audit_pages.py data/read/qwen7b    # text against the ink on each page
python reading/benchmark/ocr_bench.py --engine qwen7b --verbose   # CER against ground truth
python reading/read/prepare_test_batch.py                 # 30 known failures + controls, for a prompt change
python reading/read/qa/compare_runs.py --before data/read/a --after data/read/b
```

A prompt change invalidates every score taken before it. Regenerate the
notebook (`python reading/read/make_colab_notebook.py`) and re-run the
benchmark. `python pipeline.py check` fails while the committed notebook
is out of date.

---

## 3. figures: the diagram pass  `[GPU, optional, recommended]`

The reading pass marks a figure on only 52 of 1,000 pages, because
finding figures was a side clause in a prompt whose real job was
transcription. This pass asks only that question, and reports for each
drawing an *anchor*: the last words written above it, which is how the
drawing is placed in the answer.

```bash
python pipeline.py figures --engine qwen7b        # data/figures/diagram_batch.zip
# run reading/figures/find_diagrams.ipynb on a Colab T4 over that zip,
# then unzip its download into data/figures/  ->  data/figures/diagrams.json
```

Without it, assembly falls back to pairing the reader's own markers with
drawn regions in page order, and appends any unclaimed region at the end
of the page. That is where misplaced figures came from.

---

## 4. assemble: one booklet per student per CIE

```bash
python pipeline.py assemble --engine qwen7b
```

`reading/assemble/build_booklet.py --all`. **Reads** `data/read/<engine>/`,
`data/pages/`, and `data/figures/diagrams.json` and `rejects.json` when
present. **Writes** `data/booklets/<booklet>/booklet.md` (for a person),
`structure.json` (for the handoff) and `figures/*.png`, plus
`data/booklets/booklets.csv`.

- Groups each page's text under the question heading above it, splitting
  a page where a new heading starts. A page that opens mid-answer
  continues the open question.
- Crops each drawing. The diagram pass says what is a drawing and where
  it goes; `segment.py`'s geometry supplies the pixels where it has a
  region. Crops are deliberately generous: a spare line of writing is
  harmless, a clipped arrow is not.
- Cuts a page at a repetition loop, blanks a page with more text than
  its ink can hold, and in both cases puts the whole page image in so a
  person marks from the scan.
- A page that was never read is recorded as a gap under the question
  that was open.

Then review the figures once, and assemble again:

```bash
python reading/assemble/review_figures.py         # data/figures/review.html
# click every crop that is not a drawing, save rejects.json into data/figures/
python pipeline.py assemble --engine qwen7b
```

The reject rate is the figure precision.

---

## 5. handoff: the contract

```bash
python pipeline.py handoff
```

`reading/handoff/export.py`. **Reads** `data/booklets/*/structure.json`
and `data/pages/`. **Writes** `data/handoff/`: the booklets as
`question → part → answer[]`, with page images and crops. The schema and
its rules are in [`HANDOFF.md`](HANDOFF.md). It refuses booklets
assembled from more than one read.

---

## 6. mark: the ladder, the model, the human

```bash
python pipeline.py mark                           # load -> align -> grade -> verdicts -> human -> agreement
python pipeline.py mark --no-semantic             # without torch
python pipeline.py mark --verdicts data/marking/local_verdicts.jsonl
```

`marking/src/run_all.py`. Each rubric item is decided by the cheapest
tier that can honestly decide it (exact values, keyword coverage,
embedding similarity), and anything ambiguous is queued for the model
tier rather than guessed. **Writes** `data/marking/`: `marks/*.json` (the
source of truth), `summary.csv`, the queues, and `agreement.md`, the
comparison with the examiner's own marks.

The model tier answers `data/marking/queue_llm.jsonl` by one of three
routes: the Colab notebook (`marking/src/make_llm_notebook.py`), a local
Qwen (`marking/src/llm_local.py`, read `marking/docs/LOCAL_SETUP.md`
first), or Claude in session (`marking/src/claude_tier.py`). Then
`python pipeline.py mark` again applies the verdicts, after re-checking
every quote. [`marking/README.md`](../marking/README.md) explains the
ladder and its rules.

The rubric needs the scheme renders only to be checked against the
scans: put the three PDFs in `data/schemes/`, then
`python marking/src/render_scheme.py`.

---

## 7. review: the human tier

```bash
python pipeline.py review                         # http://127.0.0.1:8000
```

One booklet per page: the student's answer, their drawings, each rubric
item with the tier that decided it, and the examiner's mark beside ours.
A worklist filters to *still undecided* or *outside the examiner's
total*; `n` / `p` / `h` move between booklets and hide decided questions.
Every decision is appended to `data/marking/human_marks.jsonl` before the
marks change, and `python pipeline.py mark` replays it.

---

## What invalidates what

```
new pages or a new read ──> assemble ──> handoff ──> mark   (python pipeline.py run)
a prompt change         ──> re-read the pages it touches, re-score the benchmark
the diagram pass, or rejects.json ──> assemble ──> handoff ──> mark
an edit to a rubric key ──> validate_keys, make_verify_sheet ──> mark
model verdicts or human decisions ──> mark (replays them; nothing else changes)
```

`python pipeline.py run` re-runs assemble, handoff and mark in order, and
stops at the first failure. `--from handoff` or `--from mark` starts
later. Re-marking never loses a person's work: `grade.py` rewrites the
marks from scratch, but `run_all.py` replays the verdicts and
`human_marks.jsonl` on top every time.
