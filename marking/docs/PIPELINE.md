# Pipeline

What to run, in what order, and what each step proves.

The root `README.md` lists the commands. This file says what each one is
*for*, what it consumes, what it writes, and what goes wrong if you skip
it.

---

## Before anything: where the data is

**The corpus is not in git.** Every booklet is a real student's
scripts, so the handoff, and everything marking writes about it, lives
under the gitignored data root (see `common/layout.py`):

| path | override | holds |
|---|---|---|
| `data/handoff/` | `MPE_HANDOFF` | part 1's booklets, pages and crops (`python pipeline.py handoff` writes it) |
| `data/schemes/` | `MPE_SCHEMES` | the three scheme PDFs (the department's, never committed) |
| `data/marking/` | `MP_DATA` moves all of `data/` | everything this half writes |

`MPE_HANDOFF` is for marking a handoff that lives elsewhere. For
example, the one part 1 shipped in September as `handoff1/`:

```powershell
$env:MPE_HANDOFF = "D:\somewhere\handoff1"
```

Every path helper goes through `paths.require()`, which fails with the
path it looked for and a reminder about these variables, rather than a
bare `FileNotFoundError` three frames deep.

### Install

From the repo root:

```bash
pip install -e .                      # laptop tiers only
pip install -e ".[semantic]"          # adds the embedding tier
```

The split is deliberate. The base install has no ML stack at all.
`grade.py --no-semantic` must keep working, so that the ladder degrades
rather than breaks when the extra is absent.

---

## The order, and why it is this order

The stages are arranged so **a failure surfaces before it contaminates
the next one**. The rubric is checked before it is used to mark; the
examiner's arithmetic is checked before it is used to measure; the
alignment is resolved before questions are matched to it.

### Stage 1 — the rubric

```bash
python marking/src/render_scheme.py               # scheme PDFs -> PNG
python marking/src/validate_keys.py               # rubric arithmetic + structure
python marking/src/make_verify_sheet.py           # the human sign-off sheet
```

| step | proves |
|---|---|
| `render_scheme.py` | The pixels exist at a size a person can read. `--check` prints the PDFs' font count — **zero**, which is the evidence behind transcribing the rubric by hand. 200 dpi is chosen because the marks column on the right edge is the part most needed and least legible. |
| `validate_keys.py` | Items sum to their question; questions sum to the paper once Part C's choice is resolved; `choice_with` is reciprocated; no duplicate ids; every `scheme_page` names a file that exists; every short numeric `exact` value is flagged for a human. |
| `make_verify_sheet.py` | Generates `keys/VERIFY.md`. Ordered to be worked through with the PDF open, inferred splits first. |

**Why this runs first.** The rubric is the ceiling on every number this
project produces, and a slip in it is silent. A question whose items sum
to 9 when the question is worth 10 does not crash — it quietly loses one
mark on all fifty booklets and lands as a systematic disagreement that
looks like a grading problem.

`validate_keys.py` is mechanical. `VERIFY.md` is the half no machine can
do: nothing mechanical can tell whether a mark was attributed to the
right claim. **Re-run `make_verify_sheet.py` after any edit to a key.**

### Stage 2 — the examiner's marks

```bash
python marking/src/crop_covers.py                 # cover marks grids
python marking/src/gold_check.py                  # the examiner's own arithmetic
```

`crop_covers.py` cuts the marks grid out of each cover page. The crop
window starts at 0.50 of page height and **that is a privacy control,
not a framing choice** — the name row sits at ~0.30 and the USN row at
~0.37, so half a page of clearance sits between the last identifying
field and the top of the crop. Do not lower `--top` to get more context;
there is nothing above the grid worth having and a student's name below
it.

It reaches to 0.96 because the foot of the page repeats the total in a
separate box. That repeat is free redundancy and the cheapest check in
the project.

Every crop is tiled onto a contact sheet **before a single one is read**,
because a fixed fractional window is a guess: across 50 phone scans some
pages sit higher in frame, and a crop that clips the grid's top row
loses question 1's marks silently — the file still exists, still opens,
and is still wrong.

`gold_check.py` then checks the transcription against the arithmetic the
examiner wrote. It separates two questions that are easy to confuse:

- **Did we transcribe the cover correctly?** A failure is probably ours.
  Reported as a problem. (2 of ~400 cells were misreads, both caught.)
- **Did the examiner mark according to the paper?** A failure is theirs.
  Reported as an observation.

It cannot catch two compensating errors in one row, or a row left blank.
It is a filter, not a proof, and the `confidence` column records whether
a human has actually looked.

### Stage 3 — load and align

```bash
python marking/src/load_handoff.py --report       # must match part 1's totals
python marking/src/align.py --report              # what to confirm
python marking/src/align.py                       # write the proposals
```

`load_handoff.py --report` is a contract check against part 1. Three
rules, each of which produces a wrong *mark* rather than an error if
ignored:

1. `answer[]` is the source of truth, not `part.text` — the latter is a
   convenience join whose construction this project does not control.
2. Skip anything with `excluded: true`. Scoring struck-out work awards
   marks the student explicitly withdrew. Use `.get("excluded")`, never
   `item["excluded"]` — a `gap` item has no such key.
3. A `gap` means incomplete, not short.

`align.py` resolves parts carrying no usable question label. The
headline result is a negative one worth knowing: part 1 flagged 35 of
325 parts as unlabelled, and **all 35 turned out to be section headings**
the student wrote at the top of a page ("PART - A", "INTERNALS - II PART
- A:-"), every one 25 characters or shorter, none carrying a figure or a
gap. So 11% of parts is **0% of the marks**.

What does need resolving is five parts whose label is not a question id
on the paper. Three independent signals — structure, distinctive
vocabulary weighted by rarity, and whether the examiner marked that
question — and a proposal is only `confident` when they agree. **Nothing
is assigned automatically**; `grade.py` reads only the confirmed ones.

`grade.py --no-alignment` ignores the file entirely.

### Stage 4 — mark

```bash
python marking/src/grade.py --all                       # the ladder
python marking/src/grade.py --all --no-semantic         # ...without torch
python marking/src/grade.py --booklet student_01_cie_2 --verbose
```

Writes `data/marking/marks/*.json`, `data/marking/summary.csv`,
`data/marking/queue_llm.jsonl`, `data/marking/queue_human.jsonl`.

Roughly two minutes for all booklets with the semantic tier on.

### Stage 5 — calibrate

```bash
python marking/src/calibrate.py                   # sweep and write the report
python marking/src/calibrate.py --quick           # coarse
```

Sweeps the five thresholds against the examiner's marks and writes
`data/marking/calibration.md`. It reports a **frontier, not a winner**:
loosening settles more marks and makes more irreversible errors.

The selection rule is *fewest irreversible errors, then most decisive* —
not *most decisive within an error budget*. Settling more marks only
saves human time; an over-settled mark awards credit the student did not
earn and no later tier can take it back.

> **Read [`../IMPROVEMENTS.md`](../IMPROVEMENTS.md) §1 before trusting
> this stage.** The committed thresholds and the operating point in the
> report currently disagree.

### Stage 6 — the model tier

```bash
python marking/src/setup_local.py --time-it       # measure this machine first
python marking/src/llm_local.py --check           # backend and model are really there
python marking/src/llm_local.py --dry-run         # the exact prompt, sent nowhere
python marking/src/llm_local.py --limit 25        # a sample
python marking/src/llm_local.py                   # the rest, resumable
python marking/src/apply_verdicts.py data/marking/local_verdicts.jsonl --dry-run
python marking/src/apply_verdicts.py data/marking/local_verdicts.jsonl
python marking/src/llm_local.py --compare         # local vs the recorded Colab run
```

**The tier has three routes, and they are alternatives, not
replacements.**

- **Colab.** `make_llm_notebook.py` writes the notebook, which you upload
  the queue to on a free T4. It is still the fastest way through a full
  queue if you have a GPU session.
- **Local.** `llm_local.py` runs Qwen2.5 through Ollama or llama.cpp on
  this machine (the commands above).
- **In session.** `claude_tier.py` lets Claude read the queue in a Claude
  Code session, with a second pass shown the drawings.

`--compare` exists because the routes are expected to disagree, and the
disagreement should be measured rather than assumed.

What makes the routes interchangeable is the reason the tier was shaped
this way at all: it is a queue on disk, not an in-process call. Queue
records are self-contained by design, so whatever marks them needs no
access to this repo and none to the corpus. Adding a route costs one
file and no change to the ladder.

The prompt is byte-identical to the Colab one, imported by both from
[`../src/llm_prompt.py`](../src/llm_prompt.py). Verified against all 740
queue records, including the 275 carrying a chain note and the 438
carrying a figure note.

Two practical warnings, both in [`LOCAL_SETUP.md`](LOCAL_SETUP.md) in
full. **The queue is tens of hours on a CPU** — run `--limit` first and
treat a sample as a sample. And **a model too large for your RAM does
not refuse to load**, it swaps, and a run that should take twenty hours
takes a week while looking identical from the outside.

`apply_verdicts.py` is the half that matters. It refuses:

- an award whose quote is not verbatim in the answer
- a verdict for an item that is not pending (already settled, or a
  human's) — applying it would overwrite a more reliable tier
- marks above the item's value, or negative
- a `zero` on an answer whose evidence is a drawing the model was never
  shown, or a part that lost a page

`decline` is not a failure. It sends the item to the human queue holding
the crops, which is what declining is for.

Writes `data/marking/verdict_audit.md` and updates `data/marking/marks/*.json`
**in place**.

### Stage 7 — the human tier

```bash
python marking/src/serve.py --check               # the crops are really on disk
python marking/src/serve.py                       # http://127.0.0.1:8000
python marking/src/serve.py --port 8080
```

One booklet a page, with the drawings. Every decision is appended to
`data/marking/human_marks.jsonl` **before** the marks file is rewritten, so
the log is the record and the marks file is the derived state.

### Stage 8 — the deliverable

```bash
python marking/src/agreement.py                   # -> data/marking/agreement.md
```

---

## The re-run graph — what invalidates what

This is the part that is easy to get wrong, because several stages write
into `data/marking/marks/*.json` **in place** and nothing recomputes
downstream reports automatically.

```
edit a key ──> validate_keys ──> make_verify_sheet ──> grade --all
                                                          │
                                        (marks, summary, queues rewritten;
                                         ALL model and human decisions lost)
                                                          │
                                       ┌──────────────────┴──────────┐
                                       v                             v
                                 llm_local.py                serve.py (human)
                            (or make_llm_notebook)                   │
                                       v                             v
                              apply_verdicts ──────> marks updated in place
                                                          │
                                                          v
                                                     agreement.py
```

Three consequences:

1. **`grade.py --all` is destructive to the tiers above it.** It
   regenerates `data/marking/marks/*.json` from scratch. Every applied model
   verdict and every human decision in those files is gone. The record
   survives in `human_marks.jsonl` and the verdicts file, but replaying
   them is a manual re-run of stages 6 and 7.
2. **`summary.csv` is written only by `grade.py`** and is never updated
   by `apply_verdicts.py` or `serve.py`. Once either of those has run,
   it is stale by construction.
3. **`agreement.py` and `calibrate.py` must be re-run last**, after
   every tier has finished writing.

### Vintage of the committed run

The `data/marking/` directory in this working copy was not produced in one
pass. From file timestamps:

| written | file | by |
|---|---|---|
| 2026-09-18 12:40 | `alignment.json`, `alignment.md` | `align.py` |
| 2026-09-19 12:22 | `calibration.md` | `calibrate.py` |
| 2026-09-19 14:11 | `summary.csv`, `queue_llm.jsonl`, `queue_human.jsonl` | `grade.py --all` |
| 2026-09-20 21:12 | `grade_llm_verdicts.jsonl` | Colab (preserved; a local run writes `local_verdicts.jsonl` instead) |
| 2026-09-20 22:04 | `verdict_audit.md` | `apply_verdicts.py` |
| 2026-09-20 22:29 | `agreement.md` | `agreement.py` |
| **2026-09-20 22:37** | `human_marks.jsonl`, `marks/*.json` | `serve.py` |

So `marks/*.json` is the newest state, and **`agreement.md` predates the
12 human decisions recorded eight minutes after it.** The headline
numbers in the root README come from that report and would move if
`agreement.py` were re-run today.

`summary.csv` is two stages behind: it says `student_01_cie_2` has 21.0
settled, `agreement.md` says 31, and the marks file itself says 40.0.
All three were true when written. Only the last is true now.

**If you are picking this up: run `agreement.py` first, and treat
`summary.csv` as a `grade.py` artifact rather than a status report.**

---

## Verification, since there is no test suite

Every stage carries its own check. Ordered as they should be run:

| command | checks |
|---|---|
| `render_scheme.py --check` | the PDFs' font count and which pages rendered |
| `validate_keys.py --cie 2 --verbose` | rubric arithmetic and structure |
| `gold_check.py --explain` | the examiner's cover arithmetic, row by row |
| `load_handoff.py --report` | the handoff contract and part-1 totals |
| `align.py --report` | what is unlabelled and why it does not matter |
| `setup_local.py` | RAM, disk, backend, and whether the longest prompt fits |
| `setup_local.py --time-it` | one real item, timed, extrapolated to the queue |
| `llm_local.py --check` | the backend is up and the model is really pulled |
| `llm_local.py --dry-run` | the exact prompt that would be sent |
| `make_llm_notebook.py --check` | every generated cell compiles |
| `apply_verdicts.py ... --dry-run` | what would be applied and refused |
| `serve.py --check` | the crops referenced by the queue are on disk |

`--check`, `--report` and `--dry-run` *are* the verification. Treat a
change that breaks one of them the way you would treat a failing test.
