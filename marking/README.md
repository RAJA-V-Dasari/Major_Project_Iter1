# Marking: booklets against the official schemes

The second half of the pipeline. It takes part 1's handoff (every
booklet as `question → part → answer`, with its drawings) and marks it
against the department's three CIE answer schemes. It then measures the
result against the marks the faculty actually wrote on the covers.

It pays for no inference. Four tiers are plain Python. The model tier is
a queue on disk that any of three readers can answer:

| route | where the model runs | how |
|---|---|---|
| Colab | Qwen2.5-7B, 4-bit, on a free T4 | `src/make_llm_notebook.py` → `notebooks/grade_llm.ipynb` |
| local | Qwen2.5-7B via Ollama or llama.cpp, on this machine | `src/llm_local.py`; read [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md) first |
| in session | Claude, reading the queue in a Claude Code session | `src/claude_tier.py` |

The prompt has one definition ([`src/llm_prompt.py`](src/llm_prompt.py))
and every route uses it. Every check that makes the tier trustworthy
(the verbatim-quote check, the refused zeros) runs downstream of the
model, in `apply_verdicts.py`, whichever route produced the verdicts.

```bash
python pipeline.py mark            # from the repo root: the whole run, in order
python pipeline.py review          # the human tier, http://127.0.0.1:8000
```

---

## The result

Measured on part 1's September handoff: 50 booklets, one per student,
read by the other part 1 implementation (archived with the old `main`).
Each question is compared as the interval `[settled, settled + pending]`
against the examiner's mark, because most items were still undecided
when these were measured.

| run | questions compared | examiner's mark reachable | over-settled | under-settled |
|---|---|---|---|---|
| Qwen2.5-7B on Colab (2026-09-20) | 289 | **229 (79%)** | 21 | 39 |
| Claude in session, incl. a pass shown the drawings (2026-09-27) | 289 | **150 (52%)** | 31 | 108 |

The two are not a ranking. The Qwen run left far more items open, and
every open item widens the interval, which makes "reachable" easier. The
in-session run settled 946 of 1,221 items (383 by the deterministic
ladder, 563 by the model tier). It left 275 for a person: mostly
drawing-heavy answers, chain questions and parts with a lost page. Its
95 awards that rest on a drawing could not be quote-checked, and the
website labels them "not quote-checked" for spot checks. Every mark
records which reader gave it.

**The examiner is a reference, not ground truth.** Seven of the fifty
covers carry a demonstrable defect: four Part C totals that ignore the
paper's own choice instruction, one arithmetic slip, one grid with no
totals, one with no marks at all. A grader that matched this reference
perfectly would be reproducing its mistakes. Where we differ, the
finding is "we differ, and here is why", and on the four choice-policy
booklets we are right. See [`gold/gold_notes.md`](gold/gold_notes.md).

One open question on the scheme itself: CIE 3 Q3a's key gives NRZ-I at
10 Mbps as 500 kbaud / 500 kHz, but N/2 of 10 Mbps is 5 Mbaud / 5 MHz.
Check it against the scheme PDF.

---

## How it marks

Each rubric item is decided by the cheapest tier that can honestly
decide it, and every award records which tier decided and on what
evidence.

| tier | decides | runs on |
|---|---|---|
| `exact` | items naming concrete values (`57088`, `/26`, `14.24.74.126`) | laptop |
| `keyword` | items where coverage is decisive either way | laptop |
| `semantic` | MiniLM breaking ties on keyword-supported items | laptop (CPU) |
| `llm` | everything still ambiguous | any of the three routes above |
| `human` | anything whose evidence is a drawing | a person, via `serve.py` |

### Three rules that hold the thing together

**1. A cheap tier may award on evidence it finds, but is restrained from
concluding absence.** Finding the evidence is proof; failing to find it
is not. There are four carve-outs, each written against a measured
failure:

- **chain questions** (subnetting, fragmentation, CRC, the delay
  cascade) are never zeroed cheaply. A wrong block size at step one
  makes every later value legitimately differ, and zeroing those
  charges one slip five times.
- **a part containing a lost page** is incomplete, not wrong.
- **an answer whose evidence is a drawing** cannot be zeroed on prose.
  On `student_01_cie_2`, question 2b's entire routing table sits in four
  crops behind 73 characters of text. The examiner gave 5/5, and marking
  the prose alone scored 0.
- **a long answer with no keyword overlap** is more likely our
  vocabulary missing theirs than the student saying nothing.

**2. Similarity may confirm evidence. It may never supply it.** An item
whose keywords are absent is not decided by the embedding tier however
topical it reads: every wrong subnet is about subnetting. The same rule
applies to exact values. `student_23_cie_3` wrote 265 words about
Fletcher checksums without one number from the scheme, and briefly
scored 3/3 on topical similarity while the examiner gave zero.

**3. The model may not award a mark it cannot quote the student's own
words to support, and may not zero what it cannot see.**
`apply_verdicts.py` re-checks every quote against the answer and discards
any award whose quote is not there. On the first real run, 26 of
Qwen2.5-7B's 169 awards (15.4%) cited a quote that is not in the answer.
The same run zeroed 295 of 438 drawing-backed items it was told it could
not see, so those zeros are refused into the human queue (`zero_blocked`)
rather than settled. That moved under-settled questions from 167 to 38.

### Part C is a choice

`max(3a, 3b)` and `max(4a, 4b)`: the better half, never the sum, as the
paper instructs.

---

## Running it

Commands run from the repo root. `python pipeline.py mark` is
`src/run_all.py`, which runs, in this order, stopping at the first
failure:

```bash
python marking/src/load_handoff.py --report   # must match the handoff's own totals
python marking/src/align.py                   # resolve parts with no usable label
python marking/src/grade.py --all             # the ladder
python marking/src/apply_verdicts.py <file>   # the model tier's verdicts, if any
python marking/src/apply_human.py             # the human log, if any
python marking/src/agreement.py               # the deliverable
```

The model tier, whichever route you use:

```bash
python marking/src/llm_local.py --limit 25          # local: a sample first
python marking/src/make_llm_notebook.py             # Colab: upload the queue to it
python marking/src/claude_tier.py --next 12         # in session
python pipeline.py mark --verdicts data/marking/local_verdicts.jsonl
```

The rubric and the examiner's marks have their own checks, which need no
data at all:

```bash
python marking/src/render_scheme.py       # scheme PDFs -> PNG (needs data/schemes/)
python marking/src/validate_keys.py       # rubric arithmetic and structure
python marking/src/make_verify_sheet.py   # the human sign-off sheet, keys/VERIFY.md
python marking/src/gold_check.py          # the examiner's own arithmetic
python marking/src/calibrate.py           # sweep the thresholds against the examiner
```

Every stage carries its own check (`--report`, `--check`, `--dry-run`),
and `python pipeline.py check` runs all the ones that need no model.
[`docs/PIPELINE.md`](docs/PIPELINE.md) says what each step proves and
what invalidates what.

### Where the data lives

| path | holds | in git? |
|---|---|---|
| `marking/keys/` | the rubric, hand-authored from the scheme scans | yes |
| `marking/gold/gold_marks.csv` | the examiner's marks as pseudonymous integers | yes, the one exception |
| `data/handoff/` | part 1's booklets, what is marked | no |
| `data/marking/` | marks, queues, verdicts, the human log, every report | no |
| `data/schemes/` | the department's three scheme PDFs, and their renders | no |

To mark a handoff that lives elsewhere, such as the September
`handoff1/`, set `MPE_HANDOFF` to it.

---

## Things that will bite you

- **`grade.py --all` rewrites every marks file from scratch.** The model's
  and the human's decisions are put back only by `apply_verdicts.py` and
  `apply_human.py`. `python pipeline.py mark` always runs them in order;
  running `grade.py` alone shows the cheap tiers and nothing else.
- **Two verdicts files from different readers** make `run_all.py` refuse
  to guess. Name the one to apply with `--verdicts`.
- **Use one interpreter.** Without `sentence-transformers` the semantic
  tier is skipped with a warning that scrolls past, and a handful of
  items move.
- **`data/marking/queue_human.jsonl` is written before any verdict** and
  is stale the moment one is applied. `serve.py` ignores it and reads
  the marks directly.
- **The rubric is hand-authored** from phone scans, because the scheme
  PDFs contain zero font objects and cannot be parsed. It is the ceiling
  on every number downstream and has had one reader.
  [`keys/VERIFY.md`](keys/VERIFY.md) is the sign-off sheet, and it is
  still unsigned.

---

## What the numbers do not cover

- **Our inferred mark splits are measurably worse than the scheme's
  printed ones**: 89% against 96% containment. Where the scheme gave a
  total and no breakdown, the division into rubric items is our
  judgement, and that seven-point gap is what it costs.
- **The semantic tier barely earns its place.** Calibration over 1,600
  settings drives it to almost zero at every low-error operating point.
  It was the main source of over-awards, and constrained not to
  over-award it now fires on 6 items out of 1,221.
- **The thresholds in `src/tiers/__init__.py` are not the sweep's
  pick.** The deviation is recorded there and in
  [`IMPROVEMENTS.md`](IMPROVEMENTS.md) §1 rather than quietly changed,
  because changing them re-marks every booklet.
- **Every number above came from the September handoff.** The reading
  half in this repo assembles its own; its first marked run is on the
  list in [`../reading/docs/TODO.md`](../reading/docs/TODO.md).

---

## Read next

| file | answers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | how the marking works and why: the ladder, the asymmetry, the carve-outs |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | run order, what each step proves, what invalidates what |
| [`docs/DATA_FORMATS.md`](docs/DATA_FORMATS.md) | every file this half reads and writes |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | settled, pending, chain, gap, inside, zero_blocked |
| [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md) | running the model tier on this machine: read before starting |
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | known weaknesses, with evidence |
| [`keys/SCHEMA.md`](keys/SCHEMA.md) | the rubric format, and what may go in `exact` |
| [`../docs/HANDOFF.md`](../docs/HANDOFF.md) | the contract this half reads |
