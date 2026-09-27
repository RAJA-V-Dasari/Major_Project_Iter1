# Handoff — the evaluation half

Pranay: this is part 2. You read 50 booklets into `question → part →
answer`; this marks that output against the department's three CIE
answer schemes and measures the result against the marks the faculty
actually wrote on the covers.

It is self-contained. It reads your handoff and never writes to it.

---

## Which folder to run

There are two, and they are the same project with one difference:

| folder | the model tier runs | use it when |
|---|---|---|
| `Major_Project_Eval/` | on a free Colab T4, from a notebook | you have a browser and a Google account |
| `Major_Project_Eval_Local/` | on this machine, via Ollama | you want no network calls at all |

**You are reading the Colab edition's handoff.** Everything below applies
to both except the model tier. If you would rather keep the whole thing
on one machine, read `Major_Project_Eval_Local/HANDOFF.md` instead —
the prompt and the queue format are byte-identical, so the two produce
comparable results.

**The model tier's verdicts are not in git.** `output/` is gitignored,
because every verdict quotes a student, so the run that produced the
published numbers (`output/grade_llm_verdicts.jsonl`) exists only on the
machine that ran it. Without a verdicts file the ladder still runs and
the model's 740 items stay pending.

---

## What is not in git, and how to supply it

Two inputs are deliberately absent. Both are yours already.

**1. Your handoff corpus** — 187 MB, 1,361 files, every one a real
student's transcribed answer or a crop of their page. It is referenced
in place, never copied. `paths.py` looks for it at
`handoff_Pranay/home/pranay/dev/Major_Project_Iter1/handoff1` relative to
the repo root, which is where it sits on this machine. If yours is
elsewhere:

```bash
set MPE_HANDOFF=D:\wherever\handoff1        # Windows
export MPE_HANDOFF=/home/pranay/.../handoff1  # Linux
```

**2. The three scheme PDFs** — the department's material, 7.3 MB. Same
idea:

```bash
set MPE_SCHEMES=D:\wherever\answer_keys
```

They must be named exactly `CIE 1 scheme CN.pdf`, `CIE 2 Scheme CN.pdf`,
`CIE 3 Scheme CN.pdf` — the capitalisation is inconsistent because the
files are, and `paths.py` carries the real names.

Nothing else is missing. If a path is wrong you get a message naming the
path, not a stack trace.

---

## Setup

```bash
cd Major_Project_Eval
uv venv
uv pip install -e .                 # laptop tiers: pymupdf, pillow, numpy
uv pip install -e ".[semantic]"     # adds torch + sentence-transformers
```

The semantic extra is ~2 GB and decides 6 rubric items out of 1,221. You
can skip it — every script takes `--no-semantic`, and `grade.py` prints a
warning and carries on if it is absent rather than failing.

**Use the venv's interpreter, not a bare `python`.** Running with a
system Python that lacks `sentence-transformers` silently drops the
semantic tier and changes 6 items across 3 booklets. It warns, but the
warning scrolls past.

---

## Prove it works in three commands

```bash
python src/validate_keys.py      # 107 rubric items, 0 errors
python src/gold_check.py         # the examiner's own arithmetic
python src/serve.py --check      # the crops are where we think
```

None of them touch the model or the network. If those three pass, the
rubric, the examiner's marks and the corpus are all readable.

---

## The full run

Order matters — it goes cheapest tier to most authoritative, and each
step's output is the next one's input.

```bash
python src/render_scheme.py        # scheme PDFs -> PNG (once)
python src/run_all.py              # everything below, in order
```

`run_all.py` is the stitch: it reads part 1's handoff and runs

```bash
python src/load_handoff.py --report   # must match your index.json
python src/align.py                   # resolve unlabelled parts -> output/alignment.json
python src/grade.py --all             # the ladder
python src/apply_verdicts.py <the verdicts file in output/>
python src/apply_human.py             # if output/human_marks.jsonl exists
python src/agreement.py               # the deliverable
```

stopping at the first step that fails. Two ways this went wrong by hand,
which is why it is one command now:

- `align.py --report` only *prints* the resolved labels - it returns
  before writing `output/alignment.json`, so `grade.py` then runs without
  them (103 unattempted instead of 90, 9 location failures instead of 6).
  Run `align.py` without `--report`.
- With more than one verdicts file in `output/` (they come from different
  readers), `run_all.py` refuses to guess; name one with `--verdicts`.

**`grade.py --all` rewrites every marks file from scratch.** That is how
the ladder stays reproducible, but it means the last three commands are
not optional extras — they are how the model's and the human's decisions
get put back. Run them as a unit or the marks you see will be the cheap
tiers alone.

To regenerate the model verdicts rather than replay them, see
`notebooks/grade_llm.ipynb` (upload `output/queue_llm.jsonl`, run, bring
back the verdicts file).

`python src/serve.py` then opens the human review tool on
`http://127.0.0.1:8000/` — one booklet per page, with the drawings.

---

## What to consume when you stitch

Everything downstream should read `output/marks/*.json`, one per booklet.
That is the single source of truth; every other output is derived from
it. Shape:

```jsonc
{ "booklet_id": "student_01_cie_2", "cie": 2, "student": 1,
  "marks_settled": 40.0, "marks_pending": 0,
  "max_marks": 40, "counted": ["1", "2a", "2b", "2c", "3a", "4b"],
  "questions": [
    { "id": "2a", "part": "A", "counted": true,
      "marks_available": 5, "marks_settled": 5.0, "marks_pending": 0,
      "answer": "...", "figures": 2, "gaps": 0,
      "items": [
        { "point": "...", "marks_available": 1.0, "awarded": 1.0,
          "tier": "exact",          // exact|keyword|semantic|llm|human|unattempted
          "why": "...", "evidence": [...], "quote": "..." } ] } ] }
```

Three things to know before you build on it:

- **`marks_settled` is not a final mark.** It is what has been decided.
  `marks_pending` is what is still open. A booklet's true mark lies
  somewhere in `[settled, settled + pending]`, and collapsing that to a
  single number treats every undecided item as a zero.
- **`counted` matters.** Part C is answered by choice — `max(3a,3b)` and
  `max(4a,4b)`. Questions not chosen are present in `questions[]` with
  `"counted": false` and must not be summed.
- **`tier` tells you how much to trust a number.** `exact` and `keyword`
  are deterministic; `llm` survived a verbatim-quote check; `human` is a
  person who looked at the drawing.

`output/summary.csv` is the same thing one row per booklet, if that is
easier to join against.

Full shapes for every file: [`docs/DATA_FORMATS.md`](docs/DATA_FORMATS.md).

---

## Where this actually stands

Be aware of this before you present any number from it.

| | |
|---|---|
| Rubric items | 1,221 across 300 counted questions |
| Decided | 634 — exact 139, keyword 141, semantic 6, llm 246, unattempted 90, human 12 |
| **Still undecided** | **587 items across 226 questions** |
| Marks settled | 664 of 2,000 (33%) |

Against the examiner, on 289 comparable questions: **229 inside (79%)**,
21 over-settled, 39 under-settled. 39 of the 48 booklets with an
examiner total have that total inside our range.

**The honest reading of that 79%:** two thirds of the marks are still
undecided, and every undecided item widens our interval, which makes
agreement easier. It is a floor, not a final score. The 587 open items
are mostly waiting on one thing — somebody looking at a drawing — and
`serve.py` is the tool for it.

**The examiner is a reference, not ground truth.** Seven of the fifty
covers carry a demonstrable defect: four Part C totals that ignore the
paper's own choice rule, one arithmetic slip, one grid with no totals,
one with no marks. A grader that matched the covers perfectly would be
reproducing their mistakes. See [`gold/gold_notes.md`](gold/gold_notes.md).

---

## Things that will bite you

- **Run the appliers after `grade.py`.** Covered above; it is the single
  easiest way to get wrong numbers out of this repo.
- **Use the venv Python.** Also covered above.
- **Nothing derived from a booklet is committable.** All of `output/` is
  gitignored, because every file this project writes quotes a student to
  justify itself. The one exception is `gold/gold_marks.csv` —
  pseudonymous integers, no names or handwriting, and the only artifact
  here that cannot be regenerated by running something.
- **`output/queue_human.jsonl` is stale and unused.** It says 111; the
  truth is 587. `serve.py` ignores it and reads the marks directly.
  It is left only because `grade.py` still writes it.
- **The rubric is hand-authored** from phone scans of the schemes,
  because the PDFs contain zero font objects and cannot be parsed. It is
  the ceiling on every number downstream and has not had a second
  reader. [`keys/VERIFY.md`](keys/VERIFY.md) is the sign-off sheet, still
  unsigned.

---

## Read next

| file | what it answers |
|---|---|
| [`README.md`](README.md) | the result, and how the marking works |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the ladder, the asymmetry rule, the carve-outs |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | run order and what each step proves |
| [`docs/DATA_FORMATS.md`](docs/DATA_FORMATS.md) | every file shape |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | settled, pending, chain, gap, inside, zero_blocked |
| [`IMPROVEMENTS.md`](IMPROVEMENTS.md) | known weaknesses, with evidence |

If something here is wrong or missing, `IMPROVEMENTS.md` is the honest
list of what I already know is weak — start there before assuming a
surprise is a bug.
