# What can be improved

A review of `Major_Project_Eval` as it stands on 2026-09-21, plus
[§4](#severity-4--found-while-porting-the-model-tier-local), added
2026-09-23 when the model tier was moved onto this machine.

**Status: severity 1 and severity 2 have been acted on.** The findings
below are left as written, because the evidence is the useful part and a
rewritten review cannot be checked against what was actually found. What
changed in response:

| finding | resolution |
|---|---|
| 1.1 thresholds not the calibrated ones | half confirmed. The docstring's *numbers* were right - shipped really does score 11 over / 5 under / 22% decisive; `calibration.md` was the stale artifact. Its *provenance* claim was false: the sweep recommends 0.85/1.00/0.55, not the shipped row. The deviation is now recorded with both scores measured side by side, and left open - see below |
| 1.2 `grade.py --all` destroys decisions | `src/apply_human.py` replays the log. Proved by wiping all 50 booklets and restoring them byte-for-byte |
| 1.3 three contradicting vintages | `write_summary()` extracted from `grade.py`; both appliers now refresh `summary.csv` after they decide |
| 2.1 dead constants | `SEM_CONFIDENT` and `KW_CORROBORATE` removed from `semantic.py`, with a note on why a threshold has one home |
| 2.2 `exact.decide()` has no docstring | statement moved below it |
| 2.3 phantom `--no-llm` | comment corrected |
| 2.4 fourth carve-out elsewhere | documented in `exact.decide()`, pointing back from `grade_item()` |

Two further defects surfaced while checking 1.1, both in `calibrate.py`:
its module docstring described a selection rule the code had stopped
using, and `acceptable` was computed under that old rule. The docstring
now describes what the code does, and `acceptable` is explicitly
reporting-only.

**One decision is open.** A fresh sweep recommends `kw_confident=0.85,
kw_corroborate=1.00, sem_confident=0.55` against the shipped
`0.67/0.67/0.62`. Shipped trades one extra over-settled question for two
points of decisiveness, which is the opposite of this project's stated
rule. Adopting the sweep's row would re-mark all fifty booklets and would
retire the semantic tier in all but name, since `kw_corroborate=1.00`
lets similarity confirm only items whose keywords are already all
present. That is deliberate enough to be a decision rather than an edit,
so it is recorded in `Thresholds` and not taken.

Severity 3 is untouched: it is method rather than defect, and 3.1 and
3.3 are the user's call to make.

Findings are ordered by what they cost if left alone, not by how hard
they are to fix. Each one states the evidence, so it can be checked or
dismissed rather than argued about.

Findings are ordered by what they cost if left alone, not by how hard
they are to fix. Each one states the evidence, so it can be checked or
dismissed rather than argued about.

**A note on tone.** This codebase documents its own reasoning unusually
well, and several of the entries below exist *because* the code
explains itself — a comment that states a measured fact is checkable,
and three of these are cases where the code and its own comment have
drifted apart. That is a good problem to have.

---

## Severity 1 — can produce a wrong number or lose work

### 1.1 The committed thresholds are not the calibrated ones

**Evidence.** `output/calibration.md` ends with:

```python
kw_confident   = 0.6      kw_absent      = 0.34
kw_corroborate = 0.5      sem_confident  = 0.55
zero_max_chars = 300
```

`src/tiers/__init__.py` ships:

```python
kw_confident=0.67, kw_absent=0.34, kw_corroborate=0.67,
sem_confident=0.62, zero_max_chars=150
```

Three of the five differ. The `Thresholds` docstring says *"The defaults
below were CHOSEN by that sweep, over 1,600 settings against 289
examiner-marked questions. See output/calibration.md. At this point: 11
over-settled, 5 under-settled, 22% of marks decided"* — but the report
it points at records **33 over-settled, 6 under-settled, 41% decisive**
at its own chosen point, and `output/agreement.md` records 21 and 38 for
the run that actually shipped. No two of those three agree.

**Why it matters.** The calibration is the project's justification for
its operating point, and right now it justifies a setting the code does
not use. Anyone checking the claim will find this in about five minutes,
and it undercuts the strongest methodological argument the project has.

**What I cannot tell from here.** Whether 0.67/0.62/150 was a deliberate
manual tightening after the sweep — the `zero_max_chars` docstring
argues specifically for protecting 270–654-character answers, which 150
does and 300 does not, so a deliberate choice is plausible. If so, the
fix is documentation; if not, it is a copy that never happened.

**Fix.** Re-run `calibrate.py` against the current code, then either
adopt the chosen point or record in `Thresholds` why the shipped values
deviate and what that deviation costs. Better: have `calibrate.py`
*write* the chosen values to a small `keys/thresholds.json` that
`Thresholds` loads, so the two cannot drift again. The docstring's own
argument for parameterisation — *"the two copies would drift apart on
the first change"* — applies one level up, to the report and the code.

### 1.2 `grade.py --all` silently destroys every model and human decision

**Evidence.** `grade.py` writes `output/marks/<booklet>.json`
unconditionally (line ~442). `apply_verdicts.py` and `serve.py` both
update those same files in place. Nothing reads `human_marks.jsonl`
back — `grep -rl human_marks src/` returns only `serve.py`, which
writes it.

So a single `grade.py --all` throws away all 246 applied verdicts and
all 12 human decisions with no prompt, no backup and no warning. The
verdicts can be re-applied by re-running `apply_verdicts.py`. **The
human decisions cannot be replayed by anything in this repo.**

**Why it matters.** The human tier is the project's scarcest resource —
587 items are queued to it and the README is explicit that the numbers
do not move until somebody works through them. Losing that work to a
routine re-grade is the most expensive accident available here, and it
is one keystroke away.

**Fix.** Two cheap options, in order of preference:

1. A `replay_human.py` (or a `--replay` flag on `grade.py`) that folds
   `human_marks.jsonl` back into fresh marks files. The log already has
   everything needed: booklet, question, item index, marks, note,
   timestamp. This makes `grade.py --all` safe by making it recoverable.
2. Failing that, have `grade.py` refuse to overwrite a marks file
   carrying human- or llm-decided items unless `--force` is passed.

### 1.3 `output/` holds three different vintages that contradict each other

**Evidence.** By file timestamp: `summary.csv` is from the 09-19 14:11
grade run, `agreement.md` from 09-20 22:29, and `marks/*.json` from
09-20 22:37 — after 12 human decisions were recorded at 22:37.

For `student_01_cie_2`, settled marks read **21.0** in `summary.csv`,
**31** in `agreement.md`, and **40.0** in the marks file. All three were
correct when written.

**Why it matters.** The root README's headline numbers come from
`agreement.md`, which predates the human tier's only session. The
reported figures are therefore slightly conservative, and — more
importantly — a reader comparing two files in `output/` will conclude
something is broken.

**Fix.** Write a provenance stamp into every generated report: the
timestamp and the mtime of each input it read. A report that says *"this
read marks/ as of 22:29"* is self-diagnosing. Then re-run
`agreement.py`, which is a two-minute job and moves the headline.

### 1.4 The human tier is the bottleneck and has no throughput story

**Evidence.** 587 of 1,221 items (48%) are pending; 105 of those need a
human because the evidence is a drawing. Twelve have been done. At the
observed rate — the whole `human_marks.jsonl` covers one booklet's
worth of items in a single session — finishing is a substantial manual
project, and the README concedes that until somebody works through it
the numbers do not move.

**Why it matters.** This is not a code defect; it is the project's
actual critical path, and it is worth naming as such rather than leaving
implicit in a bullet near the end of the README.

**Fix.** Three things, cheapest first:

- **Report progress.** `serve.py` has no "N of 587 done, M booklets
  untouched" view. A visible burn-down turns an open-ended chore into a
  finite one.
- **Order the queue by value.** Items are currently queued in booklet
  order. Sorting by marks at stake, or by whether the item is the last
  one blocking a question from being settled, would move the headline
  numbers fastest per minute of human attention.
- **Attack the 105 figure items at the source** — see §3.1.

---

## Severity 2 — misleading to a reader, harmless to the run

### 2.1 Four module constants are dead, and one contradicts live behaviour

**Evidence.**

| constant | value | actually used? |
|---|---|---|
| `exact.KW_CONFIDENT` | 0.75 | no — `decide()` reads `th.kw_confident` (0.67) |
| `exact.KW_ABSENT` | 0.20 | no — `th.kw_absent` (0.34) |
| `semantic.SEM_CONFIDENT` | 0.62 | no — `th.sem_confident` (0.62) |
| `semantic.KW_CORROBORATE` | **0.50** | no — `th.kw_corroborate` (**0.67**) |

`grep -n 'KW_CONFIDENT\|KW_ABSENT\|SEM_CONFIDENT\|KW_CORROBORATE' src/`
finds only the definitions.

The last one is the harmful one. `semantic.py`'s comment reads *"See the
module docstring — this is the rule, not a knob"* directly above
`KW_CORROBORATE = 0.50`, so a reader learns that the corroboration
threshold is 0.50. **The code runs 0.67.** A reader reasoning about why
the semantic tier fires on only 6 of 1,221 items will reason from the
wrong number.

Both blocks are also labelled `PROVISIONAL … calibrated in step 7`,
which was true before `Thresholds` existed and is now stale.

**Fix.** Delete all four. They are leftovers from before the thresholds
moved into the frozen dataclass, and the dataclass is unambiguously the
right home.

### 2.2 `exact.decide()` has no docstring, because a statement precedes it

**Evidence.** In `src/tiers/exact.py`:

```python
def decide(item, text, *, no_zero=None, th=DEFAULT):
    if no_zero is None and len(text) > th.zero_max_chars:
        no_zero = (...)

    """Award, zero, or pass up the ladder.
    ...
    """
```

`exact.decide.__doc__` is `None`. The text is a no-op string expression
in the function body.

This is the *most important* explanation in the tier — it is what
documents the `no_zero` contract — and it is invisible to `help()`, to
IDE hovers and to any documentation tool. The `zero_max_chars` rule is
also applied before the explanation of what the function does, which
reads as if it were bolted on.

**Fix.** Move the docstring above the statement. Purely cosmetic in
behaviour, but it restores the one piece of prose a new contributor most
needs.

### 2.3 `pyproject.toml` documents a flag that does not exist

**Evidence.** The dependency comment says *"`grade.py --no-semantic
--no-llm` must keep working"*. `grade.py` has `--all`, `--booklet`,
`--no-semantic`, `--no-alignment`, `--verbose`. There is no `--no-llm`,
and `grep -rn 'no-llm' src/` returns nothing.

The intent is satisfied anyway — the llm tier is a queue, not an
in-process call, so nothing to disable — but the comment describes an
interface that was never built.

**Fix.** One-line correction to the comment.

### 2.4 One of the four carve-outs lives in a different module from the other three

**Evidence.** `grade.grade_item()` builds `no_zero` from three
conditions — chain, gap, drawing — and passes it down. The fourth, the
long-answer rule (§4.4 of `ARCHITECTURE.md`), is applied separately at
the top of `exact.decide()`:

```python
if no_zero is None and len(text) > th.zero_max_chars:
```

The behaviour is correct — the guard means it only fires when the other
three did not, and the reported tier is right in both branches. The
problem is discoverability: `grade.py`'s docstring documents four
carve-outs together, and someone auditing `grade_item()` for them will
find three and reasonably conclude the fourth was lost.

It also means `exact.decide()`'s own docstring lists only the three it
was handed, omitting the one it adds itself.

**Fix.** Move the length condition into `grade.grade_item()` beside the
other three, so the set of carve-outs is enumerated in exactly one
place. Behaviour-neutral.

---

## Severity 3 — method, not defects

These are things the project could do better, not things it did wrong.
Several are explicitly out of scope for a student project on a deadline;
they are recorded so the scope decision is visible.

### 3.1 The 105 figure items are a solvable problem, not a permanent human cost

The largest single block of human work exists because no tier reads
pixels. Part 1 already runs a vision model over these exact pages, and
already produced a diagram pass that identifies which crops are
drawings.

A VLM tier between `llm` and `human` — shown the crop *and* the rubric
item, under the same quote-or-decline discipline the text model is held
to — would collapse most of these. It would need the same enforcement
`apply_verdicts.py` already implements, and it should be measured the
same way: what fraction of its awards survive the check.

The counter-argument is real and should be recorded alongside: the
existing model fabricated 15.4% of its quotes on *text it could see*,
and a model asked to justify a mark from a routing-table crop has more
room to invent, not less. This is an experiment with a measurable
outcome, not an obvious win.

### 3.2 The semantic tier fires on 6 items out of 1,221

The README already concedes it barely earns its place. At 0.5% of
decisions it is carrying a ~90 MB optional dependency, a whole module, a
calibration axis and a paragraph of doctrine.

Two honest options: **delete it** and simplify the ladder to four tiers,
or **widen it** — the current rule requires 67% keyword coverage before
similarity may confirm, and the entire point of the tier is to catch
answers phrased in other words, which is precisely the case where
keyword coverage is low. The rule as written may be self-defeating.

Whichever, decide it deliberately and record the measurement, the way
the deleted modules in part 1 were.

### 3.3 The rubric has one reader and no second opinion

107 rubric items were transcribed by eye from skewed phone scans by one
person, and the README correctly names this as the ceiling on everything
downstream. `validate_keys.py` proves the arithmetic; `VERIFY.md` is a
checklist for the same person who wrote the keys.

**A cheap partial control:** double-enter a random sample of 15–20 items
from the scans — ideally by a second person, but even the same person
weeks later on a shuffled order — and report the disagreement rate. That
converts "the rubric is the ceiling" from an acknowledged risk into a
measured one, which is the standard the rest of the project holds
itself to.

### 3.4 There is no test suite, and a handful of pure functions deserve one

The repo states this deliberately, and for the pipeline stages the
`--check` / `--report` / `--dry-run` argument is sound: they need the
corpus, and the corpus cannot be committed.

But several functions are pure, corpus-free and load-bearing:

- `exact.normalise` — the dash zoo, NFKC, punctuation preservation
- `exact._pattern` / `contains` — the boundary rules, including the
  stemming asymmetry (`8001` must not match inside `18001`; `acknowledg`
  must match `acknowledged`; `80` must not stem into `8080`)
- `exact.keyword_coverage` — group-of-alternates semantics
- `grade.resolve_choices` — `max()`, provisional detection, tie-breaks
- `apply_verdicts`' quote normalisation — the one check the model tier's
  integrity rests on

A dozen assertions over invented strings would cost an afternoon, ship
with no student data, and pin down exactly the rules the comments say
were hard-won. The boundary rules in particular are the kind of thing
that breaks silently under a well-meaning refactor.

### 3.5 `examiner_marked` reads Part C at row level

`grade.examiner_marked()` falls back to the row total for questions 3
and 4 because the examiner used the grid's columns loosely there. This
is correct given the data, but it means the "did the examiner mark this
question?" signal — one of the three independent signals `align.py`
relies on — is weaker on Part C than elsewhere. Worth stating in the
alignment report rather than only in code.

### 3.6 Reproducibility: no dependency pins

`pyproject.toml` specifies floors (`pymupdf>=1.24`,
`sentence-transformers>=3.0`) and there is no lock file. The semantic
tier's results depend on the MiniLM revision, and `pymupdf`'s rendering
could shift the scheme PNGs that `scheme_page` references point into.

A committed `uv.lock` (or `requirements.txt` with hashes) costs nothing
and makes the reported numbers reproducible rather than approximately
reproducible.

---

## Severity 4 — found while porting the model tier local

Two findings, both pre-existing and neither introduced by the port. They
are numbered separately because they were found later, not because they
matter less — 4.1 affected every one of the 740 prompts in the recorded
run.

### 4.1 Every prompt told the model the question was "worth ? marks overall"

**Evidence.**

`USER_TEMPLATE` opens with:

```
QUESTION ({cie}, {question}, worth {q_marks} marks overall)
```

and it is filled from `record.get("question_marks", "?")`. But
`grade.py` never writes that key:

```
records carrying question_marks: 0/740
keys present: ['answer', 'answer_has_figures', 'booklet_id', 'chain',
               'cie', 'item_index', 'marks_available', 'model_solution',
               'point', 'question', 'question_text', 'why']

'question_marks' in grade.py: False
```

`question_marks` appears exactly once in the whole of `src/` — in the
template's own `.get()` default. So every prompt in the recorded Colab
run, and every prompt a local run will send today, reads:

```
QUESTION (CIE 2, 1, worth ? marks overall)
```

**Why it matters.** Rule 5 of the system prompt turns on marks:
*"Partial credit only where the item's marks allow it... unless the item
is worth more than 1 mark"*. The item's own marks **are** supplied
correctly (`marks_available`, in the rubric-item header), so the rule is
followable. What is missing is the question's total, which is the
context for judging whether one rubric item is a small part of a large
answer or most of a small one. The model was asked to weigh that and
handed a question mark.

**Why it is not simply fixed here.** Fixing it changes the prompt, and
the prompt is the one thing this edition deliberately holds byte-identical
so that `--compare` measures the backend rather than the wording. A fix
is a *new experiment*, not a patch: populate `question_marks` in
`grade.py`'s `queue_rows()` from `spec`, re-run the tier, and compare
the three runs. That is worth doing and should be done knowingly.

**Cost of fixing:** one line in `grade.py`, then a full re-run of the
tier — which on this machine is the expensive part, not the line.

### 4.2 `apply_verdicts.py` still says the verdicts come from a notebook

**Evidence.**

```
src/apply_verdicts.py:9:   The notebook's prompt tells the model...
src/apply_verdicts.py:319: help="verdicts.jsonl from the notebook"
```

Both were true when written and are now half-true: the file accepts
verdicts from `llm_local.py` as readily as from the notebook, and does
not care which produced them — which is the property that made the port
cheap and is worth stating rather than contradicting.

**Why it matters.** Only to a reader, but this repo's docstrings are
load-bearing: they are how the design is transmitted, and `apply_verdicts.py`
carries the single most important rule in the project. A stale docstring
on that file costs more trust than the same staleness anywhere else.

**Cost of fixing:** two lines, no behaviour change. Left as found so
that the port's diff stays honestly minimal — every file this edition
touched, it touched for a reason that shows up in behaviour.

---

## Deliberately not suggested

In the spirit of part 1's *"Deleted, and why — do not resurrect without
new evidence"*, these are things a reviewer might reach for that the
evidence here argues against:

- **Replacing the ladder with a single strong model.** The whole project
  is the argument against this, and the numbers support it: 32% of items
  settled with no model at all, and 15.4% of the model's awards
  fabricated their evidence.
- **Loosening thresholds to settle more marks.** `calibrate.py`'s
  selection rule already rejects this trade. Decisiveness saves human
  time; over-settling is irreversible.
- **Collapsing `[settled, settled+pending]` to a single number** to get
  a cleaner headline. It would report 67% of marks as zeros.
- **Tuning against the examiner until agreement is maximised.** Seven of
  fifty covers are demonstrably defective. Perfect agreement would mean
  reproducing four Part C policy errors and an arithmetic slip.
- **Transcribing the diagrams to text so cheap tiers can read them.**
  Part 1 rejected this with reasons — linearising a grid invents a
  reading order the page does not have.
