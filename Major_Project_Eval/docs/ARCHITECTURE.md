# Architecture

How this project marks a handwritten exam booklet, and why it is shaped
the way it is.

---

## 1. What the thing actually is

This is **part 2 of a two-part project**. Part 1 (`Major_Project_Iter1`)
read 50 handwritten Computer Networks booklets into structured
`question → part → answer` data using a vision model. This half takes
that output and marks it against the department's official answer
schemes, then measures the result against the marks a faculty member
actually awarded.

The whole problem, in numbers:

| | |
|---|---|
| booklets marked | 50 (49 with a usable examiner mark) |
| papers | 3 CIEs, 8 questions each |
| rubric questions | 24 |
| rubric items across the three keys | 107 |
| item-decisions on counted questions | **1,221** |
| source lines | ~4,400 across 17 files |

1,221 is the real size of the task. Every one of those is a decision
about whether one student made one specific claim, and the entire design
follows from the fact that most of them cannot be made reliably by the
same mechanism.

### It pays for no inference

Everything runs on a laptop except one tier, which runs on a free Colab
T4. There is no training, no fine-tuning, no API key and no GPU
dependency in the default path. `pyproject.toml` enforces the split:
the base install is `pymupdf`, `pillow` and `numpy`, and the embedding
model is an optional extra.

---

## 2. The central idea: a ladder, not a model

The obvious design — hand each answer and each rubric item to a language
model and ask "did they say this?" — is the one this project rejects,
and the rejection is the design.

Instead, **each rubric item is decided by the cheapest tier that can
honestly decide it**, and every award records which tier decided and on
what evidence.

| tier | decides | runs on | can it zero? |
|---|---|---|---|
| `exact` | items naming concrete values — `57088`, `/26`, `14.24.74.126` | laptop | yes, conditionally |
| `keyword` | items where term coverage is decisive either way | laptop | yes, conditionally |
| `semantic` | MiniLM breaking ties on items that already have keyword support | laptop, CPU | **never** |
| `llm` | everything still ambiguous | free Colab T4 | only where it could see |
| `human` | anything whose evidence is a drawing | a person, via `serve.py` | yes |

The tiers are ordered by cost, but **cost is not the point of the
order**. Each tier is trusted only in the region where it is actually
reliable, and the interesting part of this codebase is where those
regions end.

### Current distribution

From the most recent run (2026-09-20), across 1,221 items on counted
questions:

| | decided | pending |
|---|---|---|
| **total** | **634 (52%)** | **587 (48%)** |
| `llm` | 246 | 482 |
| `keyword` | 141 | — |
| `exact` | 139 | — |
| `unattempted` | 90 | — |
| `human` | 12 | 105 |
| `semantic` | 6 | — |

Two things to read off that table. First, **390 items — 32% — were
settled with no model and no human at all**, by string matching against
a hand-authored rubric. Second, the pending pile is still almost half
the corpus, and the root README is blunt about why: most of what the
model returned was refused rather than accepted.

---

## 3. The asymmetry that runs through everything

This is the single idea that, once seen, explains most of the code:

> **A cheap tier may always AWARD on evidence it finds, and is only ever
> restrained from concluding absence.**
>
> Finding the evidence is proof. Failing to find it is not.

A keyword match that fires is positive information: those characters are
in the student's answer, and the only question is whether they mean what
the rubric wanted. A keyword match that does *not* fire is almost no
information at all — the student may have used different words, written
the answer in a diagram, or had their page lost before it was scanned.

So awarding and zeroing are handled by different rules, and the code
passes a `no_zero` reason string down the ladder rather than a boolean.
When a tier is forbidden from zeroing, it says which of the carve-outs
below is stopping it, and that reason is written into the item's `why`
field and survives into the review queues.

---

## 4. The four carve-outs

Each is a case where absence of evidence in the text is *not* evidence
of absence in the answer. Each was written against a specific measured
failure, and the code names the booklet it came from.

### 4.1 Chain questions are never zeroed cheaply

Subnetting, fragmentation, CRC and the delay cascade all hang off a
sequence. A student who takes a wrong block size at step one produces
values that are correct *relative to their own error* and absent from
the key. Their `14.24.74.63` is missing from the scheme not because they
failed but because they are consistently following their own arithmetic.

Zeroing those charges one slip five times over. The key marks these
questions with a `chain` note (7 of 24 questions carry one), and they go
up the ladder where carry-forward can be judged.

### 4.2 A part containing a lost page is incomplete, not wrong

One page of the corpus (`s19_c2_p14`) was lost before it was ever read.
A part containing a `gap` is missing content that existed. Scoring it as
a thin answer would penalise a student for the project's own data loss.

### 4.3 An answer whose evidence is a drawing cannot be zeroed on prose

This is the one that was measured rather than imagined. On
`student_01_cie_2`, question 2b's entire routing table sits in four
image crops behind 73 characters of prose, and half of question 1's
answer was written onto the handshake diagram itself. **The examiner
gave 5/5 for both. Marking the text alone scored them 0 and 2.**

Nothing below the human tier reads pixels. Part 1 deliberately never
transcribed the drawings, because linearising a grid invents a reading
order the page does not have. So a drawing is carried as a path and a
page reference, and any item whose evidence is one goes straight to a
human holding the actual crop.

### 4.4 A long answer with no keyword overlap is our vocabulary failing

Measured on CIE-2 2a: seven students wrote 270–654 characters that the
examiner accepted, and the rubric's keywords scored 0.00 on all of them,
because the students described a lost acknowledgment instead of naming
the RTO. `zero_max_chars` caps how long an answer may be before a cheap
tier is allowed to call it empty.

### 4.5 And one more: marked but not found

If the examiner awarded marks for a question and this project has no
content filed under it, that is reported as a location failure — a bug
in us — not scored as a zero for the student.

---

## 5. Two rules that hold the tiers apart

### Similarity may confirm evidence. It may never supply it.

Sentence embeddings measure what an answer is *about*, not whether it is
right. Every wrong subnet is about subnetting; every mangled handshake
is about handshakes. Set loose on the CIE-2 3a rubric, one sentence of
plausible subnetting prose scores respectably against all five rubric
items at once — including the four it says nothing about.

So the semantic tier refuses to decide an item whose keywords are
absent, no matter how topical it reads, and refuses to rescue an item
naming exact values when the student named none of them.
`student_23_cie_3` wrote 265 characters about Fletcher checksums without
one number from the scheme and briefly scored 3/3 on topical similarity
while the examiner gave zero. That measurement is why the rule exists.

The tier also **never records a zero**. Low similarity might mean an
unusual phrasing, or a sentence splitter that cut the claim in half.

The result is a tier that barely fires — 6 items out of 1,221 — and the
root README says plainly that it barely earns its place.

### The model may not award a mark it cannot quote

The notebook prompt tells the model it must quote the student's own
words to support any award. **A prompt is a request; `apply_verdicts.py`
is the enforcement.** Every returned quote is looked for in the answer,
and an award whose quote is not actually there is discarded and the item
returned to the queue.

The check is deliberately downstream of the model and does not trust it.
On the first real run, Qwen2.5-7B returned 169 awards and **26 of them
(15.4%) cited a quote that is not in the answer.**

Matching is forgiving about form — whitespace, case, the dash zoo, the
model's own wrapping quotation marks — and strict about content. No
fuzzy matching, because "nearly there" is exactly how a paraphrase
passes.

The same run showed a prompt is worth as little in the other direction.
The model is told in as many words that it is never shown the drawings
and must decline where the evidence would be in one. It obeyed on 55 of
438 drawing-backed items and **zeroed 295 of them**. So the carve-outs
are enforced against the model too, in `zero_blocked`: **368 of its 471
zeros were refused** and sent to a human. Its awards are untouched by
this — only its silence is disbelieved.

---

## 6. Part C is a choice

`max(3a, 3b)` and `max(4a, 4b)` — the better half, never the sum, as the
paper instructs. A student who answered both has not earned twice the
marks.

The loser of the pair is marked `counted: false` and excluded from
totals, queues and agreement. Where the loser still has enough *pending*
marks to overtake the winner, the choice is flagged
`choice_provisional` — it is not final until those items are decided.

The examiner deviated from this on four booklets, totalling both halves.
That disagreement is reported rather than reproduced, and on those four
this project is right by construction.

---

## 7. The measurement

### The examiner is a reference, not a ground truth

Seven of the fifty covers carry a demonstrable defect: four Part C
totals that ignore the paper's own choice instruction, one arithmetic
slip, one grid with no totals, one with no marks at all. **A grader that
matched this reference perfectly would be reproducing its mistakes.**

So the question is never "did we match?" but "where do we differ, and
which of us is right?"

### Marks are compared as an interval, not a number

Most rubric items are still queued. A question therefore has no single
mark — it has `settled` (decided by the cheap tiers) and `pending` (not
yet decided). Reporting a point estimate would silently treat every
undecided item as a zero.

Each question is compared as `[settled, settled + pending]`:

```
settled <= examiner <= settled + pending     inside — no contradiction
settled > examiner                           OVER-settled  — irreversible
settled + pending < examiner                 UNDER-settled — irreversible
```

Only the last two are errors, and both are irreversible: no later tier
can take back a mark already awarded, or recover one already ruled out.
**"Inside" is not agreement — it is the absence of a contradiction**,
and `agreement.py` says so in its own output.

### The headline

On 289 comparable questions across 49 booklets: 80% keep the examiner's
mark reachable, 21 over-settled, 38 under-settled, irreversible error
rate 20.4%. The examiner's own covers carry a defect on 14% of booklets,
which is the floor this is measured against.

Before the model tier ran, 94% of questions were reachable — but only
because almost nothing was settled, and an interval that spans
everything agrees with anything. The honest comparison is the one after.

---

## 8. Module map

```
        keys/cie*.json ──┐
   (hand-authored rubric)│
                         │
   handoff (part 1) ─────┼──> grade.py ──> output/marks/*.json
   187 MB, referenced    │        │        output/summary.csv
   in place, never copied│        │        output/queue_llm.jsonl
                         │        │        output/queue_human.jsonl
   output/alignment.json─┘        │
   (resolved odd labels)          │
                                  ├──> make_llm_notebook.py ──> Colab
                                  │         verdicts.jsonl ──┐
                                  │                          │
                                  │    apply_verdicts.py <───┘
                                  │      (quote check, zero_blocked)
                                  │          └──> updated marks
                                  │               verdict_audit.md
                                  │
                                  ├──> serve.py  (the human tier)
                                  │          └──> updated marks
                                  │               human_marks.jsonl
                                  │
   gold/gold_marks.csv ───────────┴──> agreement.py ──> agreement.md
   (the examiner's grid)               calibrate.py  ──> calibration.md
```

### By file

**Setup and verification — run before anything depends on them**

| file | LOC | role |
|---|---|---|
| `paths.py` | 59 | Every path in one place. The handoff is referenced in place, never copied; `MPE_HANDOFF` / `MPE_SCHEMES` override. |
| `render_scheme.py` | 130 | Scheme PDFs → PNG at 200 dpi. The PDFs contain **zero font objects** — phone scans, no text layer, nothing to parse. |
| `validate_keys.py` | 276 | The rubric checked against itself: items sum to the question, questions sum to the paper, `choice_with` reciprocated, `exact` values audited for distinctiveness. |
| `make_verify_sheet.py` | 194 | Generates `keys/VERIFY.md`, the human sign-off. `validate_keys` proves the arithmetic; this proves the judgement. |

**The examiner's marks**

| file | LOC | role |
|---|---|---|
| `crop_covers.py` | 200 | Cuts the marks grid out of each cover **below the identity block** — the crop is the privacy control. Contact sheets so a clipped crop is caught by eye. |
| `gold_check.py` | 286 | The cover's own arithmetic. Each row has a total, the grid has a grand total, and the page repeats it in a separate box — two independent sums over every cell. |

**Loading and alignment**

| file | LOC | role |
|---|---|---|
| `load_handoff.py` | 278 | Part 1's booklets into `Booklet`/`Part` records. Three contract rules: `answer[]` not `part.text`, skip `excluded`, a `gap` means incomplete. |
| `align.py` | 410 | Resolves parts with no usable question label. Turned out 35 of 35 "unlabelled" parts were section headings — 11% of parts, **0% of the marks**. Five real cases resolved on three independent signals. |

**The marking**

| file | LOC | role |
|---|---|---|
| `tiers/__init__.py` | 58 | `Thresholds` — parameters, not constants, so `calibrate.py` sweeps the real decision logic rather than a copy of it. |
| `tiers/exact.py` | 260 | Tier 1. Normalisation keeps punctuation (the evidence *is* punctuation: `/26`, `14.24.74.126`). Word-boundary matching with a stemming rule for keywords and a strict rule for values. |
| `tiers/semantic.py` | 158 | Tier 2. MiniLM on CPU, ~90 MB. Confirms only. |
| `grade.py` | 562 | The ladder, the carve-outs, Part C resolution, the two queues. |

**The model tier**

| file | LOC | role |
|---|---|---|
| `make_llm_notebook.py` | 398 | Generates the Colab notebook. Queue records are self-contained — answer, rubric item, marks, chain note — so the notebook needs no access to this repo or the corpus. |
| `apply_verdicts.py` | 354 | The quote check, `zero_blocked`, and the audit. Does not trust the model. |

**The human tier and the report**

| file | LOC | role |
|---|---|---|
| `serve.py` | 630 | Localhost tool, one booklet at a time, with the drawings. A booklet is the unit because questions share pages, handwriting and the examiner's running judgement. |
| `agreement.py` | 305 | The deliverable. Intervals, not scores. |
| `calibrate.py` | 320 | Sweeps thresholds against the examiner. Reports a frontier, not a winner. |

---

## 9. Design decisions worth knowing about

**The rubric is hand-authored, and that is the ceiling.** The scheme
PDFs cannot be parsed, so 107 rubric items were transcribed by eye from
skewed phone scans. `keys/VERIFY.md` is the sign-off. Everything
downstream inherits any error here.

**Where the scheme printed its own mark breakdown, it was copied; where
it did not, the split is this project's judgement.** 18 of 24 questions
are `printed`, 6 are `inferred`, and `agreement.py` reports the two
separately so the cost of inferring shows up rather than hiding.

**Thresholds live in a frozen dataclass, not in module globals.** If the
ladder read them from globals, `calibrate.py` would need its own copy of
the decision logic, and the two copies would drift on the first change —
leaving a "calibrated" number that describes code nobody runs.

**Every stage carries its own check**, and the stages are ordered so a
failure surfaces before it contaminates the next one. There is no test
suite; `--report`, `--check` and `--dry-run` are the verification.

**Nothing derived from a booklet is committable.** All of `output/` is
gitignored, because every file this project writes quotes a student to
justify itself. The single exception is `gold/gold_marks.csv` —
pseudonymous integers, no names or handwriting, and the one artifact
that cannot be regenerated by running anything.
