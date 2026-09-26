# Marking handwritten exam booklets against the official schemes

Part 2 of a two-part project. Part 1 read 50 handwritten Computer
Networks booklets into structured `question → part → answer` data. This
half marks them against the department's answer schemes, and measures
the result against the marks the faculty actually awarded.

**It pays for no inference.** Everything here runs on a laptop except
one tier, which runs on a free Colab T4.

---

## The result

On **289 questions across 49 booklets**, comparing our marks with the
examiner's:

| | |
|---|---|
| Examiner's mark still reachable | **229 (79%)** |
| Over-settled — we awarded marks they did not | **21** |
| Under-settled — we ruled out marks they gave | **39** |
| Irreversible error rate | **20.8%** |

Of the 48 booklets carrying an examiner total, **39 have that total
inside our range**. The nine that do not are all short, by one to four
marks, and one of them (`student_45_cie_1`) is a booklet where the
examiner did not apply the paper's own choice rule.

These numbers are **after** the model tier ran. Before it, 94% of
questions were reachable — but only because almost nothing was settled,
and an interval that spans everything agrees with anything. The honest
comparison is this one.

Marks are compared as an **interval**, not a number: most rubric items
are still queued to the model tier or a human, and collapsing that to a
point would silently treat every undecided item as a zero.

### The examiner is a reference, not a ground truth

Seven of the fifty covers carry a demonstrable defect — four Part C
totals that ignore the paper's own choice instruction, one arithmetic
slip, one grid with no totals, one with no marks at all. **A grader that
matched this reference perfectly would be reproducing its mistakes.**
Where we differ, the finding is "we differ, and here is why", and on the
four choice-policy booklets the answer is that we are right. See
[`gold/gold_notes.md`](gold/gold_notes.md).

---

## How it marks

Each rubric item is decided by the cheapest tier that can honestly
decide it, and every award records which tier decided and on what
evidence.

| tier | decides | runs on |
|---|---|---|
| `exact` | items naming concrete values — `57088`, `/26`, `14.24.74.126` | laptop |
| `keyword` | items where coverage is decisive either way | laptop |
| `semantic` | MiniLM breaking ties on keyword-supported items | laptop (CPU) |
| `llm` | everything still ambiguous | free Colab T4 |
| `human` | anything whose evidence is a drawing | a person, via `serve.py` |

Current split of the 1,221 items on counted questions: **23% decided with
no model and no human**, 246 settled by the model, and 599 still open —
the great majority of those because the model's verdict was refused (see
below) and the item now needs eyes on a drawing.

### Three rules that hold the thing together

**1. A cheap tier may award on evidence it finds, but is restrained from
concluding absence.** Finding the evidence is proof; failing to find it
is not. Four carve-outs, each written against a measured failure:

- **chain questions** (subnetting, fragmentation, CRC, the delay
  cascade) are never zeroed cheaply — a wrong block size at step one
  makes every later value legitimately differ, and zeroing them charges
  one slip five times
- **a part containing a lost page** is incomplete, not wrong
- **an answer whose evidence is a drawing** cannot be zeroed on prose —
  on `student_01_cie_2`, question 2b's entire routing table sits in four
  crops behind 73 characters of text; the examiner gave 5/5 and marking
  the prose alone scored 0
- **a long answer with no keyword overlap** is more likely our
  vocabulary missing theirs than the student saying nothing

**2. Similarity may confirm evidence. It may never supply it.** An item
whose keywords are absent is not decided by the embedding tier no matter
how topical it reads — every wrong subnet is about subnetting. The same
rule applies to exact values: `student_23_cie_3` wrote 265 words about
Fletcher checksums without one number from the scheme and briefly scored
3/3 on topical similarity while the examiner gave zero.

**3. The model may not award a mark it cannot quote the student's own
words to support, and may not zero what it cannot see.** `apply_verdicts.py` re-checks every quote against
the answer and discards awards whose quote is not there. The check is
downstream of the model and does not trust it — tested against
fabricated quotes, mangled paraphrases and over-awards, all rejected.

The first real run is what this rule is for. Qwen2.5-7B returned 169
awards; **26 of them (15.4%) cited a quote that is not in the answer**
and were discarded. That number is a property of the model, not of the
corpus, and no prompt would have produced it honestly.

The same run showed a prompt is worth as little in the other direction.
The model is told, in as many words, that it is never shown the
drawings, and that if the evidence would be in one it must decline. It
obeyed on 55 of 438 drawing-backed items and **zeroed 295 of them**. It
is told the scheme's carry-forward note on chain questions and told to
mark against the student's own earlier values; on CIE-2 3b it zeroed all
five organizations of two booklets against the scheme's absolute
addresses, where the examiner gave 10/10 and 6/10.

So the carve-outs above are enforced against the model too, in
`zero_blocked`: **368 of its 471 zeros were refused** and sent to a
human rather than settled. Its awards are untouched by this — finding
evidence in the text is still proof, and still has to survive the quote
check. Only its silence is disbelieved. Refusing those zeros moved
under-settled questions from 167 to 38.

### Part C is a choice

`max(3a, 3b)` and `max(4a, 4b)` — the better half, never the sum, as the
paper instructs.

---

## Running it

```bash
uv venv && uv pip install -e .            # laptop tiers only
uv pip install -e ".[semantic]"           # add the embedding tier

python src/render_scheme.py               # scheme PDFs -> PNG
python src/validate_keys.py               # rubric arithmetic + structure
python src/make_verify_sheet.py           # the human sign-off sheet

python src/crop_covers.py                 # cover marks grids (identity excluded)
python src/gold_check.py                  # the examiner's own arithmetic

python src/load_handoff.py --report       # must match part 1's totals
python src/align.py --report              # parts with no usable label

python src/grade.py --all                 # the ladder
python src/grade.py --all --no-semantic   # ...without torch
python src/calibrate.py                   # sweep thresholds vs the examiner
python src/agreement.py                   # the deliverable

python src/make_llm_notebook.py           # -> notebooks/grade_llm.ipynb
python src/apply_verdicts.py output/grade_llm_verdicts.jsonl
python src/apply_human.py                 # replay the human tier's log

python src/serve.py --check               # the crops are really on disk
python src/serve.py                       # the human tier, one booklet a page
```

Every stage carries its own check, and they are ordered so a failure
surfaces before it contaminates the next one. There is no test suite;
`--report`, `--check` and `--dry-run` are the verification.

### Where the data lives

The 187 MB corpus is **referenced in place, not copied** — override with
`MPE_HANDOFF` / `MPE_SCHEMES`. Nothing derived from a booklet is
committable: all of `output/` is gitignored, because every file this
project writes quotes a student to justify itself. The single exception
is `gold/gold_marks.csv` — pseudonymous integers, no names or
handwriting, and the one artifact that cannot be regenerated by running
anything.

---

## What the numbers do not cover

- **The rubric is hand-authored** from skewed phone scans, because the
  scheme PDFs contain zero font objects and cannot be parsed. It is the
  ceiling on everything downstream. `keys/VERIFY.md` is the sign-off.
- **Our inferred mark splits are measurably worse than the scheme's
  printed ones** — 89% vs 96% containment. Where the scheme gave a total
  and no breakdown, the division into rubric items is our judgement, and
  that seven-point gap is what it costs.
- **All five under-settled questions are content we could not locate**,
  not marks we misjudged. No threshold fixes that; they are queued to a
  human.
- **The semantic tier barely earns its place.** Calibration over 1,600
  settings drives it to almost zero at every low-error operating point —
  it was the main source of over-awards, and constrained not to
  over-award it now fires on 6 items out of 1,221.
- **67% of marks are still pending.** The model tier has run; most of
  what it returned was refused rather than accepted, so the pending pile
  barely moved. **226 questions holding 599 items** now wait on a human,
  162 of them carrying a drawing. `serve.py` is the tool for that, and
  until somebody works through it these numbers do not move.
- **The nine unreachable booklets are the cost of settling anything.**
  While nothing was settled our range spanned every possible mark and
  contained the examiner by construction. It no longer does, and nine
  totals now sit above our ceiling. That is the measurement starting to
  say something rather than the measurement getting worse.
- **The model tier is the weakest link, and it is measurable.** 15.4% of
  its awards cited a quote that does not exist, and 63% of its zeros were
  on answers it was structurally unable to read. Both numbers are in
  `output/verdict_audit.md`, and both argue the same thing: this tier is
  useful only with the checks around it.
