# Glossary

This repo uses a small private vocabulary very precisely, and several of
the terms mean something narrower than they sound. Where a word is doing
load-bearing work, the entry says what breaks if you read it loosely.

---

### backend

Whatever actually runs the model at the `llm` tier. Two exist here:
`ollama` (HTTP to a local server, the default) and `llama-cpp` (a GGUF
file in-process). The backend is recorded in every verdict's `model`
field, because the quantisation and the runtime are part of what
produced the verdict rather than incidental to it.

### booklet

One student's answers for one CIE. The unit of almost everything:
`student_07_cie_2`. 50 booklets here; part 1 prepared 153.

### carve-out

A case where a cheap tier is **forbidden from recording a zero**, though
it may still award. Four of them — chain question, lost page, drawing,
long answer with no overlap. Passed down the ladder as a `no_zero`
reason *string*, not a boolean, so the item's `why` field can say which
one stopped it. See [`ARCHITECTURE.md` §4](ARCHITECTURE.md).

### chain question

A question whose marks hang off a sequence: subnetting, fragmentation,
CRC, the delay cascade. A wrong value at step one makes every later
value legitimately differ from the scheme. 7 of 24 questions carry a
`chain` note, which ships into the model prompt as a carry-forward
instruction: mark against the student's own preceding values, not the
scheme's absolutes.

Zeroing a chain question cheaply charges one slip five times over.

### choice pair

Part C is answered by internal choice. `3a`/`3b` and `4a`/`4b` are
pairs, and the pair is worth `max()` of the two — **the better half,
never the sum**. The loser gets `counted: false`.

### `choice_provisional`

The loser of a choice pair still has enough *pending* marks to overtake
the winner. The choice is not final until those items are decided.

### CIE

Continuous Internal Evaluation — the departmental term for one of the
three internal exams. `cie_1`, `cie_2`, `cie_3`, 40 marks each.

### confidence (on a gold row)

Whether a **human** has actually looked at that cover crop, as distinct
from whether the arithmetic reconciles. `no_gold` marks the cover with
no marks written on it at all.

### `counted`

Survived Part C's choice. Only counted questions contribute to totals,
queues and agreement.

### decisive / decisiveness

The fraction of marks settled without the model or a human. A
calibration axis, traded against irreversible errors. **Being more
decisive is not better on its own** — settling more only saves human
time, while settling wrongly cannot be undone.

### `decline`

A first-class verdict from the model tier, not a failure. It means the
evidence would be in a drawing the model was never shown. It routes the
item to a human holding the crop.

### evidence

The alternates and exact values that actually matched — recorded on
every decided item so an award can be explained without re-running
anything.

### exact (the field)

Literal values that are right or wrong with no interpretation: `57088`,
`/26`, `14.24.74.126`. Tier 1 decides on these with no model.

**Only distinctive values belong here.** `"2"`, `"88"` and `"ok"` occur
by accident in prose about anything. `validate_keys.py` reports every
short numeric `exact` as a warning so each one is a value a person
decided to trust. If a correct answer has no distinctive literal form,
leave `exact` empty and let it go up the ladder — that is the ladder
working, not a gap.

### gap

A marker from part 1 meaning a page was lost before it was ever read
(`s19_c2_p14`). **A gap means incomplete, not short.** A part containing
one cannot be cheaply zeroed.

### GGUF

The file format llama.cpp uses for quantised weights. The local model
tier reads one of these instead of loading the model through
`transformers`, because 4-bit `bitsandbytes` needs CUDA and this project
assumes no GPU. Same Qwen2.5 weights, different container — which is why
local and Colab verdicts are comparable but not identical.

### gold

`gold/` holds the examiner's marks. The name is historical and the
directory's own notes say to treat it as a filename, not a claim: it is
a **reference, not a ground truth**. Seven of fifty covers carry a
demonstrable defect.

### handoff

Part 1's output: every booklet as `question → part → answer`, with its
page images and drawing crops. `python pipeline.py handoff` writes it to
`data/handoff/`, which is gitignored like the rest of `data/`. Set
`MPE_HANDOFF` to mark a handoff that lives elsewhere. The contract is
[`../../docs/HANDOFF.md`](../../docs/HANDOFF.md).

### inside

The examiner's mark lies within `[settled, settled + pending]`. **This
is not agreement** — it is the absence of a contradiction. `agreement.py`
says so in its own output, and the distinction matters because an
interval that spans everything agrees with anything.

### irreversible error

An over-settle or an under-settle. Named this way because no later tier
can fix either: a mark already awarded cannot be taken back by the human
tier, and a mark already ruled out is never revisited. They are the
calibration target for exactly that reason.

### `num_ctx`

The context window the local backend is given, fixed at 4096. Named here
because the default is a trap: Ollama's own default is 2048, the longest
prompt in the queue is about 1,825 tokens plus a 300-token reply, and a
truncated prompt loses the **end** of the user turn — which is the
student's answer. The model would then mark an answer it was never shown
and return a confident `zero`.

### item / rubric item

The atom of marking: one claim the student had to make, with its own
marks. 107 across the three keys; 1,221 item-decisions across the 50
booklets' counted questions.

### ladder

The ordered tiers — `exact`, `keyword`, `semantic`, `llm`, `human` —
each trusted only where it is reliable. Ordered by cost, but **cost is
not the point of the order**.

### `missing_but_marked`

The examiner awarded marks for a question and this project has no
content filed under it. Reported as a location failure — a bug in us —
never scored as a zero for the student.

### over-settled

We have already awarded marks the examiner did not. `settled >
examiner`. Irreversible.

### pending

Marks on items no tier has honestly decided yet — `awarded: null`.
**Not zero.** Collapsing pending to zero is the single most damaging
misreading of this project's output.

### printed vs inferred (`breakdown_source`)

Where the scheme printed its own mark breakdown, it was copied
(`printed`, 18 questions). Where it did not, the division into rubric
items is this project's judgement (`inferred`, 6 questions).
`agreement.py` reports the two separately so the cost of inferring shows
up rather than hiding — measured at 89% vs 96% containment, and that
seven-point gap is what inferring costs.

### quote (on an award)

The student's own words, verbatim, supporting a mark. Required from the
model tier and re-checked downstream by `apply_verdicts.py`. 15.4% of
the model's awards cited a quote that was not in the answer.

### reachable

Synonym for *inside*: the examiner's mark has not been ruled out.

### scheme

The department's official answer key, as a PDF of phone scans. **Zero
font objects** — no text layer, nothing to parse, which is why the
rubric is hand-authored.

### settled

Marks a tier has decided, in either direction. Reported as a **lower
bound** on a booklet's total, because every pending item is excluded
rather than assumed zero.

### tier

Who decided an item, recorded on the item itself. `exact`, `keyword`,
`semantic`, `llm`, `human`, or `unattempted`.

### `unattempted`

No content filed under the question at all. A real decided zero, not a
pending one — and distinct from `missing_but_marked`, where the examiner
says there *should* be content.

### under-settled

We ruled out marks the examiner gave. `settled + pending < examiner`.
Irreversible.

### `zero_blocked`

The mechanism in `apply_verdicts.py` that refuses a model `zero` on an
item the model was structurally unable to judge — the evidence is in a
drawing it never saw, or a part that lost a page. **368 of the model's
471 zeros were refused this way.** Its awards are untouched; only its
silence is disbelieved.

### `zero_max_chars`

How long an answer may be before a cheap tier is allowed to call it
empty. Past that length, zero keyword overlap is far more likely to mean
this project's vocabulary missed the student's than that the student
said nothing. Measured on CIE-2 2a, where seven students wrote 270–654
accepted characters that the rubric's keywords scored 0.00 on.

### recorded run

`data/marking/grade_llm_verdicts.jsonl` — the Colab run of the model tier that
every published number in this repo comes from. Kept intact. A local run
writes `data/marking/local_verdicts.jsonl` beside it, and `llm_local.py`
refuses to append one model's verdicts to another's file.
