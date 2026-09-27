# Major Project — handwritten answer scripts, read and marked

**Branch: `segmentation-regrouping`.**

61 students' handwritten Computer Networks booklets (3 CIEs, 153
booklets). **Part 1** reads each booklet into `question → part → answer`
plus the drawings. **Part 2** marks that against the department's three
CIE answer schemes and compares the result with the marks the faculty
wrote on the covers. The two halves are stitched: one command takes
Part 1's output to final marks, and a local website is where a person
finishes what the machine could not.

---

## Run it

```bash
cd Major_Project_Eval_Local                 # or Major_Project_Eval (Colab model tier)
uv venv && uv pip install -e ".[semantic]"  # once

export MPE_HANDOFF=/path/to/handoff1        # not needed if handoff1/ sits at the repo root
.venv/bin/python src/render_scheme.py       # once: scheme PDFs -> PNG
.venv/bin/python src/run_all.py             # part 1's handoff -> marks, end to end
.venv/bin/python src/serve.py               # the website: http://127.0.0.1:8000
```

`run_all.py` runs every Part 2 step in the one order that is right:
load the handoff → resolve unlabelled parts → the marking ladder → the
model tier's verdicts → human decisions → agreement with the examiner.
It stops at the first step that fails.

### The website

`serve.py` binds to 127.0.0.1 only, because the pages quote real students.

- **Worklist.** The index shows items decided, items left and booklets
  outside the examiner's total. You can filter to *still undecided* or
  *outside the examiner's total*, and **Continue reviewing** opens the
  first booklet with work left.
- **One booklet per page.** Every question shows the student's answer,
  their drawings (click to enlarge), each rubric item with the tier
  that decided it and its evidence, and the examiner's mark beside ours.
- **Marking.** Every item defaults to *keep*. Choose 0 / half / full or
  type a mark, then **Save** or **Save & next undecided**. Keyboard:
  `n` next undecided, `p` previous, `h` hide questions already decided.
- **Nothing is lost.** Every decision is appended to
  `output/human_marks.jsonl` before the marks change, and `run_all.py`
  replays it after a re-grade.
- **Export.** *download marks CSV* and *per-question CSV* give the marks
  as they stand right now, including decisions made in the browser.

---

## Where the data comes from

The scanned pages live in two **private Hugging Face dataset repos**.
[`docs/HUGGING_FACE.md`](docs/HUGGING_FACE.md) explains the format and
the ways to import them. In short:

```bash
Major_Project_Eval_Local/.venv/bin/python scripts/fetch_hf.py --list
Major_Project_Eval_Local/.venv/bin/python scripts/fetch_hf.py --students 1-5
```

The token goes in `.env` (`HF_TOKEN=...`). Cover pages and the name/USN
roster are skipped unless you ask for them.

Two other inputs are deliberately not in git:

- **Part 1's handoff** (`handoff1/`): every student's transcribed
  answers and page crops. `paths.py` finds it at the repo root, or wherever
  `MPE_HANDOFF` points.
- **The three scheme PDFs**: the department's material. Put them at
  `answer_keys/`, or point `MPE_SCHEMES` at them.

---

## Folders

| folder | what it is | read first |
|---|---|---|
| [`Major_Project_Eval_Local/`](Major_Project_Eval_Local/) | Part 2, model tier on this machine (Ollama, or read in session) | [HANDOFF.md](Major_Project_Eval_Local/HANDOFF.md) |
| [`Major_Project_Eval/`](Major_Project_Eval/) | Part 2, same code, model tier on a free Colab T4 | [HANDOFF.md](Major_Project_Eval/HANDOFF.md) |
| [`scripts/`](scripts/) | `fetch_hf.py`: import the pages from Hugging Face | [docs/HUGGING_FACE.md](docs/HUGGING_FACE.md) |
| [`extra/`](extra/) | Part 1's archived reading/assembly modules, experiments, old plans | `extra/README.md` |

The two Part 2 editions share their code byte-for-byte. They differ
only in where the model tier runs.

---

## Where it stands

1,221 rubric items across 300 counted questions in 50 booklets:

- **946 decided**:
  - 383 by the deterministic ladder (exact, keyword and semantic matching, plus answers left unattempted);
  - 563 by the model tier.
- **275 waiting for a person** in the website: mostly drawing-heavy answers, chain questions and parts with a lost page.

The model tier on this machine was read **in session by Claude**, under
the same prompt and the same checks as the recorded Qwen run.
`apply_verdicts.py` re-checked every award's quote against the student's
text and rejected none. 95 awards rest on a drawing and could not be
quote-checked; the website labels them "not quote-checked" for spot
checks. Every mark records which reader gave it.

Against the examiner on 289 comparable questions: **150 inside our range
(52%), 31 above what the examiner gave, 108 below**. The recorded Qwen
run reached 79%, but with far more items left open, and every open item
widens the range being compared. Our marking follows the scheme item by
item; the examiner is visibly more lenient (e.g. 3/5 for a Q1 answer
with client and server reversed). The examiner's covers are a reference,
not ground truth: 7 of 50 carry a demonstrable defect (see each
folder's `gold/gold_notes.md`).

One open question on the scheme itself: CIE 3 Q3a's key gives NRZ-I at
10 Mbps as 500 kbaud / 500 kHz, but N/2 of 10 Mbps is 5 Mbaud / 5 MHz.
Check it against the scheme PDF.

---

## What is not tracked, and why

No student writing is in this repo, and `.gitignore` enforces it rather
than relying on care. All of each Part 2 folder's `output/` is ignored,
because every file there quotes a student. The same goes for `handoff1/`, the
Hugging Face downloads (`dataset/`, `dataset_cleaned/`, `data_hf/`),
`answer_keys/`, `.env` and the page-bearing folders under `extra/`. The
single exception is `gold/gold_marks.csv`: pseudonymous integers, no
names or handwriting.
