# Documentation

Documentation for `Major_Project_Eval_Local` — the edition whose model
tier runs on this machine rather than on a free Colab T4.

If you are looking for what makes this copy different, it is one new
file (`src/llm_local.py`), one extracted one (`src/llm_prompt.py`), one
preflight (`src/setup_local.py`), and
[`LOCAL_SETUP.md`](LOCAL_SETUP.md). Everything else describes code that
was already here and is unchanged.

The root [`README.md`](../README.md) is the project's own account of
itself: what it does, what it measured, and what the numbers do not
cover. Read that first. These files sit underneath it and answer the
questions it does not have room for.

| file | answers |
|---|---|
| [`LOCAL_SETUP.md`](LOCAL_SETUP.md) | How do I run the model tier here, what will it cost me in hours, and what does going local change? |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How does the marking actually work, and why is it built this way? |
| [`PIPELINE.md`](PIPELINE.md) | What do I run, in what order, and what does each step prove? |
| [`DATA_FORMATS.md`](DATA_FORMATS.md) | What is in each file this project reads and writes? |
| [`GLOSSARY.md`](GLOSSARY.md) | What does this repo mean by *settled*, *chain*, *gap*, *carve-out*? |
| [`../IMPROVEMENTS.md`](../IMPROVEMENTS.md) | What is wrong or unfinished, and what would fix it? |

Two documents that were already here and are worth reading in full:

- [`keys/SCHEMA.md`](../keys/SCHEMA.md) — the rubric format, and the rule
  for what may go in `exact`. Authoritative; `DATA_FORMATS.md` does not
  repeat it.
- [`gold/gold_notes.md`](../gold/gold_notes.md) — why the examiner's
  marks are a reference and not a ground truth, with the seven defective
  covers itemised.

## Where to start, by what you want

**"Explain the project in five minutes."**
Root `README.md`, then the ladder table in `ARCHITECTURE.md`.

**"I have to run this."**
`PIPELINE.md`. Note the environment variables — the corpus is not in
this repo and is referenced in place.

**"I have to run the model tier."**
`LOCAL_SETUP.md`, and run `python src/setup_local.py --time-it` before
anything else. On a machine with no NVIDIA GPU the full queue is tens of
hours, and a model too large for your RAM will not say so — it will swap
and look merely slow.

**"I have to defend the design."**
`ARCHITECTURE.md`, specifically *The asymmetry* and *The four
carve-outs*. Every one of them was written against a measured failure,
and each names the booklet it came from.

**"I have to extend it."**
`GLOSSARY.md` for the vocabulary, `DATA_FORMATS.md` for the shapes, then
`IMPROVEMENTS.md` — the first three entries are traps you would
otherwise walk into.

## A note on what is not in git

`output/`, `gold/covers/` and `keys/scheme_pages/` are gitignored, and
the handoff corpus is never copied into this repo at all. If you cloned
this and the directories are empty, that is correct and intended — see
the reasoning in [`.gitignore`](../.gitignore), which is itself written
as documentation.

The consequence for a reader: **the numbers quoted throughout these docs
came from one particular run** on the author's machine, and describe
files you may not have. They are dated where it matters.

One more consequence specific to this edition: those numbers came from
the **Colab** run of the model tier, which is preserved here as
`output/grade_llm_verdicts.jsonl`. A local run writes to
`output/local_verdicts.jsonl` and changes nothing above until it is
applied. The two are not expected to agree item for item — different
quantisation, different runtime — and `python src/llm_local.py
--compare` prints the confusion matrix rather than leaving you to
assume.
