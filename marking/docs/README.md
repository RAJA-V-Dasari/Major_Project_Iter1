# Marking documentation

[`../README.md`](../README.md) is this half's own account of itself:
what it does, what it measured, and what the numbers do not cover. Read
that first. These files sit underneath it and answer the questions it
does not have room for.

| file | answers |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How does the marking actually work, and why is it built this way? |
| [`PIPELINE.md`](PIPELINE.md) | What do I run, in what order, and what does each step prove? |
| [`DATA_FORMATS.md`](DATA_FORMATS.md) | What is in each file this half reads and writes? |
| [`GLOSSARY.md`](GLOSSARY.md) | What does this repo mean by *settled*, *chain*, *gap*, *carve-out*? |
| [`LOCAL_SETUP.md`](LOCAL_SETUP.md) | How do I run the model tier on this machine, what will it cost in hours, and what does going local change? |
| [`../IMPROVEMENTS.md`](../IMPROVEMENTS.md) | What is wrong or unfinished, and what would fix it? |

Two documents beside the data they describe are worth reading in full:

- [`../keys/SCHEMA.md`](../keys/SCHEMA.md): the rubric format, and the
  rule for what may go in `exact`. It is authoritative, and
  `DATA_FORMATS.md` does not repeat it.
- [`../gold/gold_notes.md`](../gold/gold_notes.md): why the examiner's
  marks are a reference and not a ground truth, with the seven defective
  covers itemised.

The contract this half reads, the handoff, is documented for both sides
in [`../../docs/HANDOFF.md`](../../docs/HANDOFF.md).

## Where to start, by what you want

**"Explain the project in five minutes."**
`../README.md`, then the ladder table in `ARCHITECTURE.md`.

**"I have to run this."**
`python pipeline.py mark` from the repo root, then `PIPELINE.md` for
what each step proves.

**"I have to run the model tier."**
Pick a route: Colab (`make_llm_notebook.py`), local (`LOCAL_SETUP.md`
first, then `python marking/src/setup_local.py --time-it`), or in
session (`claude_tier.py`). On a machine with no NVIDIA GPU the local
route takes tens of hours, and a model too large for your RAM will not
say so. It will swap and look merely slow.

**"I have to defend the design."**
`ARCHITECTURE.md`, specifically *The asymmetry* and *The four
carve-outs*. Each was written against a measured failure and names the
booklet it came from.

**"I have to extend it."**
`GLOSSARY.md` for the vocabulary, `DATA_FORMATS.md` for the shapes, then
`IMPROVEMENTS.md`. Its first three entries are traps you would
otherwise walk into.

## A note on what is not in git

Everything this half writes is under `data/marking/`, the handoff it
reads is under `data/handoff/`, and the scheme PDFs and their renders
are under `data/schemes/`. All of `data/` is gitignored, because every
file there quotes a student or is the department's material. If you
cloned this and those folders are empty, that is correct and intended.

The numbers quoted throughout these docs come from particular runs on
particular machines, on the September handoff. They are dated where it
matters, and they describe files you may not have.
