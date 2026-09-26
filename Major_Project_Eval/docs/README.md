# Documentation

Added documentation for `Major_Project_Eval`. Nothing here changes
behaviour — these files describe code that already exists.

The root [`README.md`](../README.md) is the project's own account of
itself: what it does, what it measured, and what the numbers do not
cover. Read that first. These files sit underneath it and answer the
questions it does not have room for.

| file | answers |
|---|---|
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
