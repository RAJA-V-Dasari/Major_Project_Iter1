# CLAUDE.md

Guidance for Claude Code when working in this repository.

Handwritten exam booklets (61 students × 3 CIEs) are read into
structured answers by a vision model (`reading/`) and marked against the
department's schemes (`marking/`). One driver, `pipeline.py`, runs every
stage. Start with `README.md`. The runbook is `docs/PIPELINE.md`, the
contract between the halves is `docs/HANDOFF.md`, and the data rules are
in `docs/DATA.md`.

## Commands

```bash
python pipeline.py status                   # what exists under data/, and what next
python pipeline.py check                    # tests + every check needing no model/GPU
python -m unittest discover -s tests -t .   # the tests alone
python pipeline.py fetch --students 7 --cie 2           # pages from Hugging Face
python pipeline.py read                     # coverage of data/read/<engine>
python pipeline.py run --engine <engine>    # assemble -> handoff -> mark
python pipeline.py review                   # marking website, 127.0.0.1:8000
python reading/benchmark/ocr_bench.py --engine qwen7b --verbose   # reader CER
python reading/read/make_colab_notebook.py  # regenerate after editing the prompt
```

Every script also runs on its own and documents itself in its docstring.
Verification is `--check` / `--report` / `--dry-run` on the scripts, plus
`tests/test_pipeline.py`, which runs assemble → handoff → marking on
synthetic pages. Run `python pipeline.py check` before committing.

## Layout rules

- **Every path comes from `common/layout.py`.** Nothing hard-codes
  `data/...` or a stage folder; the marking half reaches it through
  `marking/src/paths.py`. Scripts find the repo root by walking up to
  `common/layout.py`, so they run from any working directory.
- **All generated data lives under `data/`** (or `$MP_DATA`), gitignored
  as one line. Never write student-derived files anywhere else.
- **`.env` holds the HF token** and is read by `common/layout.py`. Never
  print it, log it, or commit it.

## Architecture: what not to undo

**The load is asymmetric.** Only reading pages (and the optional diagram
pass) needs a GPU, and it runs elsewhere (Colab, Kaggle, a server)
behind `reading/read/readers.py`. Keep `reading/` and the marking base
install free of torch and transformers. `sentence-transformers` is an
optional extra; `grade.py --no-semantic` must keep working.

**The prompt has one home.** The reading prompt lives in
`reading/read/make_colab_notebook.py`, and `read_pages.py` and
`kaggle_run.py` import it. The marking prompt lives in
`marking/src/llm_prompt.py`. A prompt change invalidates prior scores.
Regenerate the notebook; `python pipeline.py check` fails if you do not.

**Structure comes from the reader's headings, split where each heading
is written.** `build_booklet.py` once filed whole pages under the last
heading on them; the tests hold the fix.

**Never ask a vision model for pixel coordinates.** Position comes from
words it read (the diagram pass's `anchor`), and pixels from
`reading/assemble/segment.py`. Over-crop on purpose.

**Marking is a ladder, not a model.** Each rubric item goes to the
cheapest tier that can honestly decide it. Two rules are load-bearing,
and each was written against a measured failure; read the docstrings
before loosening either: similarity may only *confirm* keyword evidence,
and the model may not award what it cannot quote, nor zero what it
cannot see (`apply_verdicts.py`, `zero_blocked`). `grade.py --all`
rewrites marks from scratch; `run_all.py` replays model verdicts and
human decisions after it.

**The handoff is a contract.** `reading/handoff/export.py` writes it and
`marking/src/load_handoff.py --report` checks it. Change its shape only
with `docs/HANDOFF.md` and both sides together.

## Privacy: non-negotiable

`page_01` of every booklet is the identity cover. It is opt-in at fetch
and excluded at every stage that could send a page off the machine, not
once at the start. A transcription *is* the student's answer. Before
any broad `git add`, check what is staged: `data/` must never appear,
and the only student-derived file in git is `marking/gold/gold_marks.csv`
(pseudonymous integers).

## Measurement discipline

`reading/benchmark/ocr_bench.py` is the arbiter for reading, and
comparisons mean something only on the same pages with the same scorer.
A benchmark page must never become training data. A lower CER is not
automatically better: read the diff. For marking, `agreement.md`
compares intervals `[settled, settled + pending]`, and the examiner's
covers are a reference, not ground truth.

## Environment

Windows and PowerShell are the main development platform; Python 3.11+.
When writing files from PowerShell, use
`[System.IO.File]::WriteAllText(path, text, (New-Object System.Text.UTF8Encoding $false))`,
because `>` and `Out-File` add a BOM. `pipeline.py` runs every stage in
UTF-8 mode, because the scripts print students' text and a Windows
console code page would choke on it.

## Retired, and why: do not resurrect without new evidence

All of it is in git history.

- The line-segmentation pipeline (`02_segment` line crops, `03_router`,
  `05_math`, `07_reconstruct`, TrOCR fine-tuning): reading whole pages
  is 4.7× better and needs no crops. `reading/docs/DONE.md` has the
  numbers.
- `annotation/`, CVAT-to-YOLO layout labelling, abandoned at epoch 49.
- The old marking prototype under `04_evaluate` (`grade.py`, its keys,
  `add_method_marks.py`), superseded by `marking/`.
- The second copy of the marking half (`Major_Project_Eval/`): the
  `marking/` package generates the Colab notebook itself.
