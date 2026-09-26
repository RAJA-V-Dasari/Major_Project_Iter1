# What is left

Ordered by what unblocks the most. Each item says what "done" means, so
it can be checked rather than argued about.

*As of 2026-09-14. The previous version of this file was stamped
2026-08-23 and understated the state of the project badly - it claimed
15 of 1,231 pages were read and that the answer keys did not exist. Both
had been false for weeks. Check this file against the disk before
trusting it again.*

---

## Where the project actually is

| | |
|---|---|
| pages prepared | 1,231 (61 students, 153 booklets) |
| pages read | **1,000**, CER 0.099 |
| booklets assembled | **122**, with figures embedded |
| booklets graded | **122**, 61.6% of rubric items decided with no human |
| answer keys | 3 CIEs, 24 questions, 121 rubric items, 0 mismatches |

### Scope decision: batch04 is cut

`batch_04.zip` (231 pages, 31 booklets, students 46-61) is packaged and
was never run. **It is deliberately not being run.** The project ships on
the 1,000 pages already read.

The reason it matters beyond scope: the LLM grading tier was held back
only to stop batch04 overwriting its reports. Cutting batch04 unblocks
it. If the corpus is ever finished, `grade.py --all` must be re-run
before the LLM tier, not after.

---

## P0 - the remaining path to a finished deliverable

### 1. Run the diagram pass
`experiments/diagram_pass/find_diagrams.ipynb` on Colab, over
`upload/diagram_batch.zip` (1,000 pages, 147 MB). Roughly an hour on a
T4; checkpoints to Drive every 25 pages and resumes.

**Why it exists.** The reading pass emits a `![diagram: ...]` marker on
52 of 1,000 pages - finding figures was a side clause in a prompt whose
real job was transcription. `build_booklet.py` pairs the nth marker to
the nth region down the page, so when the markers run out the leftover
regions are appended at the *end* of the page block. That, plus a
geometry rule that fires on 37% of pages, is why figures are currently
misplaced and often not figures.

**Done when:** `diagrams.json` is in `experiments/diagram_pass/output/`,
the pass fires on well above 5% of pages, and the built-in benchmark
leaves the 6 hand-labelled prose pages alone.

### 2. Rebuild the booklets from it

    python modules/03_assemble/src/build_booklet.py --engine all_read \
        --diagrams experiments/diagram_pass/output/diagrams.json --all

**Done when:** total figures has *fallen* from 631, `unreferenced` is
0, and `student_19_cie_1` carries its last-page diagram in place.

### 3. Review the figures

    python modules/03_assemble/src/review_figures.py

Open `modules/03_assemble/output/review.html`, click anything that is
not a drawing, save `rejects.json`, rebuild with `--rejects`.

**Done when:** the reject rate is recorded. That number *is* the
figure precision, and it is the only evidence behind "the diagrams are
diagrams".

### 4. Finish grading
`grade.py --all`, then `make_grader_notebook.py`, then the 399-call LLM
notebook on Colab, then `grade.py --apply verdicts.jsonl`.

**Done when:** `summary.csv` `marks_pending` has fallen by the LLM
tier's share. The 710-item human queue is marking work, not
engineering - it is the designed-in human tier, and reporting it as
such is the honest outcome.

---

## P1 - trust the output

### 5. The reader never says "I cannot read this"
**Still the most important open risk.** Across the benchmark pages the
7B emitted zero `[?]` and zero `![table]`, despite a prompt asking for
both. A model that never declines is not a model that is always right;
it is one whose errors are invisible, and for marking an invented
answer is worse than a blank one.

Mitigations now in the code, neither of which is the same as the model
declining: `loop_start()` cuts a page at a repetition loop, and the
`CONFABULATION_YIELD` test blanks a page carrying more text than its
ink can support. Both were built after real failures.

### 6. More ground truth
23 pages hand-transcribed against a target of 30+, and too few carry
tables - the exact case that fails.

### 7. Spot-check at corpus scale
CER on 15 pages says nothing about page 900. Sample 20 at random,
compare against the image, confirm the error rate has not drifted.

---

## P2 - engineering debt

- **No dependency manifest.** `cv2`, `numpy`, `torch`, `transformers`
  imported and declared nowhere. One `pyproject.toml`.
- **No tests.** Natural first three: `ocr_bench.normalise`,
  `assemble.normalise_question`, and `readers.CachedReader` key lookup -
  the last of which had a bug that reported an empty booklet.
- **Local inference unproven.** `LocalReader` is written but never run.
  A 496 MB Radeon cannot hold Qwen-7B, so a laptop demo means
  Qwen2.5-VL-3B at Q4 (~2 GB) and an honest note about the CER cost.
- **`modules/04_evaluate/README.md` is stale** - it documents a deleted
  `06_evaluation` module.
- **`03_assemble/output/coverage.csv` is stale**, left by an older
  `assemble.py` run. `booklets.csv` is the current one.
- **The marking stage is untracked in git** - `keys/`, `grade.py`,
  `add_method_marks.py`, `make_grader_notebook.py`. Decide whether
  `answer_keys/*.pdf` should be committed before any broad `git add`.

---

## Explicitly not doing

- **Fine-tuning TrOCR.** Costed: ~53 pages of transcription and ~126
  hours of CPU training, to beat a zero-shot reader that already works.
- **A trained figure detector.** A YOLOv8s was trained on 678 boxes from
  the geometry rule and reached mAP50 0.626 - but that is agreement with
  the detector that labelled it. On the hand-labelled holdout it tied
  geometry at 9/14, having learned the teacher including its blind spot.
  A GPU dependency that ties the incumbent is not worth keeping.
- **A column-profile prose/diagram classifier.** Band AUC 0.82, F1 0.56.
  Not good enough to gate a figure on.
- **Pinning exact figure edges.** Both of the above, and a three-way
  fusion scorer, were built to find a drawing's precise vertical extent.
  The deliverable does not need it: a crop carrying a spare line of
  writing reads fine in the Markdown, and one that clips an arrow does
  not. Over-crop on purpose and the problem goes away.
- **Hand-transcribing the corpus.** Transcription is ground truth for
  measuring the reader, not a substitute for it.
