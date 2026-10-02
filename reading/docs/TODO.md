# What is left: the reading half

Ordered by what unblocks the most. Each item says what "done" means, so
it can be checked rather than argued about. The marking half keeps its
own list in [`marking/IMPROVEMENTS.md`](../../marking/IMPROVEMENTS.md).

*As of 2026-10-02, after the restructure into one pipeline. The previous
version of this file (2026-09-14) is in git history.*

---

## Where it stands

| | |
|---|---|
| pages prepared | 1,384 in the `cleaned` Hugging Face repo: 61 students, 153 booklets |
| pages read | 1,000, by Qwen2.5-VL-7B, 0.099 CER on the 15-page benchmark. **The transcriptions are not in git or on the Hub.** They are on the machine that ran the Colab notebook. |
| booklets assembled | 122 before the filing fix below. **All need rebuilding.** |
| answer keys | 3 CIEs, 24 questions, 107 rubric items, in `marking/keys/` |

`batch_04` (231 pages, students 46-61) was deliberately never read. If
the corpus is ever finished, everything after `read` must be re-run:
`python pipeline.py run`.

---

## P0: the path to a finished deliverable

### 1. Put the existing transcriptions where the pipeline reads them

The 1,000 pages read on Colab are the most expensive thing this half has
produced, and they live only on the machine that read them, under the
old layout's `modules/02_read/output/all_read/` or
`modules/04_evaluate/predictions/qwen7b/`. Copy the `.md` files to
`data/read/all_read/`.

If they are lost, read again: `python pipeline.py batches`, the Colab
notebook, then unzip its download into `data/read/<engine>/`.

**Done when:** `python pipeline.py read --engine all_read` reports about
1,000 content pages read.

### 2. Rebuild every booklet

Until 2026-10-02 `build_booklet.py` filed every page whole under the
**last** question heading on it. A page carrying the end of 1 and the
start of 2a gave question 1's text to 2a, and 1 then looked unattempted
to the grader. Short questions share pages constantly, so every booklet
assembled before the fix is affected. `tests/test_pipeline.py` shows the
old code doing it and holds the fix in place.

    python pipeline.py assemble --engine all_read

**Done when:** `data/booklets/booklets.csv` is regenerated and every
booklet folder carries a `structure.json`.

### 3. Run the diagram pass

The reading pass emits a figure marker on only 52 of 1,000 pages, because
finding figures was a side clause in a prompt whose real job was
transcription. The diagram pass asks only that question.

    python pipeline.py figures --engine all_read
    # run reading/figures/find_diagrams.ipynb on a Colab T4 over
    # data/figures/diagram_batch.zip (~145 MB, about an hour), then
    # unzip its download into data/figures/
    python pipeline.py assemble --engine all_read   # uses diagrams.json on its own

**Done when:** total figures has *fallen* from 631, `unreferenced` is 0,
and `student_19_cie_1` carries its last-page diagram in place.

### 4. Review the figures

    python reading/assemble/review_figures.py

Open `data/figures/review.html`, click anything that is not a drawing,
save `rejects.json` into `data/figures/`, then assemble again.

**Done when:** the reject rate is recorded. That number *is* the figure
precision, and it is the only evidence behind "the diagrams are
diagrams".

### 5. Mark this half's own reading, end to end

The marking half was built and measured against a 50-booklet handoff
from the *other* part 1 implementation, on the archived `main` branch.
This half's 1,000-page read has never been through it.

    python pipeline.py handoff
    python pipeline.py mark

**Done when:** `data/marking/agreement.md` exists for this read and its
headline sits beside the 50-booklet one in `marking/README.md`: same
rubric, same examiner, different reader.

---

## P1: trust the output

### 6. The reader never says "I cannot read this"

**Still the most important open risk.** Across the benchmark pages the
7B emitted zero `[?]` and zero `![table]`, despite a prompt asking for
both. A model that never declines is not always right. Its errors are
just invisible, and for marking an invented answer is worse than a
blank one.

The code now has two mitigations, though neither is the model declining:
`loop_start()` cuts a page at a repetition loop, and the
`CONFABULATION_YIELD` test blanks a page carrying more text than its ink
can support. Both were built after real failures.

### 7. More ground truth

23 pages hand-transcribed against a target of 30+, and too few carry
tables, which is exactly the case that fails. Ground truth goes in
`data/benchmark/ground_truth/`. It is students' words, so it never goes
in git.

### 8. Spot-check at corpus scale

CER on 15 pages says nothing about page 900. Sample 20 at random,
compare each against its image, and confirm the error rate has not
drifted.

### 9. Rebuilding pages from raw does not reproduce the published ones

Measured 2026-10-02 on `student_07/cie_2` (9 pages). Running
`python pipeline.py prepare` on the raw scans gives 2 pages
byte-identical to the `cleaned` repo (1 under OpenCV 4.14) and 7 that
differ: mean absolute difference up to 25 grey levels, with 12.5% of
pixels off on the worst page. The large differences are the same under
OpenCV 4.14 and 5.0, and the original scripts from git produce exactly
what today's do. So `cleaned` was made by an earlier version of prepare.

Until that is understood, treat `cleaned` as canonical and do not
republish over it with `scripts/publish_dataset.py`.

A related fix is waiting on the archived `main` branch. Its deskew uses a
30px rule kernel instead of 60, because at 60 the most skewed pages
(about 3 degrees) lose every printed rule and are left unrotated. That
code is in the backup of the old `main` (see the root README).

**Done when:** the version that produced `cleaned` is identified, or the
difference is measured to be harmless, for example as unchanged reader
CER on the benchmark pages.

---

## P2: engineering debt

- **The tests cover the stitch, not the stages.** `tests/test_pipeline.py`
  runs assemble → handoff → marking on synthetic pages. Natural next
  tests: `ocr_bench.normalise`, `build_booklet.split_label` and
  `loop_start`, and `readers.CachedReader`'s key lookup, which once had
  a bug that reported an empty booklet.
- **Local reading is unproven.** `LocalReader` is written but has never
  run. A 496 MB Radeon cannot hold Qwen-7B, so a laptop demo means
  Qwen2.5-VL-3B at Q4 (~2 GB), with an honest note about the CER cost.
- **No stage reads the cover.** A booklet's identity in the handoff is
  its corpus number only. `reading/prepare/tools/cover_template.py` is
  groundwork for reading the cover form.

---

## Explicitly not doing

- **Fine-tuning TrOCR.** Costed at about 53 pages of transcription and
  126 hours of CPU training, to beat a zero-shot reader that already
  works.
- **A trained figure detector.** A YOLOv8s trained on 678 boxes from the
  geometry rule reached mAP50 0.626, but that only measures agreement
  with the detector that labelled it. On the hand-labelled holdout it
  tied geometry at 9/14, having learned the teacher's blind spot too.
  A GPU dependency that ties the incumbent is not worth keeping.
- **A column-profile prose/diagram classifier.** Band AUC 0.82, F1 0.56,
  which is not good enough to gate a figure on.
- **Pinning exact figure edges.** Both of the above, and a three-way
  fusion scorer, were built to find a drawing's precise vertical extent.
  The deliverable does not need it: a crop carrying a spare line of
  writing reads fine in the Markdown, and one that clips an arrow does
  not. Over-crop on purpose and the problem goes away.
- **Hand-transcribing the corpus.** Transcription is ground truth for
  measuring the reader, not a substitute for it.
