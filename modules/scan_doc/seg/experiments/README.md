# Rejected approaches

Kept so these are not tried again from scratch. Neither is part of the
pipeline; nothing in the parent module imports from here.

## Grounding DINO zero-shot pre-labeling

`bootstrap_label.py`, `visualize_bootstrap.py`

**Idea.** Use an open-vocabulary detector to pseudo-label pages by
prompting with the target class names ("table", "handwritten sentence",
"code snippet", …), so a human only corrects labels rather than drawing
every box.

**Why it was rejected.** Tested at `IDEA-Research/grounding-dino-tiny`
and `-base` on the sample booklet:

- Whole-page inference gave 2–9 detections per page (a page has ~20
  text lines), all scoring 0.20–0.35.
- Labels came back tokenizer-mangled and merged across prompt phrases:
  `"##written sentence"`, `"hand sentence code snippet text"`,
  `"table diagram mathematical equation"` — an artifact of overlapping
  phrases in a single period-separated prompt.
- Per-crop inference (feeding it one region at a time, so the visual
  context is trivial) did not fix it: a ruled answer table came back
  `"diagram"` at 0.50.

The model has no useful grounding for "ruled handwritten answer sheet".
Raising recall by lowering thresholds only added noise.

**What replaced it.** `detect_lines.py` (docTR DBNet) for text geometry
and `detect_nontext.py` for everything else. See the parent README.

**Reproduce, if you must:**

```bash
cd .. && .venv/bin/python experiments/bootstrap_label.py
```

(Paths inside assume the parent directory; run from there.)
