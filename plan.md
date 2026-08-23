# Project plan

Turning scanned handwritten exam answer scripts into one clean Markdown
document per booklet, ready for marking.

61 students x up to 3 CIEs = 153 booklets, 1,231 content pages of
computer-networks answers.

This file is the map: what the pipeline is and how to run it. Two
companions carry the detail, and they are the ones to read next:

- **[DONE.md](DONE.md)** - what is built, what it scores, and the
  evidence behind each decision.
- **[TODO.md](TODO.md)** - what remains, in the order worth doing it.

*State as of 2026-08-23.*

---

## 1. The pipeline

```
MP_Dataset/cleaned/                 normalised scans        [gitignored]
    |
    v
modules/01_prepare/                 deskew -> crop -> tone
    |
    v
modules/02_read/                    page image -> Markdown        [GPU]
    |     Qwen2.5-VL-7B, zero-shot
    |     question numbers, LaTeX maths, ![diagram](x1,y1,x2,y2)
    v
modules/03_assemble/                group pages by question number
    |
    v
modules/04_evaluate/                CER/WER vs hand transcription
    |
    v
modules/05_pipeline/                one booklet in, one document out
```

**Reading the page is the only stage that needs a GPU.** Everything else
is plain Python and runs on a laptop in seconds. That asymmetry is why
reading sits behind a swappable reader rather than being wired in.

---

## 2. Running it

```bash
# one booklet, from markdown already produced
python modules/05_pipeline/src/run_booklet.py student_07/cie_2 --reader cached

# one booklet, against a GPU serving the model over HTTP
python modules/05_pipeline/src/run_booklet.py student_07/cie_2 \
    --reader server --url http://gpu-box:8000/v1

# score any engine against the 15 hand-transcribed pages
python modules/04_evaluate/src/ocr_bench.py --engine qwen7b --verbose
```

### The three readers

| reader | needs | use it for |
|---|---|---|
| `cached` | nothing | demos, and re-runs over pages already read |
| `server` | a URL | the real answer - vLLM or llama.cpp on any GPU |
| `local` | GGUF + llama.cpp | no network, no GPU; minutes per page |

The corpus pass is a batch job, not an interactive one:
`02_read/upload/` holds five ~65MB zips covering all 1,231 pages, and
the notebook resumes, so a dropped Colab session costs only the pages
in flight.

---

## 3. Layout

```
modules/
    01_prepare/     01_deskew  02_crop  03_tone
    02_read/        read_pages_colab.ipynb  make_colab_notebook.py
                    prepare_corpus_batches.py       upload/  [gitignored]
    03_assemble/    assemble.py  question_schema.py
    04_evaluate/    ocr_bench.py  bench_pages.json  RESULTS.md
                    ground_truth/  predictions/     [gitignored]
    05_pipeline/    run_booklet.py  readers.py
```

Tracked: source, READMEs, the benchmark page selection, results.
Gitignored: every page image and every transcription of one.

---

## 4. Privacy - the rule that shapes the design

**Everything generated from this corpus is student work.** `page_01` of
every booklet is the identity block: name, USN, signature, marks.

It is excluded at every stage that touches the corpus - in
`prepare_corpus_batches.py` by construction, again inside the notebook
on arrival, and again in `run_booklet.py`. Not once at the start,
because "the caller already handled it" is exactly the assumption that
leaks student data to a hosted GPU.

That exclusion is the entire basis on which these pages may go to a
rented GPU at all. Answer pages carry handwriting but no identity; cover
pages never leave the machine.

The source lives in a private Hugging Face repo and `HF_TOKEN` comes
from `.env`. Keep it that way.