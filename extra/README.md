# Handwritten answer scripts to Markdown

Turns scanned handwritten exam booklets into one clean Markdown document
per booklet, ready for marking.

61 students x up to 3 CIEs = 153 booklets, 1,231 content pages of
computer-networks answers.

```
prepared page  ->  Qwen2.5-VL  ->  Markdown  ->  grouped by question
                                                  ->  one document per booklet
```

Measured at **0.099 character error rate** on 15 hand-transcribed pages,
zero-shot, with no training and no labelled data.

---

## Read next

| file | what it is |
|---|---|
| **[plan.md](plan.md)** | the map - architecture, how to run it, privacy |
| **[DONE.md](DONE.md)** | what is built, what it scores, the evidence |
| **[TODO.md](TODO.md)** | what is left, in the order worth doing it |
| [DATASET.md](DATASET.md) | the Hugging Face card for the prepared corpus |

---

## Quick start

```bash
# one booklet, from markdown already produced
python modules/05_pipeline/src/run_booklet.py student_07/cie_2 --reader cached

# score an engine against the 15 hand-transcribed pages
python modules/04_evaluate/src/ocr_bench.py --engine qwen7b --verbose

# group page markdown into one answer per question
python modules/03_assemble/src/assemble.py --engine qwen7b

# package the corpus for a GPU run
python modules/02_read/src/prepare_corpus_batches.py
```

Reading a page is the only stage that needs a GPU. It runs from
`modules/02_read/read_pages_colab.ipynb` on a free Colab T4, or from any
machine serving the model over HTTP:

```bash
python modules/05_pipeline/src/run_booklet.py student_07/cie_2 \
    --reader server --url http://gpu-box:8000/v1
```

---

## Setup

### Environment

```bash
uv venv --python 3.11 .venv
.venv/Scripts/python -m pip install opencv-python-headless numpy
```

That is the whole local dependency list. **No torch, no transformers.**
Those left with the TrOCR line-recognition pipeline, which reading whole
pages replaced - see [DONE.md](DONE.md).

Two optional extras:

- `huggingface_hub` - only for `01_prepare/publish_dataset.py`
- `llama-cpp-python` - only for `--reader local`, running a GGUF model
  on CPU with no network

### Data

The pipeline expects prepared pages here:

```
modules/01_prepare/03_tone/output/student_<NN>/cie_<C>/page_<PP>.png
```

- `NN` student, 01-61 - `C` CIE, 1-3 - `PP` page, zero-padded
- all pages 1598x2177, 8-bit greyscale
- `page_01` is the cover sheet and is skipped everywhere

A directory junction (or symlink) pointing at a corpus elsewhere on disk
works fine, and is how this is set up locally.

**There is no ingestion script in this repo.** The step that normalised
the raw Hugging Face download into that tree lived in a `preprocessing/`
directory that no longer exists; recover it from git history if you need
to rebuild the corpus from scratch. `01_prepare/` picks up from the
normalised tree onward: deskew, crop, tone.

The source is a **private** Hugging Face dataset. Ask a team member for
the repo id and an `HF_TOKEN` with read access, and put it in `.env` at
the repo root:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

---

## Privacy

**Everything generated from this corpus is student work.** `page_01` of
every booklet is the identity block - name, USN, signature, marks.

It is excluded at every stage that touches the corpus: by construction
in `prepare_corpus_batches.py`, again inside the notebook on arrival,
and again in `run_booklet.py`. Not once at the start, because "the
caller already handled it" is exactly the assumption that leaks student
data to a hosted GPU. That exclusion is the basis on which these pages
may go to a rented GPU at all.

Never commit a page image or a transcription of one. `.gitignore`
covers `input/`, `output/`, `upload/`, `predictions/`, `ground_truth/`,
`crops/`, `markers/` and `.venv/`; the tracked files are source,
documentation, the benchmark page selection and the results.