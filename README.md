# Major Project Iter 1

Pipeline for turning scanned handwritten exam answer scripts into a clean,
uniformly-sized, OCR-ready page set with content localised on each page.

```
convert_dataset.py  ->  clean.py  ->  cleaner/ (train.py, apply.py)  ->  segment.py
   (HF -> output/)      (deskew/trim)   (learned denoise/vectorise)     (find content)
```

## 1. Dataset

The source data is a **private** Hugging Face dataset of scanned exam
booklets. It contains real student names, USNs, signatures and marks, so it
is never committed to this repo — `preprocessing/output/` and everything
derived from it are gitignored.

Ask a team member for:
- the Hugging Face repo ID
- an `HF_TOKEN` with read access to it

### Setup

1. Create `.env` in the repo root:

   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   ```

2. Load it and fetch the dataset:

   ```bash
   set -a; . ./.env; set +a
   cd preprocessing
   python convert_dataset.py            # convert students 1..31
   python convert_dataset.py --check    # report only, writes nothing
   ```

   `REPO_ID` in `convert_dataset.py` currently points at
   `prss-majorproject-37/Handwritten-AnswerScripts-MajorProject` — update it
   if your team uses a different repo.

### Format

`convert_dataset.py` normalises the source (a mix of per-page PDFs,
multi-page PDFs and loose PNGs) into one predictable tree:

```
preprocessing/output/student_<NN>/cie_<C>/page_<PP>.png
```

- `NN` — student number, zero-padded to 2 digits (`01`–`61`)
- `C` — CIE number, 1–3, unpadded
- `PP` — page number, zero-padded to 2 digits
- `page_01` of every booklet is the cover page (identity block + marks) —
  pipelines that shouldn't see identity data skip it
- Pages are PNG, grayscale-convertible, nominally 1700×2338 px (one outlier
  scan comes in at a different scale; `clean.py` corrects it)

Any dataset you point at `convert_dataset.py` should match this shape:
one booklet per student per CIE, page order preserved, cover page first.

## 2. Cleaning

```bash
cd preprocessing
python clean.py --preview     # before/after pairs, no writes
python clean.py               # deskew, trim, flatten, canonicalise -> cleaned/
python clean.py --verify      # confirm uniform output size
```

Output: `preprocessing/cleaned/` — same tree as `output/`, geometrically
uniform (1700×2338, deskewed, scanner lip trimmed).

Then the learned cleaner (denoises and vectorises the handwriting —
removes ruling, shadows, bleed-through and binding smudges):

```bash
cd preprocessing/cleaner
python make_data.py           # mine training pairs from output/ -> data/
python train.py                # train the U-Net -> runs/cleaner.pt
python apply.py --preview      # a few before/after pairs
python apply.py                # whole corpus -> ../vectorised/
```

`runs/cleaner.pt` and everything under `data/`, `runs/`, `preview/` are
gitignored (weights and generated crops, not source).

## 3. Segmentation

Locates content (paragraphs, lines) on each cleaned page — geometry only,
no classification:

```bash
cd annotation
python segment.py              # every content page -> segmented/
python segment.py --count 40   # a sample
```

Output: `annotation/segmented/segmentation.json` (page → block → line
hierarchy), `pages.csv` / `blocks.csv` / `lines.csv`, and rendered preview
images per page.

## Requirements

Python 3.10+, and:

```bash
pip install pymupdf pillow huggingface_hub opencv-python numpy torch
```

(`modules/*/requirements.txt` cover the separate OCR/routing modules in
this repo.)

## Notes

- Never commit anything under `preprocessing/output/`, `preprocessing/cleaned/`,
  `preprocessing/vectorised/`, or any rendered preview of a real page — see
  `.gitignore` for the full list. These all contain identifiable student work.
- The actual Hugging Face dataset link is shared separately with the team,
  not in this repo.
