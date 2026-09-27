# Getting the data from Hugging Face

The answer scripts live in two **private** dataset repos on the Hugging
Face Hub, under the `prss-majorproject-37` organisation. This page
explains what a Hugging Face dataset repo actually is, what ours look
like, and how to bring them onto a machine — by command line, by script,
or from Python.

---

## 1. What a Hugging Face dataset repo is

A dataset repo is **a git repository hosted on the Hub**. That's it.

- Small text files (`README.md`, `.gitattributes`, CSVs) are stored like
  any git file.
- Large binaries — images, archives, Parquet — are stored through
  **Git LFS / Xet**: git only keeps a pointer, and the Hub serves the
  actual bytes. `.gitattributes` lists which patterns go that way
  (`*.png`, `*.parquet`, `*.zip`, …).
- `README.md` is the **dataset card**: a YAML header (`license`,
  `pretty_name`, `tags`) followed by free-form documentation.
- A repo is **public or private**. Private repos need an access token
  from an account that belongs to the organisation.

There are two ways data is laid out inside a repo, and they are imported
differently:

| layout | what is in the repo | how you read it |
|---|---|---|
| **file / folder dataset** | ordinary files in folders (`*.png`, `*.pdf`, `*.csv`) | download the files, walk the folders |
| **Parquet (the `datasets` format)** | `data/train-00000-of-00004.parquet` …: tables with typed columns, e.g. an `image` column of type `Image()` plus `student`, `cie`, `page` columns | `datasets.load_dataset(...)` gives you rows |

**Ours are folder datasets.** There is no Parquet, no splits, no
`dataset_infos` — just the pages as PNG files in a fixed folder
structure. That is deliberate: every stage of the pipeline works on page
image files, and a folder of PNGs needs no special library to read.

---

## 2. Our two repos

| short name | repo id | what it holds | pages | size |
|---|---|---|---|---|
| `raw` | `prss-majorproject-37/Handwritten-AnswerScripts-MajorProject` | the scans, normalised: all PNG, correct page order, flip fixed, native 1700×2338 | 1,385 | 7.3 GB |
| `cleaned` | `prss-majorproject-37/cleaned-handwritten-answerscripts` | deskewed, cropped to 1598×2177, tone-flattened, 8-bit greyscale | 1,384 | 381 MB |

Both use the same layout:

```
student_<NN>/            student_01 … student_61
    cie_<M>/             cie_1, cie_2, cie_3 — only the exams actually sat
        page_01.png      the cover: name, USN, signature, examiner's marks
        page_02.png      answers start here, in reading order
        ...
README.md                the dataset card (processing notes, known gaps)
students.csv             raw repo only: the roster (student_id, usn, name,
                         pages per CIE, review_flag)
```

61 students, 153 booklets. Known gaps are real, not lost data: students
18–61 sat only some CIEs, and `cleaned` drops one scan outlier
(`student_19/cie_2/page_14`).

**The `cleaned` repo is the one the pipeline reads.** Part 1's first
stage expects exactly `student_NN/cie_C/page_PP.png` at 1598×2177, which
is this repo as downloaded. Use `raw` only if you need the untouched
scans.

---

## 3. Before you import: the token

1. Get a **read** token: huggingface.co → Settings → Access Tokens. Your
   account must be a member of `prss-majorproject-37`.
2. Put it in `.env` at the repo root (gitignored):

   ```bash
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
   ```

   or `export HF_TOKEN=...` in your shell, or run `hf auth login` once.
   `huggingface_hub` reads `HF_TOKEN` automatically.

Never paste the token into code, a notebook, or a commit.

---

## 4. Four ways to import

### a. The project script (recommended)

`scripts/fetch_hf.py` wraps the download with this project's privacy
rules: it **skips every `page_01` cover** and the **`students.csv`
roster** unless you ask for them, and it writes into gitignored folders.

```bash
PY=Major_Project_Eval_Local/.venv/bin/python   # already has huggingface_hub

$PY scripts/fetch_hf.py --list                       # describe the repo, download nothing
$PY scripts/fetch_hf.py                              # all cleaned pages -> dataset_cleaned/
$PY scripts/fetch_hf.py --students 1-5 --cie 2       # a subset
$PY scripts/fetch_hf.py --repo raw --students 7      # raw scans -> dataset/
$PY scripts/fetch_hf.py --with-covers --with-roster  # only if you really need names
```

It is resumable: run it again and files already on disk are skipped.

### b. The `hf` command line

```bash
hf download prss-majorproject-37/cleaned-handwritten-answerscripts \
    --repo-type dataset --local-dir dataset_cleaned \
    --include "student_05/*" --exclude "*/page_01.png"
```

(`huggingface-cli download …` is the older spelling of the same command.)

### c. From Python, a whole folder or one file

```python
from huggingface_hub import snapshot_download, hf_hub_download

# a subset of the repo, as files on disk
root = snapshot_download(
    "prss-majorproject-37/cleaned-handwritten-answerscripts",
    repo_type="dataset",
    local_dir="dataset_cleaned",
    allow_patterns=["student_05/cie_2/*.png"],
    ignore_patterns=["*/page_01.png"],
)

# exactly one file (cached under ~/.cache/huggingface)
page = hf_hub_download(
    "prss-majorproject-37/cleaned-handwritten-answerscripts",
    "student_05/cie_2/page_03.png",
    repo_type="dataset",
)
```

Then read it like any folder:

```python
from pathlib import Path
from PIL import Image

for path in sorted(Path("dataset_cleaned").glob("student_*/cie_*/page_*.png")):
    student, cie, page = path.parts[-3], path.parts[-2], path.stem
    if page == "page_01":
        continue                      # the cover - personal data
    image = Image.open(path)          # 1598x2177, greyscale
```

### d. With the `datasets` library

`pip install datasets`, then:

```python
from datasets import load_dataset

pages = load_dataset(
    "imagefolder",
    data_dir="dataset_cleaned",       # after a download as above
    split="train",
)
```

This works, but for **our** layout it is the weaker option: `imagefolder`
turns folder names into a single `label` column and loses the
student / CIE / page structure, and it gives you nothing the file paths
don't already say. It pays off only if you convert the pages into a
Parquet dataset with explicit `student`, `cie` and `page` columns —
worth it for training, not for this pipeline.

---

## 5. How this connects to the rest of the project

```
HF: cleaned-handwritten-answerscripts
        │   scripts/fetch_hf.py  (or hf download)
        ▼
dataset_cleaned/student_NN/cie_C/page_PP.png
        │   part 1 — reading and segmentation (see extra/)
        ▼
handoff1/                       part 1's output: booklet.json per booklet,
        │                       page images, drawing crops
        │   part 2 — python src/run_all.py
        ▼
output/marks/*.json, summary.csv, agreement.md   →   python src/serve.py
```

**Part 2 does not read the Hub directly.** It reads Part 1's handoff
(`handoff1/`), which is not on Hugging Face. If you want to share the
handoff the same way, push it to a *private* dataset repo in the same
organisation. Anyone on the team can then import it with:

```bash
$PY scripts/fetch_hf.py --repo prss-majorproject-37/<handoff-repo> --out handoff1
```

`paths.py` looks for `handoff1/` at the repo root first, so
`python src/run_all.py` picks it up without `MPE_HANDOFF`.

---

## 6. The rules, once more

- The repos stay **private**. The data is students' names, USNs,
  signatures, marks and handwriting.
- Downloads go to gitignored folders: `dataset/`, `dataset_cleaned/`,
  `data_hf/`, `handoff1/`. Check `git status` before any `git add -A`.
- Covers (`page_01`) and the roster are opt-in, never default.
- The token lives in `.env` or your shell, never in the repo.
