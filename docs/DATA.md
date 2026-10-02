# Data: where it comes from, where it lives, and the rules

Everything this project reads or writes about students lives under one
gitignored folder, `data/` at the repo root. `.gitignore` excludes it in
one line, and that line is the whole protection against committing
student work.

---

## 1. The data root

```
data/                          $MP_DATA moves the whole tree
  raw/                         untouched scans (HF raw repo)          only for `prepare`
  prepare/                     deskew and crop intermediates, previews
  pages/                       prepared pages (HF cleaned repo)       the reader's input
  read/<engine>/               one transcription per page: s07_c2_p03.md
  read/batches/                zips for a hosted GPU
  figures/                     the diagram pass: batch, diagrams.json, review.html, rejects.json
  geometry/                    segment.py's standalone output, for inspection
  booklets/<booklet>/          booklet.md, structure.json, figures/
  handoff/                     the contract marking reads (docs/HANDOFF.md)
  marking/                     marks, queues, verdicts, the human log, every report
  schemes/                     the three scheme PDFs; pages/ holds their renders
  benchmark/ground_truth/      hand transcriptions the reader is scored against
```

`common/layout.py` is the only place these paths are defined; every
script asks it. Page ids are `s<NN>_c<C>_p<PP>`, booklet ids are
`student_<NN>_cie_<C>`.

**If this repo sits in a synced folder** (OneDrive, Dropbox), `data/` is
synced with it: student pages, transcriptions and marks. To keep them on
this machine only, put `MP_DATA=D:\somewhere\local` in `.env`.

---

## 2. The Hugging Face repos

The scans live in two **private** dataset repos under the
`prss-majorproject-37` organisation.

| name | repo id | holds | pages | size |
|---|---|---|---|---|
| `cleaned` | `prss-majorproject-37/cleaned-handwritten-answerscripts` | deskewed, cropped to 1598×2177, tone-flattened, 8-bit greyscale | 1,384 | 381 MB |
| `raw` | `prss-majorproject-37/Handwritten-AnswerScripts-MajorProject` | the scans normalised: PNG, page order fixed, native 1700×2338, plus `students.csv` (the roster) | 1,385 | 7.3 GB |

Both are **folder datasets**, not Parquet: `student_<NN>/cie_<M>/page_<PP>.png`,
61 students, 153 booklets. Gaps are real, not lost data: students 18–61
sat only some CIEs, and `cleaned` drops one scan outlier
(`student_19/cie_2/page_14`).

**`cleaned` is what the pipeline reads**, straight into `data/pages/`.
`raw` is needed only to rebuild the prepared pages with
`python pipeline.py prepare`, and that rebuild does not yet reproduce
`cleaned` exactly (see [`reading/docs/TODO.md`](../reading/docs/TODO.md)
item 9). Treat `cleaned` as canonical.

### The token

1. huggingface.co → Settings → Access Tokens → a **read** token, from
   an account in `prss-majorproject-37`.
2. Put it in `.env` at the repo root, which is gitignored:
   `HF_TOKEN=hf_...`. `.env.example` lists every other setting.

Never paste the token into code, a notebook or a commit.
`python pipeline.py status` says whether one is set, without printing it.

### Fetching

```bash
python pipeline.py fetch --list                     # describe the repo, download nothing
python pipeline.py fetch                            # all cleaned pages -> data/pages
python pipeline.py fetch --students 1-5 --cie 2     # a subset
python pipeline.py fetch --repo raw --students 7    # raw scans -> data/raw
python pipeline.py fetch --with-covers              # only if you really need page 1
python pipeline.py fetch --repo raw --with-roster   # only if you really need names
```

`fetch` is `scripts/fetch_hf.py`. It is resumable, and it **skips every
`page_01` cover and the roster unless asked**.

Other ways in work too, as long as the files land in `data/pages/`:

```bash
hf download prss-majorproject-37/cleaned-handwritten-answerscripts \
    --repo-type dataset --local-dir data/pages \
    --include "student_05/*" --exclude "*/page_01.png"
```

```python
from huggingface_hub import snapshot_download
snapshot_download("prss-majorproject-37/cleaned-handwritten-answerscripts",
                  repo_type="dataset", local_dir="data/pages",
                  allow_patterns=["student_05/cie_2/*.png"],
                  ignore_patterns=["*/page_01.png"])
```

`datasets.load_dataset("imagefolder", ...)` also works, but for this
layout it is the weaker option. It turns folders into one `label`
column and loses the student / CIE / page structure the paths already
carry.

### Publishing

`python scripts/publish_dataset.py --dry-run` describes what would go up.
Without `--dry-run` it uploads `data/pages/` over the `cleaned` repo
(always private), with `reading/docs/DATASET_CARD.md` as its card. Do not
run it while the rebuild discrepancy above is open.

---

## 3. Inputs that are not on the Hub

| what | where it goes | where it comes from |
|---|---|---|
| page transcriptions | `data/read/<engine>/` | the read: the Colab notebook, Kaggle, or a server (`docs/PIPELINE.md`). The recorded 1,000-page read exists only on the machine that ran it. |
| diagram pass results | `data/figures/diagrams.json` | `reading/figures/find_diagrams.ipynb` on Colab |
| the three scheme PDFs | `data/schemes/` (or `MPE_SCHEMES`) | the department. Named `CIE 1 scheme CN.pdf`, `CIE 2 Scheme CN.pdf`, `CIE 3 Scheme CN.pdf`: the capitalisation is theirs |
| part 1's September handoff | anywhere, with `MPE_HANDOFF` pointing at it | the other part 1 implementation (archived with the old `main`) |
| hand transcriptions | `data/benchmark/ground_truth/` | people, page by page |

---

## 4. The rules

- **The repos stay private.** The data is students' names, USNs,
  signatures, marks and handwriting.
- **Nothing under `data/` is ever committed**, and nothing derived from a
  booklet is committed anywhere else. The single exception is
  `marking/gold/gold_marks.csv`: pseudonymous integers, no names or
  handwriting. Check `git status` before any broad `git add`.
- **`page_01` is the identity cover.** It is opt-in at fetch, and it is
  excluded again at every stage that could send a page off this machine:
  `prepare_corpus_batches.py` by construction, inside the Colab notebook
  on arrival, by `kaggle_run.py`, and by `read_pages.py` before any
  reader sees a page. That redundancy is deliberate. "The caller already
  handled it" is the assumption that leaks student data to a hosted GPU.
- **The roster is opt-in.** Booklets are anonymised on disk as
  `student_NN`; the roster is the only link to a person.
- **The token lives in `.env` or the shell**, never in the repo.
- **The marking website binds 127.0.0.1 only.** Its pages quote real
  students.
