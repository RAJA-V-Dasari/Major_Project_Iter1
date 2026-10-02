# Handwritten answer booklets: read, then marked

61 students' handwritten Computer Networks booklets (3 CIEs, 153
booklets, about 1,230 answer pages). This repository turns each scanned
booklet into structured answers with a vision model, marks them against
the department's three CIE answer schemes, and measures the marks
against what the faculty wrote on the covers. A local website is where a
person finishes what the machine could not.

```
 Hugging Face ─fetch─> pages ─read [GPU]─> transcriptions ─assemble─> booklets
                                                                          │
         examiner's covers ──> agreement <── marks <──mark── handoff <──handoff─┘
                                                │
                                                └──> review website (a person)
```

| stage | what it does | runs on |
|---|---|---|
| `fetch` | prepared pages from the private Hugging Face repo | laptop |
| `read` | each page to Markdown with Qwen2.5-VL-7B, **0.099 CER** zero-shot | **GPU**: Colab, Kaggle or a server |
| `figures` | the diagram pass: which pages hold a drawing, and where it goes | **GPU**, optional |
| `assemble` | pages grouped by question into one booklet each, drawings cropped | laptop |
| `handoff` | booklets packaged as the contract marking reads | laptop |
| `mark` | each rubric item decided by the cheapest tier that can honestly decide it | laptop (+ a model tier) |
| `review` | the human tier: one booklet per page, with the drawings | browser, 127.0.0.1 |

Every stage is one command, `python pipeline.py <stage>`, and every
stage is also a script you can run on its own.

---

## Quick start

```bash
git clone https://github.com/RAJA-V-Dasari/Major_Project_Iter1
cd Major_Project_Iter1

python -m venv .venv
.venv\Scripts\activate               # Windows; source .venv/bin/activate elsewhere
pip install -e .                     # add ".[semantic]" for the embedding tier

copy .env.example .env               # cp on macOS/Linux; then set HF_TOKEN=... in .env
python pipeline.py check             # tests and every check that needs no data
python pipeline.py fetch             # prepared pages -> data/pages/
```

Reading the pages is the one step that needs a GPU. Run
`reading/read/read_pages_colab.ipynb` on a free Colab T4 (or use Kaggle,
or a server; see [`docs/PIPELINE.md`](docs/PIPELINE.md#2-read-pages-to-markdown--gpu)),
and unzip its download into `data/read/qwen7b/`. Then everything else
runs on a laptop:

```bash
python pipeline.py run --engine qwen7b   # assemble -> handoff -> mark
python pipeline.py review                # http://127.0.0.1:8000
python pipeline.py status                # what exists, and what to run next
```

---

## Repository layout

```
pipeline.py                 the one entry point: status, fetch ... review, run, check
common/layout.py            every data path, in one place
reading/                    part 1: scanned pages -> structured booklets
  prepare/                  raw scan -> prepared page (only to rebuild data/pages)
  read/                     page -> Markdown: the prompt, the Colab notebook,
                            the swappable readers, batch packaging, qa/ checks
  figures/                  the diagram pass (Colab notebook + batch builder)
  assemble/                 pages -> booklets: grouping, figure crops, geometry
  handoff/                  booklets -> the contract marking reads
  benchmark/                the reader's CER against hand transcription
  docs/                     what was measured and decided, what is left
marking/                    part 2: the handoff -> marks -> agreement
  src/                      the ladder, the model tier's three routes, the website
  keys/                     the rubric, hand-authored from the scheme scans
  gold/                     the examiner's marks, as pseudonymous integers
  notebooks/                the model tier's Colab notebook (generated)
  docs/                     architecture, run order, data formats, glossary
scripts/                    fetch_hf.py, publish_dataset.py
tests/                      the stitch, end to end, on synthetic pages
docs/                       PIPELINE.md, DATA.md, HANDOFF.md, slides/, history/
data/                       everything generated - gitignored, never committed
```

---

## Data and privacy

Every page, transcription, crop, mark and report is a real student's
work, so all of it lives under **`data/`, which git ignores in one
line**. The scans come from two private Hugging Face repos, read with
the token in `.env`. The only student-derived file in git is
`marking/gold/gold_marks.csv`: pseudonymous integers, no names and no
handwriting.

- **Covers are opt-in.** `page_01` of every booklet carries a name, USN,
  signature and marks. `fetch` skips it unless asked, and every stage
  that could send a page to a hosted GPU refuses it again.
- **Not on the Hub:** the page transcriptions (the GPU read produces
  them), the diagram pass results, and the department's scheme PDFs.
  [`docs/DATA.md`](docs/DATA.md) says where each goes.
- **This folder may be synced** (OneDrive). To keep student data on this
  machine only, set `MP_DATA` in `.env` to an unsynced folder.

---

## Where it stands

**Reading.** On 15 hand-transcribed pages, same scorer, same pages:

| reader | character error rate |
|---|---|
| segment into lines, TrOCR each line | 0.463 |
| Qwen2.5-VL-3B, whole page, zero-shot | 0.229 |
| **Qwen2.5-VL-7B, whole page, zero-shot** | **0.099** |

1,000 of the 1,231 content pages have been read; the last 231 were
deliberately cut. Those transcriptions live only on the machine that
read them, and copying them into `data/read/` is step one of
[`reading/docs/TODO.md`](reading/docs/TODO.md).

**Marking.** Measured on the September handoff of 50 booklets (one per
student), 1,221 rubric items over 289 questions with an examiner mark.
Each question is compared as the interval `[settled, settled + pending]`:

| model tier | examiner's mark reachable | over-settled | under-settled |
|---|---|---|---|
| Qwen2.5-7B on Colab | 229 (79%) | 21 | 39 |
| Claude in session, incl. a pass shown the drawings | 150 (52%) | 31 | 108 |

The Qwen run left far more items open, and an open item widens the
interval, so this is not a ranking. The in-session run decided 946 of
1,221 items and left 275 for a person, mostly answers whose evidence is
a drawing. The examiner is a reference, not ground truth: 7 of 50 covers
carry a demonstrable defect. [`marking/README.md`](marking/README.md)
has the detail.

**This restructure (2026-10-02)** joined the two halves into one
pipeline and fixed what joining them exposed:

- `build_booklet.py` filed every page whole under the last question
  heading on it, so short answers sharing a page went to the wrong
  question. The fix is covered by tests.
- The marking code crashed on unlabelled answers and on booklets with no
  examiner marks.
- Rebuilding pages from the raw scans does not reproduce the published
  pages exactly. This is recorded, and still open.

Next steps, in order: [`reading/docs/TODO.md`](reading/docs/TODO.md) and
[`marking/IMPROVEMENTS.md`](marking/IMPROVEMENTS.md).

---

## Documentation

| file | read it for |
|---|---|
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | every stage: what it reads, writes and checks, and what invalidates what |
| [`docs/DATA.md`](docs/DATA.md) | the data root, the Hugging Face repos, the token, the privacy rules |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | the contract between reading and marking |
| [`reading/README.md`](reading/README.md) | how pages become booklets, and the design facts behind it |
| [`marking/README.md`](marking/README.md) | how booklets are marked, the results, the traps |
| [`reading/docs/DONE.md`](reading/docs/DONE.md) | what was measured, and the decisions it settled |
| [`marking/docs/`](marking/docs/) | the marking architecture, data formats, glossary, local model setup |
| [`CLAUDE.md`](CLAUDE.md) | conventions for working on the code (also read by Claude Code) |

`docs/slides/` holds the two review decks, and `docs/history/` holds the
marking half's git history from before it joined this repository (a git
bundle: `git clone docs/history/major_project_eval_history.bundle`).

---

## History

This repository grew as two halves built in parallel: a reading half
and a marking half, the second developed in its own repository first.
Until 2026-10-02 `main` held a second, independent reading
implementation, a line-segmentation and page-reading pipeline that
produced the September handoff the marking results above were measured
on. It was retired from this repository when this branch became `main`,
and is kept as a separate archive copy. Everything else, including every
file removed in the restructure, is in this repository's git history.
