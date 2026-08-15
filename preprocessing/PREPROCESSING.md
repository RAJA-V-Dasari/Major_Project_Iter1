# Dataset preprocessing — context and decisions

Normalising the Hugging Face answer-script dataset into a predictable
tree for the segmentation pipeline.

**Scope of this pass: students 1–31 only.** The repo contains more
(see [Out of scope](#out-of-scope)); they were deliberately left alone.

---

## Source

| | |
|---|---|
| Repo | `prss-majorproject-37/Handwritten-AnswerScripts-MajorProject` |
| Type | `dataset`, **private** |
| Auth | fine-grained token in `.env` as `HF_TOKEN` (gitignored) |
| Files | 1194 total — 1086 `.png`, 104 `.pdf`, 1 `.md`, 3 no-extension |

Load the token before running anything:

```bash
set -a; . ../.env; set +a
```

### Token gotcha

The first token failed with a **404, not a 403** — HF hides repos you
lack permission for. The token authenticated fine and reported org
membership, but its scoped permissions for `org:prss-majorproject-37`
were an empty list. If you hit a 404 on a repo you know exists, check
token scopes before assuming the repo name is wrong.

---

## Output layout

```
preprocessing/output/
    student_<N>/          N = 1..31
        cie_<M>/          M = 1..3
            page_1.png
            page_2.png
            ...
```

Gitignored — see [Privacy](#privacy).

---

## The four inconsistencies in the source

### 0. Orientation — do not "fix" it

**The source pages are correctly oriented. No flip, rotation, or
per-page orientation guessing is needed or wanted.**

This cost two wrong fixes, so it is worth stating why.

The first version of the converter pulled pages out of PDFs with
`extract_image()`, which returns the **raw stored image bytes and
ignores the page's transformation matrix**. Scanner PDFs routinely
store the scan flipped and carry a negative-Y matrix that turns it back
the right way at render time. Bypassing the matrix produced mirrored
pages **from PDF sources only** — PNG sources came through fine.

That mixed result was then misdiagnosed twice:

1. as a 180° rotation — wrong operation; rotating leaves the text
   mirrored, which is invisible on a thumbnail and only shows when you
   zoom into the printed header;
2. as a uniform flip — wrong assumption; applying it "corrected" the
   PDF-sourced pages and **broke the PNG-sourced ones**, producing
   exactly the "some flipped, some not" state that gave the game away.

The fix is to **render** the page (`get_pixmap`), which applies the
matrix, rather than extract the embedded bytes.

> If pages ever look flipped again, **suspect the extraction path
> before adding any transform.** A blanket transform can only ever be
> right if every page is wrong in the same way — verify that first
> with `check_orientation.py`.

Rendering still has to hit native resolution, so each page is rendered
at the zoom reproducing its own embedded image size (see
[Resolution](#3-resolution--why-we-do-not-render-at-a-fixed-dpi)).

### Verifying orientation

`check_orientation.py` measures **every page individually** rather than
sampling. It exploits the fact that handwriting on a ruled line is
vertically asymmetric — ink sits mostly just above the rule — so the
ink's centre of mass within each text band flips when the page does.

It calibrates against the corpus median and reports pages that
*disagree with the majority*, rather than using a hardcoded threshold
that would not transfer to a different ruling.

```bash
python check_orientation.py                  # measure all, list outliers
python check_orientation.py --contact-sheet  # render outliers to eyeball
```

It flags candidates for human review; it is not authoritative.

### 1. Mixed formats

Booklets arrive as any of:

- one PDF per page (`Scan_20260803 (2).pdf`)
- a single multi-page PDF (`document.pdf`)
- already-PNG pages (`document-0000.png`)
- **a PDF with no file extension at all** (`Student 20/CIE - 1`,
  ~20 MB, sniffed as `%PDF-` from its magic bytes)

So format is detected by **content**, not by extension.

### 2. Page ordering

Scanner output names pages `Scan_x.pdf`, `Scan_x (2).pdf`, …
`Scan_x (10).pdf`. Sorting these lexicographically is wrong twice over:
it puts `(10)` before `(2)`, and puts the **unnumbered file — which is
page 1 — last**.

`sort_key()` parses the trailing index and treats a bare name as index
1, keying on `(stem, index)` so pages from different scan sessions in
one folder stay grouped and internally ordered.

### 3. Resolution — why we do *not* render at a fixed DPI

The PDFs disagree about page geometry while holding identical scans:

| Source | Page box | Embedded image |
|---|---|---|
| `Student 4/CIE - 1/Scan_20260803.pdf` | 1700×2338 pt | 1700×2338 px |
| `Student 20/CIE - 1` | 612×842 pt | 1700×2338 px |

Rendering both at 200 DPI would leave Student 20 correct and **upscale
Student 4 by ~2.8×**, inventing detail that is not in the scan.

Instead the converter **extracts the embedded image** when a page holds
exactly one — lossless, native pixels, geometry-independent. It falls
back to rasterising at 200 DPI only for pages that are not a single
full-page image.

Native scan resolution is **1700×2338 px** ≈ A4 at 200 DPI.

> ### ⚠️ This breaks `scan_doc_v2`
>
> `modules/scan_doc_v2` was tuned on the older 300 DPI sample
> (`initial_dataset/CIE_Dataset_RV.pdf`, 2480×3509). Its thresholds are
> in **absolute pixels** — pitch ≈89 px, `MATH_MAX_COMPONENT_WIDTH`
> = 29 px, stroke probes, `MIN_SIDE`.
>
> This dataset is 200 DPI, so every one of those shrinks by ⅓
> (pitch → ~59 px, component width → ~19 px). The pipeline will **not
> error — it will silently misclassify.**
>
> Fix before running v2 on this data: derive thresholds from the
> measured `pitch` rather than hardcoding pixels. `pitch` is already
> measured per page by `normalize_page.py`, so the ratios are available.

---

## Per-booklet decisions

### Student 4 / CIE - 1 — duplicate scans dropped

The only folder mixing PDF and PNG: two `Scan_20260803*.pdf` plus eight
`Scan_20260804*.png`.

Checked by rendering both: the PDFs are **re-scans of the same first
two pages** the PNGs already cover — same booklet number 241713, same
cover sheet, same "1 b. a host." page. The 8 PNGs are a complete
booklet on their own.

**Decision:** keep the PNGs, drop the PDFs. Appending them would have
duplicated pages 1–2. Encoded in `FORMAT_OVERRIDES`; if a similar
folder appears later, verify visually before adding an entry — do not
assume "PDF + PNG" always means duplicates.

### Student 20 / CIE - 1 and CIE - 2 — extensionless PDFs

Listed as bare files, not folders, so they first looked like empty
placeholders. They are 20 MB and 18 MB **multi-page PDFs** (12 pages
each) whose extension was lost on upload. Recovered by magic-byte
sniffing; **not** missing data.

### Student 25 / CIE - 2 — single multi-page PDF

One `document.pdf` rather than the usual per-page files. Expands
normally.

---

## Missing data (students 1–31)

| Student | Status |
|---|---|
| **19** | **Entire student absent** — no `Student 19` in the repo |
| **18** | Only CIE-1 (4 pages). CIE-2, CIE-3 absent |
| **20** | CIE-1/2/3 all present *(the extensionless files are real PDFs)* |

All other 28 students have three complete CIEs.

Per instruction, nothing was fabricated or substituted for the gaps —
they are simply absent from the output tree.

---

## Out of scope

Present in the repo, intentionally **not** converted in this pass:

- `t 1` … `t 12` — some hold real booklets (`t 1` has CIE-2 and CIE-3);
  others only a single `document.pdf` per CIE
- `Student 1001` … `Student 1018`

Extending the run is just `FIRST_STUDENT` / `LAST_STUDENT` in
`convert_dataset.py`, plus a mapping decision for the `t N` folders —
their numbering does not continue the `Student N` sequence, so how they
should be renamed is an open question.

---

## Privacy

These are real student answer booklets. Cover sheets carry **names,
USNs, signatures, and marks**.

- The source HF repo is **private**.
- `preprocessing/output/` and `preprocessing/.hf_cache/` are
  **gitignored**; verified with `git check-ignore`.
- `.env` (holding the token) is gitignored and untracked.

Do not commit converted pages, publish them, or push them to a public
HF repo.

---

## Usage

```bash
set -a; . ../.env; set +a

python3 convert_dataset.py --check          # report only, writes nothing
python3 convert_dataset.py                  # convert students 1..31
python3 convert_dataset.py --students 4,20  # a subset
```

`--check` reports source-file counts, so a multi-page PDF shows as
`1p` there and expands to its true page count on the real run.

Requires `pymupdf`, `pillow`, `huggingface_hub`.
