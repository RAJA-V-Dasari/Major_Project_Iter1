# Annotation

Ground truth for layout segmentation of the handwritten answer scripts.

Everything here is reproducible from `preprocessing/output/` — no
image is stored in git, only the choices made about them.

| File | What it is |
|---|---|
| `LABELING_GUIDE.md` | **Read first.** Class definitions and edge cases |
| `sample_pages.py` | Chooses which pages to annotate |
| `manifest.csv` | The chosen 120 pages, with the train/val/test split |
| `make_review_sheets.py` | Contact sheets, for surveying the sample |
| `preannotate.py` | Candidate boxes as COCO, to start from |
| `prepare_upload.py` | Packages the pages as a zip for CVAT |
| `validate_labels.py` | Checks an export against the guide's rules |
| `build_dataset.py` | Corrected labels → YOLO training tree |
| `train_layout.py` | Fine-tunes YOLO11n on the labels |
| `evaluate.py` | Scores a trained model on a held-out split |
| `pseudo_label.py` | Labels the whole corpus with a trained model |
| `cvat/` | Low-memory compose override |
| `images/` | Symlinks to the sampled pages *(gitignored — PII)* |
| `review/`, `preannotations/`, `dataset/` | Generated *(gitignored)* |

---

## The sample

120 pages out of 1385, chosen by `sample_pages.py` with a fixed seed.

```
train    43 students    83 pages
val       9 students    19 pages
test      9 students    18 pages
```

**All 61 students appear.** Handwriting varies far more between
students than between pages by one student, so coverage of writers was
prioritised over volume of pages.

**The split is by student, not by page.** Pages from one booklet share
handwriting, ruling, scan quality and subject matter, so a page-level
split would put near-identical pages either side of the train/test line
and report a score the model has not earned. It is fixed in the
manifest before any measurement is taken, which is the only point at
which it can be chosen honestly.

15 of the 120 are cover pages — capped deliberately. Covers are ~11% of
the corpus but are the same printed form every time, so annotating them
in proportion would buy almost no variety.

---

## Workflow

### 1. Build the sample

```bash
cd annotation
python3 sample_pages.py
```

Re-running with the same seed reproduces the identical set. **Changing
`SEED` or `--size` reshuffles everything and invalidates annotation
already done** — treat both as frozen once labelling starts.

### 2. Read the guide

`LABELING_GUIDE.md`. The two rules that cause the most damage when
missed:

- **Text inside a figure stays inside the figure** — do not carve
  diagram labels out as `paragraph`.
- **`crossed_out` is an overlapping layer**, not a sixth exclusive
  class. Label the region normally, *then* add a `crossed_out` box over
  the cancelled span.

### 3. Generate pre-annotations

```bash
python3 preannotate.py
```

Writes `preannotations/preannotations_coco.json` (481 boxes over 120
pages, ~4.0 per page) and `preannotations/cvat_labels.json`.

Every box comes out as `paragraph`. That is deliberate — see
*Why the boxes are unlabelled* below.

### 4. Annotate in CVAT

CVAT **self-hosted**, because these pages carry student names, USNs,
signatures and marks. Do not use a hosted annotation service.

#### Install Docker (needs sudo — your step)

```bash
sudo pacman -S docker docker-compose
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"     # then log out and back in
docker info                          # should print without sudo
```

#### Package the pages

```bash
python3 prepare_upload.py
```

`images/` holds symlinks into `preprocessing/output/`, so an upload that
does not follow links would send 120 broken files. This writes
`preannotations/cvat_images.zip` (120 pages, 616 MB, dereferenced).

#### Start CVAT

```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Then `http://localhost:8080`:

1. **Create task** → paste the six labels from
   `preannotations/cvat_labels.json` (Raw tab)
2. Upload `preannotations/cvat_images.zip`
3. **Actions → Upload annotations → COCO 1.0** →
   `preannotations_coco.json`
4. Correct boxes; export as **COCO 1.0** into `annotation/labels/`

Image order in the task does not matter — COCO binds annotations to
`file_name`, not position.

#### If it thrashes

CVAT asks for 8 GB; this host has 7 GB. Try the stock stack first and
keep it if it works. If memory becomes the problem,
`cvat/docker-compose.override.yml` here turns analytics off and drops
each service to one process, and the analytics containers can then be
skipped at start:

```bash
docker compose -f docker-compose.yml \
  -f /path/to/annotation/cvat/docker-compose.override.yml \
  up -d cvat_db cvat_redis_inmem cvat_redis_ondisk

docker compose -f docker-compose.yml \
  -f /path/to/annotation/cvat/docker-compose.override.yml \
  up -d --no-deps cvat_opa cvat_server cvat_ui traefik \
    cvat_worker_import cvat_worker_export cvat_worker_utils \
    cvat_worker_annotation cvat_worker_chunks
```

This skips clickhouse, vector and grafana (~1.5 GB) plus the webhook,
quality-report and consensus workers, none of which the annotation
workflow uses. `--no-deps` is required because every backend service
lists clickhouse under `depends_on`.

**Untested here** — Docker is not installed on this machine, so this
variant is derived from reading CVAT's compose file (commit `a15a51e`),
not from running it.

### 5. Validate the export

```bash
python3 validate_labels.py labels/instances_default.json
```

Exits non-zero on any error, so it can gate a training run. It enforces
the guide's two structural rules directly:

- a `crossed_out` box that overlaps no region — meaning the underlying
  region was **deleted** instead of kept, losing the good text around
  the cancellation
- a `paragraph`/`math`/`code` box nested inside a `figure` or `table` —
  the carve-out error

plus out-of-bounds boxes, unknown labels, suspiciously tiny boxes, and
how much of the manifest is still unannotated.

### 6. Build the training set

```bash
python3 build_dataset.py labels/instances_default.json
```

Writes `dataset/` in Ultralytics YOLO layout with `data.yaml`. **The
split is read from the manifest, never recomputed** — re-splitting at
page level here is the easy mistake and would inflate the score.

It warns about classes with no examples (unlearnable) or under 30
(poor recall expected).

### 7. Train

```bash
../.venv/bin/python train_layout.py --smoke     # 2 epochs, proves plumbing
../.venv/bin/python train_layout.py             # real run
../.venv/bin/python evaluate.py                 # scores on val
```

Environment is `.venv` at the repo root — Python 3.11 via `uv`, with
**CPU-only** torch. System Python is 3.14, which has no torch wheels
(this previously failed on `torchvision::nms`); and there is no NVIDIA
GPU here, so the CUDA build would be 2.5 GB of dead weight.

**The pipeline is verified end-to-end** — a smoke run trains, saves
weights and evaluates cleanly on the pre-annotation data. The scores it
produces are meaningless (one class, two epochs); it proves plumbing,
not learning.

#### Expect it to be slow

Measured here: **15.5 s/epoch at 320px** on 83 images, 6 CPU threads.
Compute scales with the square of image size, so at the default 1024px
that is roughly **2.5 min/epoch — about 4 hours for 100 epochs.**

Fine overnight, painful to iterate on. For tuning, drop to
`--imgsz 640`; for a final model, a GPU box is worth it and the same
command picks the GPU up with no changes.

Do not go below ~640: a struck word is only ~40px wide on a 1700px
page, and `crossed_out` stops being resolvable.

#### Augmentation is deliberately restricted

`train_layout.py` disables horizontal and vertical flips, and mosaic.
A mirrored answer page is not a thing that exists — the originals were
already flipped once by the scanner and that was a *bug*, not variety.
Mosaic stitches four pages into one, destroying page-level layout,
which is the entire signal here. Mild rotation stays on, because real
scans genuinely are tilted a couple of degrees.

### 8. Extend to all 1385 pages

Hand-labelling 120 pages does not mean training on only 120. The other
~1265 come in through **bootstrapping**, once a model exists:

```bash
../.venv/bin/python pseudo_label.py --weights runs/layout/weights/best.pt \
    --human labels/instances_default.json
```

- pages with human labels are **skipped, never overwritten**
- only detections at or above `--conf` are written
- pages the model was unsure about go to `preannotations/pseudo_review.txt`
- every box keeps its `score`, so the bar can be raised later without
  re-running inference

Then hand-fix the review list, merge, and retrain.

**Why not just auto-label everything with the classical classifier?**
Because it is measurably wrong — over-calling `table` ~3x and barely
finding `figure` (numbers below). A model cannot come out better than
the labels it was trained on, so that route produces a slower copy of a
classifier we already know is broken. Bootstrapping works because the
model learns from *correct* human examples first; its errors on new
pages are ordinary generalisation error, not a bias stamped onto every
page.

`preannotate.py --all` does run the classical pass over the full corpus
(~15 min, parallel), and it is useful as a geometry starting point —
but treat its **classes** as a draft, not as ground truth.

### 9. Track progress

The manifest is the checklist. Annotate `train`, `val` and `test`
identically — and do not look at `test` scores while tuning thresholds.
`evaluate.py` defaults to `val` and requires an explicit `--test` for
exactly this reason.

---

## Why the boxes are unlabelled

`preannotate.py` can emit predicted classes (`--classify`). By default
it does not, and that is a measured decision rather than caution.

Running the `scan_doc_v2` classifier over **all 1385 corpus pages**
(5503 boxes):

| Class | Predicted | Reality |
|---|---|---|
| `paragraph` | 2470 (44.9%) | plausible |
| `table` | 1503 (27.3%) | **~3x too many** — real tables are ~1 page in 10 |
| `math` | 1292 (23.5%) | plausible, unverified |
| `figure` | 238 (4.3%) | **far too few** — ~1 page in 3 has a diagram |
| `code` | 0 | **structurally impossible** — no such class exists in it |
| `crossed_out` | 0 | see below |

Four of six classes at best. It over-calls `table` on plain prose
(handwriting produces enough vertical strokes to read as column
dividers; CRC long division especially so) and almost never finds
`figure`.

**`crossed_out` was wired in and then removed, on evidence.** The
per-line strikethrough test from `classify_blocks` was added to this
path and audited against 18 randomly sampled detections on real corpus
pages: **zero were genuine strikethroughs.** Every one was a diagram
arrow, a box edge, or ordinary prose.

That failure is structural rather than a threshold needing a nudge. The
signal is "a short dense horizontal pen stroke", and the edges and
arrows of hand-drawn diagrams are exactly that. It scored well on the
old 300 DPI test set only because that set contained almost no
diagrams; this corpus is full of them. It is the same confusion the
labelling guide warns annotators about, appearing in the detector.

Annotators anchor on whatever is already on screen: a confident-looking
wrong label gets accepted more often than a blank one gets missed. Seeding
ground truth with these predictions would bake that error into the data
the model is later scored against — so the geometry is kept and the
naming is not.

**Block geometry, by contrast, is sound** and worth starting from. One
caveat: blocks are built from runs of ruled text lines, so a figure or
table typically arrives split across two or three boxes that need
merging.

---

## Known limitations

- **~4.0 boxes/page is over-segmented** for figures and tables.
  Expect merging, not just relabelling.
- **18 pages have no dependable ruling** (cover sheets, near-empty
  pages). `preannotate.py` lists them; their boxes are weaker.
- **Pitch is unreliable, in both directions.** Where a student writes
  on alternate rules, autocorrelation locks onto the 2x harmonic and
  reports ~116 against a true ~58; on sparse pages the peak barely
  clears the confidence floor and it under-reports (`s19_c2_p14.png`
  measures 22). Anything keyed to pitch inherits that error — which is
  why the block size floor is a physical length instead.
- **Pure-diagram pages fall back to the ink extent.** A page that is
  one big drawing has its long horizontals stripped as printed ruling,
  leaving no ink rows to profile. Rather than emit nothing, those pages
  get a single box around the extent of the handwriting, which for a
  whole-page figure is close to right.
- **Boxes, not polygons.** The downstream consumer routes regions to
  handlers and does not need pixel-accurate outlines; polygons cost
  roughly 3x the effort for a gain this pipeline will not use.
- **Scans are slightly rotated and page-warped**, so boxes around
  tilted content carry some slack. Not worth compensating by hand.
- **`code` and `figure` will be scarce** even after annotation. If the
  trained model is weak on them, the fix is targeted extra sampling for
  those classes, not more random pages.

---

## Privacy

`images/`, `review/` and `preannotations/` are gitignored: they are
readable reproductions of real answer scripts. `manifest.csv` and the
exported labels carry geometry and filenames only, and are safe to
track.

Source data lives in a **private** Hugging Face repo. Keep it that way.
