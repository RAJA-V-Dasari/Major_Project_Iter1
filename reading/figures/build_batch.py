"""
Package pages for the diagram-finding pass.

    python reading/figures/build_batch.py              # every read page
    python reading/figures/build_batch.py --first 5    # students 1-5

    data/pages/ + data/read/<engine>/  ->  data/figures/diagram_batch.zip

Then run reading/figures/find_diagrams.ipynb on a Colab T4 and unzip its
download into data/figures/, so diagrams.json lands where
build_booklet.py looks for it.

WHAT THIS IS FOR
----------------
The reading pass emits a `![diagram: ...]` marker on only 52 of 1,000
pages. That is not evidence that the model cannot see drawings - finding
them was a side clause in a prompt whose real job was transcription. The
consequence downstream is worse than a low count: `build_booklet.py`
pairs the nth marker to the nth region, so when the markers run out the
leftover regions get appended at the END of the page instead of sitting
in the answer. That is why figures are currently misplaced.

This batch feeds a pass whose ONLY job is to enumerate drawings, and its
output replaces the reading pass as the marker source.

WHICH PAGES
-----------
Only pages that have a transcription under `data/read/<engine>/`,
because a drawing is placed by matching its anchor text against that
transcription - a page with no text to match is a page this pass cannot
help. With `all_read` that is 1,000 pages.

THE BENCHMARK COMES FOR FREE
----------------------------
The hand-labelled pages from `labels.py` beside this file are inside
the batch, and their truth is written into `batch_truth.json` as
fractions of page height, so the notebook scores itself without anything
else being uploaded.

Privacy: `page_01` is the identity block and is excluded by
construction, as everywhere else that touches the corpus.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import zipfile

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.exists(os.path.join(ROOT, "common", "layout.py")):
    ROOT = os.path.dirname(ROOT)

sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)                    # labels.py

from common import layout                   # noqa: E402

UPLOAD = str(layout.FIGURES)
STAGE = str(layout.FIGURES / "_staging")

TONE = str(layout.PAGES)
READ = str(layout.READ)

# The model is shown a page this tall. Full 2177px buys nothing for
# spotting an object the size of a diagram and costs visual tokens.
LONG_EDGE = 1280


def page_id(path):
    q = path.replace("\\", "/").split("/")
    return (f"s{int(q[-3].split('_')[1]):02d}"
            f"_c{int(q[-2].split('_')[1])}"
            f"_p{int(os.path.splitext(q[-1])[0].split('_')[1]):02d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=layout.DEFAULT_ENGINE,
                    help="which read output to mirror - the folder under "
                         f"data/read/ (default {layout.DEFAULT_ENGINE})")
    ap.add_argument("--first", type=int,
                    help="limit to the first N students, for a trial run")
    args = ap.parse_args()

    from labels import DRAWN, PROSE_ONLY

    # Only pages that were actually read. The anchor text this pass
    # returns is matched against the transcription, so a page without
    # one has nothing to be placed into.
    engine_dir = os.path.join(READ, args.engine)
    if not os.path.isdir(engine_dir):
        raise SystemExit(f"no read output at {engine_dir}")
    read_ids = {os.path.splitext(f)[0] for f in os.listdir(engine_dir)
                if f.endswith(".md")}
    print(f"{len(read_ids)} pages transcribed under {args.engine}")

    pages = sorted(glob.glob(os.path.join(
        TONE, "student_*", "cie_*", "page_*.png")))
    pages = [p for p in pages if not p.endswith("page_01.png")]
    if args.first:
        pages = [p for p in pages
                 if int(p.replace("\\", "/").split("/")[-3].split("_")[1])
                 <= args.first]
    pages = [p for p in pages if page_id(p) in read_ids]

    if not pages:
        raise SystemExit("no pages found - is the corpus junction present?")

    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(os.path.join(STAGE, "pages"), exist_ok=True)

    truth = {}
    manifest = {}

    for path in pages:
        pid = page_id(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h0, w0 = img.shape
        scale = LONG_EDGE / max(h0, w0)
        small = cv2.resize(img, (int(round(w0 * scale)),
                                 int(round(h0 * scale))),
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(STAGE, "pages", pid + ".png"), small)
        manifest[pid] = {"h0": h0, "w0": w0}

        # hand truth, expressed as fractions of page height so the
        # notebook never has to know the original pixel size
        if pid in DRAWN:
            truth[pid] = {"prose_only": False,
                          "drawn": [[a / h0, b / h0] for a, b in DRAWN[pid]],
                          "drawn_px": DRAWN[pid], "h0": h0}
        elif pid in PROSE_ONLY:
            truth[pid] = {"prose_only": True, "drawn": [], "drawn_px": [],
                          "h0": h0}

    json.dump(truth, open(os.path.join(STAGE, "batch_truth.json"), "w",
                          encoding="utf-8"), indent=2)
    json.dump(manifest, open(os.path.join(STAGE, "manifest.json"), "w",
                             encoding="utf-8"), indent=2)

    os.makedirs(UPLOAD, exist_ok=True)
    zpath = os.path.join(UPLOAD, "diagram_batch.zip")
    if os.path.exists(zpath):
        os.remove(zpath)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) \
            as z:
        for base, _, files in os.walk(STAGE):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, os.path.relpath(full, STAGE))

    shutil.rmtree(STAGE)

    n_drawn = sum(1 for v in truth.values() if not v["prose_only"])
    n_regions = sum(len(v["drawn"]) for v in truth.values())
    print(f"{len(manifest)} pages packed (page_01 excluded)")
    print(f"benchmark inside the batch: {n_drawn} drawn pages "
          f"({n_regions} regions), "
          f"{sum(1 for v in truth.values() if v['prose_only'])} prose pages")
    print(f"wrote {zpath}  ({os.path.getsize(zpath) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
