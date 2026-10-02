"""
Build a contact sheet of every cropped figure, for one human pass.

    python reading/assemble/review_figures.py
    start data/figures/review.html

Then click any crop that is not a drawing, save `rejects.json` where the
page tells you (data/figures/rejects.json), and rebuild:

    python pipeline.py assemble

build_booklet.py picks data/figures/diagrams.json and rejects.json up on
its own, so the rebuild needs no flags.

WHY THIS EXISTS
---------------
"The diagrams definitely have to be diagrams" is a claim about
precision, and no detector can supply it. The diagram pass is good -
on the hand-labelled pages it left every prose page alone - but good is
measured on sixteen pages, and the corpus is a thousand. A reviewer
clicking through three hundred crops takes half an hour and turns the
claim into a measurement: the reject rate IS the precision, and what is
left is verified rather than estimated.

It reads the `figures.json` that `build_booklet.py` writes beside each
booklet, so it never recomputes a crop or re-runs the segmentation - it
shows exactly the images that are in the Markdown.

PRIVACY
-------
The crops are pieces of real student pages, so `review.html` is a page
image wearing a text file's extension. It lands under `data/figures/`,
which is gitignored with the rest of `data/`. Do not move it somewhere
that is not.
"""

import argparse
import base64
import html
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "common" / "layout.py").exists())))

from common import layout                                  # noqa: E402

OUT_DIR = layout.FIGURES
BOOKLETS = layout.BOOKLETS

# Crops are shown at this width. Wide enough to judge whether something
# is a drawing, small enough that three hundred of them fit in one file.
THUMB_W = 400

CSS = """
* { box-sizing: border-box; }
body { font: 14px/1.5 system-ui, sans-serif; margin: 0;
       background: #f2f2ef; color: #1b1b1a; }
header { position: sticky; top: 0; z-index: 5; background: #fff;
         border-bottom: 1px solid #d8d8d4; padding: 14px 22px;
         display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
h1 { font-size: 17px; margin: 0; }
.count { font-variant-numeric: tabular-nums; color: #555; }
.count b { color: #1b1b1a; }
button { font: inherit; padding: 6px 14px; border: 1px solid #bbb;
         background: #fff; border-radius: 5px; cursor: pointer; }
button:hover { background: #f0f0ee; }
.hint { color: #666; padding: 14px 22px 0; max-width: 70ch; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; padding: 14px 22px 60px; }
figure { margin: 0; width: 420px; background: #fff; padding: 8px;
         border: 2px solid #ddd; border-radius: 6px; cursor: pointer;
         transition: border-color .1s, opacity .1s; }
figure img { width: 100%; display: block; border-radius: 3px;
             background: #fff; }
figure.out { border-color: #c0392b; opacity: .4; }
figure.out .mark::after { content: 'rejected'; color: #c0392b;
                          font-weight: 600; }
figcaption { font-size: 12px; color: #444; margin-top: 6px;
             display: flex; justify-content: space-between; gap: 8px; }
.pid { font-family: ui-monospace, monospace; color: #888; }
.tag { font-size: 11px; color: #777; }
dialog { border: 1px solid #bbb; border-radius: 8px; padding: 18px;
         max-width: 620px; width: 90%; }
textarea { width: 100%; height: 260px; font-family: ui-monospace, monospace;
           font-size: 12px; }
"""

JS = """
const KEY = 'figure_rejects_v1';
let out = new Set(JSON.parse(localStorage.getItem(KEY) || '[]'));

function paint() {
  let n = 0;
  document.querySelectorAll('figure').forEach(f => {
    const bad = out.has(f.dataset.key);
    f.classList.toggle('out', bad);
    if (bad) n++;
  });
  const total = document.querySelectorAll('figure').length;
  document.getElementById('kept').textContent = total - n;
  document.getElementById('cut').textContent = n;
  document.getElementById('rate').textContent =
    total ? ((n / total) * 100).toFixed(1) + '%' : '-';
  try { localStorage.setItem(KEY, JSON.stringify([...out])); } catch (e) {}
}

document.querySelectorAll('figure').forEach(f => {
  f.addEventListener('click', () => {
    const k = f.dataset.key;
    out.has(k) ? out.delete(k) : out.add(k);
    paint();
  });
});

function payload() {
  const by = {};
  [...out].forEach(k => {
    const i = k.lastIndexOf(':');
    const pid = k.slice(0, i);
    (by[pid] = by[pid] || []).push(parseInt(k.slice(i + 1), 10));
  });
  Object.values(by).forEach(v => v.sort((a, b) => a - b));
  return JSON.stringify(by, null, 1);
}

document.getElementById('save').addEventListener('click', () => {
  const text = payload();
  document.getElementById('blob').value = text;
  const a = document.getElementById('dl');
  a.href = URL.createObjectURL(new Blob([text], {type: 'application/json'}));
  a.download = 'rejects.json';
  document.getElementById('dlg').showModal();
});

document.getElementById('reset').addEventListener('click', () => {
  if (confirm('Clear every rejection and start again?')) {
    out = new Set(); paint();
  }
});

paint();
"""


def thumb(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    h, w = img.shape
    if w > THUMB_W:
        k = THUMB_W / w
        img = cv2.resize(img, (THUMB_W, max(1, int(h * k))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--booklets", default=str(BOOKLETS))
    ap.add_argument("--out", default=str(OUT_DIR / "review.html"))
    args = ap.parse_args()

    root = Path(args.booklets)
    manifests = sorted(root.glob("*/figures.json"))
    if not manifests:
        raise SystemExit(
            f"no figures.json under {root} - run build_booklet.py with "
            f"--diagrams first")

    cards, n = [], 0
    for man in manifests:
        booklet = man.parent.name
        for fig in json.loads(man.read_text(encoding="utf-8")):
            b64 = thumb(man.parent / fig["file"])
            if not b64:
                continue
            key = f"{fig['pid']}:{fig['index']}"
            n += 1
            cards.append(
                f'<figure data-key="{html.escape(key)}">'
                f'<img src="data:image/png;base64,{b64}" alt="" loading="lazy">'
                f'<figcaption><span>{html.escape(fig["caption"])}</span>'
                f'<span class="mark"></span></figcaption>'
                f'<figcaption><span class="pid">{html.escape(fig["pid"])}</span>'
                f'<span class="tag">{html.escape(booklet)} &middot; '
                f'{html.escape(fig["placed"])} &middot; '
                f'{html.escape(fig["box"])}</span></figcaption>'
                f'</figure>')

    out_path = Path(args.out)
    try:
        rel = os.path.relpath(out_path.parent / "rejects.json", Path.cwd())
    except ValueError:                  # a data root on another drive
        rel = str(out_path.parent / "rejects.json")

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Figure review</title>
<style>{CSS}</style>
<header>
  <h1>Figure review</h1>
  <span class="count">keeping <b id="kept">0</b> &middot;
    rejected <b id="cut">0</b> &middot; reject rate <b id="rate">-</b></span>
  <button id="save">Save rejects</button>
  <button id="reset">Start again</button>
</header>
<p class="hint">Click any crop that is <b>not</b> a drawing &mdash; a block
of handwriting, a ruled margin, a piece of nothing. Click again to undo.
Your choices are kept in this browser, so you can close the page and come
back. When you are done, press <b>Save rejects</b> and put the file at
<code>{html.escape(rel)}</code>.</p>
<div class="grid">
{chr(10).join(cards)}
</div>
<dialog id="dlg">
  <p>Save this as <code>rejects.json</code>, then rebuild the booklets.</p>
  <textarea id="blob" readonly></textarea>
  <p><a id="dl" href="#">Download rejects.json</a> &nbsp;
     <button onclick="document.getElementById('dlg').close()">Close</button></p>
</dialog>
<script>{JS}</script>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")

    size = out_path.stat().st_size / 1e6
    print(f"{n} crops from {len(manifests)} booklets")
    print(f"wrote {out_path}  ({size:.1f} MB)")
    print("open it, click the ones that are not drawings, press Save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
