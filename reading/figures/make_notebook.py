"""
Generate the Colab notebook for the diagram-finding pass.

    python reading/figures/make_notebook.py
        -> reading/figures/find_diagrams.ipynb

This pass replaces the reading pass as the source of figure markers. The
reading pass emits `![diagram: ...]` on 52 of 1,000 pages - finding
drawings was a side clause in a prompt whose real job was transcription -
so almost nothing gets paired with a region, and every unpaired region
gets dumped at the end of the page. This asks the question directly and
asks it as the only question.

nbformat `source` lists: every element except the last MUST keep its
trailing newline. A reader rejoins with "".join, so bare lines collapse
into one unreadable row. Colab is not forgiving about this; it cost a
debugging session once already, so there is a guard at the bottom.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "find_diagrams.ipynb")


# ---------------------------------------------------------------- prompt
#
# The prompt IS the system. Everything downstream is bookkeeping, so this
# is written out in full rather than kept terse, and every constraint in
# it was paid for by a measured failure:
#
#   * No headings in capitals. When the model has nothing to say it
#     echoes them back verbatim - two pages once returned nothing but a
#     capitalised heading from the reading prompt.
#   * No worked example of the output CONTENT, only of its shape. A
#     filled-in example primes the model to find that thing on blank
#     pages; an all-zeros example primes it to report nothing.
#   * ONE TEST, applied to everything: would this survive being retyped
#     as plain lines? That is the whole taxonomy, and it is phrased as a
#     question the model answers rather than a category it matches.
#
#     It exists because the content splits four ways and only the last
#     needs a person:
#
#       prose               retypes fine   -> transcribed, marked
#       table of text       reported       -> transcribed, marked
#       simple mathematics  retypes fine   -> transcribed, marked
#       everything else     reported       -> cropped, marked by a human
#
#     Mathematics falls on BOTH sides, which is why a rule about "maths"
#     would be wrong. `throughput = 1000 / 200 = 5 Mbps` reads left to
#     right and retypes without loss, so it is text and the grader can
#     mark it - that is the formula-substitution case the marking ladder
#     was built for. A CRC long-division staircase cannot be retyped at
#     all: the indentation IS the method. Same subject, opposite
#     handling, and the retyping test separates them without naming
#     either.
#
#   * `diagram` is defined as the RESIDUAL, never enumerated. Run 1
#     listed six example shapes and got back `packet-format strip` on 20
#     of 37 regions - the model answered from the nearer list. A
#     leftover category has nothing to echo and cannot miss a shape
#     nobody thought to list. parse() still snaps anything that is not a
#     table to `diagram`, so a future leak degrades to wrong-but-usable.
#   * The span is asked for generously. Run 1 returned stubs: correctly
#     placed at the top of the real region but about half its height,
#     which cost most of the benchmark misses. A crop carrying a spare
#     line of text is fine in the finished Markdown; a crop that clips an
#     arrow is not, so the instruction leans one way on purpose.
#   * `anchor` is the field that actually places the figure. Qwen reads
#     far better than it measures - run 1 snapped every coordinate to a
#     multiple of 0.05 - so the insertion point in the Markdown comes
#     from words it READ rather than a number it guessed. Matched
#     locally with difflib against the page's transcription.
#
PROMPT = """\
You are looking at a single page from a handwritten exam answer booklet.
A student has written their answers by hand on ruled paper, and the page
has already been transcribed by someone else. Your job is to find the
parts of the page that ordinary typed text cannot carry.

Do not transcribe the page. Do not read the answer, judge it, or say
whether it is correct. Find those parts, say where each one sits, and
stop.

The test to apply

For each part of the page, ask one question: if this were retyped as
plain lines of text, would it still mean the same thing?

Most of the page passes. Sentences, paragraphs, bulleted lists, headings
and ordinary lines of mathematics all retype without loss. A line like
throughput = 1000 / 200 = 5 Mbps means exactly what it meant on paper,
because it reads from left to right and nothing about where it sits on
the page matters. Do not report any of that. It is writing, however
untidy or crooked or hard to read.

Some parts fail. Their meaning lives in how the marks are arranged, so
retyping them destroys it. Those are what you are looking for.

The two kinds

  table     the student ruled a grid, in rows and columns, and wrote
            inside its cells

  diagram   anything else that fails the test

Diagram is deliberately the leftover category. Do not try to decide what
sort of drawing it is or whether it has a name you recognise. If the
arrangement carries the meaning and it is not a ruled grid, it is a
diagram.

Things that fail the test are often, but not always, drawn rather than
written: shapes joined by lines, sketches, charts. But written marks can
fail it too. Mathematical working laid out as a staircase, where each
row is indented to sit under part of the row above, is a diagram - the
indentation is the method, and a retyped column of digits would lose it.
So is a calculation whose columns must line up to be read.

A part that fails the test often has words inside it: names on shapes,
numbers along lines, a label underneath. Those words do not make it
writing. Judge by whether the arrangement matters, not by what the marks
say.

Some of these are faint, crooked, or drawn with a ruler that slipped.
Some are only a handful of strokes with a lot of blank paper around
them. A sparse one still counts, and sparse ones are the ones most often
missed, so look carefully wherever the writing stops and the page opens
out.

What to report for each one

  kind      exactly one of two words and nothing else:
            table if the student ruled a grid and wrote inside its cells
            diagram for everything else you are reporting

  from      where its top edge sits, as a fraction of the page height

  to        where its bottom edge sits, as a fraction of the page height

  caption   a few words saying what it shows, in your own words, as a
            caption would read underneath it

  anchor    the last few words of handwriting immediately above it,
            copied from the page exactly as the student wrote them

About from and to

Fractions run down the page. 0.0 is the very top edge of the paper, 0.5
is halfway down, 1.0 is the very bottom. Judge them by eye; close is
good enough, and there is no need for more than two decimal places.

Measure from its topmost mark to its lowest, and take in any labels or
caption the student wrote around it. When you are unsure where an edge
falls, choose the more generous number. Including a line of ordinary
writing above or below costs nothing. Cutting off the bottom row ruins
it.

About anchor

The anchor is how it gets put back in the right place in the transcribed
text, so it matters that the words are really on the page. Copy five or
six words of handwriting from just above it, spelling and mistakes and
all. Do not paraphrase them and do not tidy them up. If it sits at the
very top of the page with no writing above it, leave the anchor empty.

Counting them

Report one entry for each separate one, in the order they appear down
the page, top first. Two of them one above the other are two entries
even when they show related things. One whose parts are joined together
is a single entry however many shapes it contains.

How to reply

Reply with a JSON array in exactly this shape, and no other text before
or after it:

[{"kind": "", "from": 0.0, "to": 0.0, "caption": "", "anchor": ""}]

Use only the word diagram or the word table for kind. If everything on
this page would survive being retyped, which is true of most pages,
reply with an empty array and nothing else.
"""

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW = 400
CKPT_EVERY = 25

CELLS = []


def lines_of(src):
    return src.strip().splitlines(keepends=True)


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": lines_of(src)})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {},
                  "execution_count": None, "outputs": [],
                  "source": lines_of(src)})


md("""
# What on this page cannot be typed

One pass over every read page, asking a single question of everything on
it: **would this survive being retyped as plain lines of text?**

Whatever would not is reported, and its output becomes the figure source
for `build_booklet.py`, replacing the markers the reading pass emits.

### The four kinds of content, and why only one needs a person

| | retypes? | handled by |
|---|---|---|
| prose | yes | transcription, then the marking ladder |
| a table of text | no, but it is a grid | transcription as a Markdown table |
| simple mathematics | yes | transcription, then method marks |
| everything else | no | **cropped, and a human marks it** |

Mathematics sits on both sides, which is why this asks about retyping
rather than about subject matter. `throughput = 1000 / 200 = 5 Mbps`
reads left to right and loses nothing when typed, so it is text and the
grader marks it. A CRC long-division staircase cannot be typed at all -
the indentation *is* the method - so it is cropped and a person marks
it. Same subject, opposite handling, one test that separates them.

### Why

The reading pass marks **52 of 1,000 pages (5%)**. Finding figures was a
side clause in a prompt whose real job was transcription, so almost no
marker ever gets paired with a region - and `build_booklet.py` dumps
every unpaired region at the *end* of the page block instead of in the
answer. That is why figures are currently in the wrong place.

Asked as the only question on a 131-page trial, the same model fired on
**24%** of pages, hit **3/3** genuinely drawn hand-labelled pages, left
**4/4** prose pages alone, and found two sparse diagrams on `s04_c3_p03`
that the geometry rule scores **0.00** on.

### The `anchor` field

New this run, and the reason placement will work. The model reads far
better than it measures - the trial snapped every coordinate to a
multiple of 0.05 - so each drawing also reports **the last few words of
handwriting above it**. Locally those words are fuzzy-matched against
the page's transcription to find the insertion point. Position comes
from words it read, not a number it guessed.

### Built-in benchmark

The 16 hand-labelled pages are inside this batch, so the notebook scores
itself at the end with nothing else uploaded.

**Runtime → Change runtime type → GPU** before running.
""")

code("""
!pip -q install -U transformers accelerate bitsandbytes qwen-vl-utils
import torch, transformers
print('transformers', transformers.__version__)
print('cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
""")

md("""
## 1. Drive, for the checkpoint

A thousand pages is long enough that a disconnect is likely. If Drive
mounts, progress is saved there every 25 pages and a re-run picks up
where it stopped. If it does not mount the run still works - it just
starts over after a disconnect.
""")

code("""
import os

CKPT_DIR = '/content'
try:
    from google.colab import drive
    drive.mount('/content/drive')
    CKPT_DIR = '/content/drive/MyDrive'
except Exception as e:
    print('no Drive, checkpointing to /content only:', e)

CKPT = os.path.join(CKPT_DIR, 'diagram_ckpt.json')
print('checkpoint ->', CKPT)
""")

md("""
## 2. Data in

`diagram_batch.zip` is about 145 MB, which is more than the upload
widget likes. Put it in the top level of your Drive first and this cell
finds it; otherwise drop it in the file browser and re-run.
""")

code("""
import zipfile, json

CANDIDATES = ['/content/diagram_batch.zip',
              '/content/drive/MyDrive/diagram_batch.zip']
ZIP = next((p for p in CANDIDATES if os.path.exists(p)), None)

if ZIP is None:
    from google.colab import files
    up = files.upload()
    ZIP = '/content/' + list(up)[0]

if not os.path.isdir('/content/batch'):
    with zipfile.ZipFile(ZIP) as z:
        z.extractall('/content/batch')

PAGES = '/content/batch/pages'
truth = json.load(open('/content/batch/batch_truth.json'))
ids = sorted(f[:-4] for f in os.listdir(PAGES) if f.endswith('.png'))
print(len(ids), 'pages from', ZIP)
print(len(truth), 'of them hand-labelled '
      f"({sum(1 for v in truth.values() if not v['prose_only'])} drawn, "
      f"{sum(1 for v in truth.values() if v['prose_only'])} prose)")
""")

md("""
## 3. Load the model

Same model and same 4-bit quantisation as the reading pass, so any
difference in the result comes from the prompt and not from the setup.
""")

code(f"""
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
import torch

MODEL = {MODEL!r}

bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, quantization_config=bnb, device_map='auto',
    torch_dtype=torch.float16)
proc = AutoProcessor.from_pretrained(MODEL)
model.eval()
print('loaded')
""")

md("""
## 4. The prompt

This is the whole system. Edit it here and re-run the cells below to try
a variant - but delete the checkpoint first, or the old answers will be
reused.
""")

code(f"""
PROMPT = {PROMPT!r}

MAX_NEW = {MAX_NEW}
print(PROMPT)
print('---')
print(len(PROMPT), 'chars')
""")

md("""
## 5. Run it over every page

A few seconds a page, so roughly an hour for the full corpus. Progress
is written to the checkpoint every 25 pages; if the runtime drops, just
re-run this cell.
""")

code(f"""
from qwen_vl_utils import process_vision_info
import json, re, time

CKPT_EVERY = {CKPT_EVERY}


def ask(path):
    msg = [{{'role': 'user', 'content': [
        {{'type': 'image', 'image': path}},
        {{'type': 'text', 'text': PROMPT}}]}}]
    text = proc.apply_chat_template(msg, tokenize=False,
                                    add_generation_prompt=True)
    imgs, vids = process_vision_info(msg)
    batch = proc(text=[text], images=imgs, videos=vids,
                 padding=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=MAX_NEW,
                             do_sample=False, repetition_penalty=1.05)
    trimmed = out[0][len(batch.input_ids[0]):]
    return proc.decode(trimmed, skip_special_tokens=True).strip()


def parse(raw):
    \"\"\"Pull the JSON array out, tolerating a stray fence or sentence.\"\"\"
    m = re.search(r'\\[.*\\]', raw, re.S)
    if not m:
        return None
    try:
        val = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(val, list):
        return None
    out = []
    for d in val:
        if not isinstance(d, dict):
            continue
        try:
            a, b = float(d.get('from', 0)), float(d.get('to', 0))
        except (TypeError, ValueError):
            continue
        if b < a:
            a, b = b, a
        # Closed vocabulary, enforced here as well as in the prompt.
        # Run 1 invented six kinds by echoing the prompt's own examples;
        # a wrong-but-valid value costs less than a junk one.
        kind = str(d.get('kind', '')).lower().strip()
        kind = 'table' if 'table' in kind or 'grid' in kind else 'diagram'
        out.append({{'kind': kind,
                    'from': max(0.0, min(1.0, a)),
                    'to': max(0.0, min(1.0, b)),
                    'caption': str(d.get('caption', '')).strip(),
                    'anchor': str(d.get('anchor', '')).strip()}})
    return out


raw_out, found = {{}}, {{}}
if os.path.exists(CKPT):
    saved = json.load(open(CKPT))
    raw_out, found = saved.get('raw', {{}}), saved.get('found', {{}})
    print(f'resuming, {{len(found)}} pages already done')


def save():
    tmp = CKPT + '.tmp'
    json.dump({{'raw': raw_out, 'found': found}}, open(tmp, 'w'))
    os.replace(tmp, CKPT)


todo = [p for p in ids if p not in found]
t0 = time.time()
for i, pid in enumerate(todo, 1):
    r = ask(f'{{PAGES}}/{{pid}}.png')
    raw_out[pid] = r
    found[pid] = parse(r)
    if i % CKPT_EVERY == 0 or i == len(todo):
        save()
        ok = sum(1 for v in found.values() if v is not None)
        print(f'{{i}}/{{len(todo)}}  done {{len(found)}}  parsed {{ok}}  '
              f'{{(time.time() - t0) / i:.1f}}s/page')

bad = [p for p, v in found.items() if v is None]
print(f'\\nunparseable: {{len(bad)}}', bad[:8])
""")

md("""
## 6. What came back

The number to beat is the reading pass's 5%. The trial run reached 24%.
""")

code("""
n_any = sum(1 for v in found.values() if v)
n_reg = sum(len(v) for v in found.values() if v)
kinds, anchored = {}, 0
for v in found.values():
    for d in (v or []):
        kinds[d['kind']] = kinds.get(d['kind'], 0) + 1
        anchored += bool(d['anchor'])

print(f'pages with at least one drawing : {n_any}/{len(ids)} '
      f'({n_any / max(1, len(ids)):.0%})')
print(f'regions reported                : {n_reg}')
print(f'kinds                           : {kinds}')
print(f'regions carrying an anchor      : {anchored}/{n_reg}')
print()
print('compare: the reading pass marked 52/1000 pages = 5%')
print()
for pid in ids[:200]:
    for d in (found.get(pid) or []):
        print(f"  {pid}  {d['kind']:<8} {d['from']:.2f}-{d['to']:.2f}  "
              f"{d['caption'][:40]:<42}| {d['anchor'][:34]}")
""")

md("""
## 7. Score against the hand labels

A drawing counts as found when a reported span covers at least half of
the hand-marked one. The prose pages should come back empty.

The numbers to beat on these same pages: the geometry rule and the YOLO
detector each find **9/14**, and **10/14** between them.
""")

code("""
hit = miss = spurious = clean = 0
rows = []

for pid, t in sorted(truth.items()):
    got = found.get(pid) or []
    spans = [(d['from'], d['to']) for d in got]

    if t['prose_only']:
        spurious += len(spans)
        clean += (len(spans) == 0)
        rows.append((pid, 'prose', '-', 'clean' if not spans
                     else f'{len(spans)} FALSE: ' +
                          '; '.join(d['caption'][:28] for d in got)))
        continue

    for a, b in t['drawn']:
        cov = 0.0
        for p, q in spans:
            cov = max(cov, max(0.0, min(b, q) - max(a, p)) / (b - a))
        h = cov >= 0.5
        hit += h
        miss += not h
        rows.append((pid, 'drawn', f'{a:.2f}-{b:.2f}',
                     f"{'HIT ' if h else 'MISS'} cover {cov:.2f}"))

print(f"{'page':<14}{'kind':<7}{'range':<12}result")
for r in rows:
    print(f'{r[0]:<14}{r[1]:<7}{r[2]:<12}{r[3]}')

n_prose = sum(1 for v in truth.values() if v['prose_only'])
print(f'\\ndrawings found   {hit}/{hit + miss} '
      f'({hit / max(1, hit + miss):.0%})')
print(f'prose left alone {clean}/{n_prose}')
print(f'spurious spans   {spurious}')
""")

md("""
## 8. See it

Reported spans in orange, hand-marked truth as a green bar on the left.
""")

code("""
import cv2, matplotlib.pyplot as plt

show = sorted(truth)[:16]

fig, axes = plt.subplots(4, 4, figsize=(18, 22))
for ax, pid in zip(axes.ravel(), show):
    im = cv2.imread(f'{PAGES}/{pid}.png')
    h, w = im.shape[:2]
    for d in (found.get(pid) or []):
        y1, y2 = int(d['from'] * h), int(d['to'] * h)
        cv2.rectangle(im, (34, y1), (w - 8, y2), (0, 120, 255), 4)
        cv2.putText(im, d['kind'][:8], (40, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
    for a, b in (truth.get(pid, {}).get('drawn') or []):
        cv2.rectangle(im, (6, int(a * h)), (26, int(b * h)), (0, 170, 0), -1)
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    lab = '  (prose)' if truth.get(pid, {}).get('prose_only') else ''
    ax.set_title(pid + lab, fontsize=9)
    ax.axis('off')
for ax in axes.ravel()[len(show):]:
    ax.axis('off')
plt.tight_layout(); plt.show()
""")

md("""
## 9. Package

Downloads `diagram_result.zip`. Unzip it into `data/figures/` in the
repo, so `diagrams.json` lands at `data/figures/diagrams.json`, then
locally:

    python pipeline.py assemble --engine all_read

`build_booklet.py` picks `data/figures/diagrams.json` up on its own and
uses it as the figure source.
""")

code("""
import zipfile

os.makedirs('/content/out', exist_ok=True)
json.dump(found, open('/content/out/diagrams.json', 'w'), indent=2)
json.dump(raw_out, open('/content/out/raw_replies.json', 'w'), indent=2)
json.dump({'prompt': PROMPT, 'model': MODEL, 'max_new': int(MAX_NEW),
           'pages': int(len(ids)),
           'pages_with_drawing': int(n_any), 'regions': int(n_reg),
           'anchored': int(anchored),
           'bench_hit': int(hit), 'bench_miss': int(miss),
           'bench_spurious': int(spurious), 'bench_prose_clean': int(clean)},
          open('/content/out/summary.json', 'w'), indent=2)

with zipfile.ZipFile('/content/diagram_result.zip', 'w',
                     zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir('/content/out'):
        z.write('/content/out/' + f, f)

from google.colab import files
files.download('/content/diagram_result.zip')
""")

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

for i, c in enumerate(nb["cells"]):
    for j, line in enumerate(c["source"][:-1]):
        if not line.endswith("\n"):
            raise SystemExit(f"cell {i} line {j} lost its newline")

# Every code cell must actually compile - a generated notebook that
# fails on cell 7 wastes a GPU session.
import ast
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.lstrip().startswith("!"):
        continue
    try:
        ast.parse(src)
    except SyntaxError as exc:
        raise SystemExit(f"cell {i} does not parse: {exc}")

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)

print(f"wrote {OUT}")
print(f"{len(CELLS)} cells, prompt {len(PROMPT)} chars")
