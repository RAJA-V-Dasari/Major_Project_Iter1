"""
Generate the Colab notebook that reads pages with a VLM.

Written as a generator rather than a checked-in .ipynb because the
prompt is the important part of that notebook and it is easier to
review, diff and re-tune here than inside notebook JSON.

    python reading/read/make_colab_notebook.py
        -> reading/read/read_pages_colab.ipynb
    python reading/read/make_colab_notebook.py --check   # compile, compare

The prompt lives here and only here. read_pages.py and kaggle_run.py
import it, so every route to a GPU reads with the same words.
"""

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "read_pages_colab.ipynb"

# The prompt is the whole experiment. Two clauses in it are doing the
# real work, and both come from what the incumbent got wrong:
#
#   "transcribe exactly ... do not correct" - these are exam answers
#   being marked. A model that silently fixes a student's spelling or
#   arithmetic has destroyed the thing the grader needs to see.
#
#   "write [?] rather than guessing" - TrOCR's measured failure was not
#   garbled output, it was FABRICATION: "the destination port number is
#   the next" came back as "Although the situation had not yet yet
#   been". Fluent invention is the one failure a marker cannot catch,
#   so the model is given an explicit way to say it cannot read
#   something, and [?] is greppable afterwards.
PROMPT = """You are transcribing one page of a handwritten exam answer
so that a human can mark it. Report what is on the paper, and nothing
else.

Do not answer the exam. Do not use what you know about the subject to
fill in, complete or correct anything. If you recognise the worked
example, that recognition is a trap: copy the marks that are there,
even where they contradict what the answer should be. A wrong value
copied faithfully is correct output. A right value you supplied is a
serious error, because the marker cannot tell that you invented it.

Most pages are ordinary handwriting. Transcribe them as plain lines,
keeping the line breaks, spelling, grammar and arithmetic exactly as
written, mistakes included.

Where you cannot read a word or a number, write [?] and carry on. Use
it freely. A page with [?] in it is useful, because the marker knows
which parts to check. A fluent page that is partly invented is not,
because nothing in it says which parts to distrust.

Drawings
A drawing is anything the student drew rather than wrote: a figure, a
graph of nodes and edges, a flowchart, a timing chart, a network or
circuit sketch, a tree. Do not describe it. Do not transcribe the
labels inside it. Do not lay it out as rows and columns. Replace the
whole drawing with a single line:

![diagram: a few words saying which drawing this is]

Put that line where the drawing sits in the answer, with the writing
that comes before it above and the writing that follows it below.
Judge by what the marks are, not by what they contain: a graph whose
nodes hold numbers is still a drawing.

Tables
A table is a grid the student ruled on the page, with writing sitting
inside its cells. Read it as a Markdown table, one row per ruled row.

Read each cell on its own, and put [?] in any cell you cannot make
out. Never work a cell out from its neighbours: a column that looks
like it continues a sequence is exactly where a wrong value gets
invented.

Every cell you write must hold something you actually read on the
page. If that leaves you with a table you cannot fill, write this
instead and move on:

![table: a few words saying which table this is]

If the student ruled no grid, there is no table. Ordinary lines, the
steps of a calculation, a list of addresses: those are lines, and they
stay lines.

Question numbers
The number in the left margin is one of 1, 2a, 2b, 2c, 3a, 3b, 4a, 4b.
Write it as a heading, like: ### 2a)
A part within it (i, ii, iii, or a, b, c) becomes: #### i)

Nothing else is ever a heading. A numbered point inside an answer is a
list item, written 1. or *, because a heading there would split one
answer into several.

Also
Mathematics goes inline, between $ and $.
Struck-out writing is wrapped in ~~ ~~.

Give the Markdown for the page and nothing else: no preamble, no
commentary, and no sentence taken from these instructions. Do not put
the page inside a code fence. Use a fence only where the student
themselves laid writing out as a block."""


# The prompt batch00 was read with. It is kept verbatim, and
# selectable in the notebook, because it is the only prompt whose
# behaviour over 250 pages is actually known: 219 of 250 pages
# structurally clean, 0.099 CER on the 15 benchmark pages. Its
# faults are known too - it never declines and its box coordinates
# are guesses - but they are measured and detectable, which an
# untested rewrite's faults are not. If a run goes wrong mid
# session, switching back is one line rather than a git revert.
PROMPT_BATCH00 = """You are transcribing a handwritten exam answer that a
human will mark. Your only job is to report what is on the paper.

THE ONE RULE THAT MATTERS
You are not answering this exam and you are not helping the student.
Do not use what you know about the subject to fill in, complete or
correct anything. If the page shows a worked example you recognise,
that recognition is a trap: transcribe the marks that are there, even
where they contradict what the answer should be. A wrong value copied
faithfully is correct output. A right value you supplied is a serious
error, because the marker cannot tell you invented it.

WHEN YOU CANNOT READ SOMETHING
Write [?] in place of the word or number. Do this readily - an answer
peppered with [?] is far more useful than a fluent one that is partly
invented. Never substitute a plausible word for an illegible one.

TABLES
Transcribe tables as Markdown tables. Most are comparisons - two
columns of words - and you should read them normally.

The care is needed per CELL, not per table. Any single cell you cannot
read with certainty becomes [?]. Never infer a cell's value from the
pattern of the cells around it: a column of numbers that looks like it
continues a sequence is exactly where a wrong value gets invented.

Only if MOST of the cells would be [?] - a dense numeric working you
cannot resolve - skip the table entirely and emit

![table](x1,y1,x2,y2)

using the box rule below.

DIAGRAMS
Diagrams, graphs, figures, flowcharts, timing charts, circuit
sketches: never describe them in prose and never transcribe the labels
inside them as text. Emit a placeholder with the box that encloses the
whole figure, including its labels:

![diagram](x1,y1,x2,y2)

Coordinates are pixels in the image as you see it: x1,y1 is the
top-left corner and x2,y2 the bottom-right. Give one box per distinct
figure. Make the box tight around the drawing but do not clip any part
of it, and do not merge two separate figures into one box.

STRUCTURE
- Question numbers appear in the left margin (1, 2a, 2b, 2c, 3a, 3b,
  4a, 4b). Emit each as: ### 2a)
- Sub-parts (i, ii, iii ... or a, b, c ...) as: #### i)
- Mathematics: inline LaTeX between $ ... $
- Struck-out or cancelled text: wrap in ~~ ~~
- Keep the line breaks as written.
- Preserve the student's spelling, grammar and arithmetic exactly,
  errors included.

Output only the Markdown. No commentary, no preamble."""


def lines_of(source):
    """Split into nbformat's `source` list.

    Every element except the last MUST keep its trailing newline. A
    notebook reader concatenates the list with "".join, so a list of
    bare lines is reassembled into one enormous line and every cell
    opens as a single unreadable row. Most local tooling is forgiving
    about this; Colab is not.

    splitlines(keepends=True) is exactly the rule: it keeps the newline
    on every line that had one and leaves the last line bare.
    """
    return source.strip().splitlines(keepends=True)


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines_of(source)}


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": lines_of(source)}


CELLS = [
    md("""
# Reading answer scripts with a VLM

Produces one Markdown file per input image, named identically, ready to
score with `reading/benchmark/ocr_bench.py` and to assemble into booklets.

**Before uploading:** cover pages (`page_01`) carry names, USNs and
marks. They are excluded by prepare_corpus_batches.py, and checked again below.

**Runtime → Change runtime type → T4 GPU** before running anything.
"""),

    md("## 1. Confirm the GPU"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),

    md("## 2. Install"),
    code("""
!pip -q install "transformers>=4.49" accelerate qwen-vl-utils bitsandbytes
"""),

    md("""
## 3. Get the pages in

`prepare_corpus_batches.py` writes `batch_00.zip` … `batch_04.zip`
(~250 pages, ~65MB each) plus a MANIFEST. Filenames are page ids and
become the output names.

### Somebody else's GPU, the owner's Drive

Colab GPU quota is per account, so the reading can be done from another
account while the corpus stays in the owner's Drive.

**Owner, once.** Right-click the folder holding the batch zips →
*Share* → *General access* → **Anyone with the link** → *Viewer*. Copy
the link into `LINK` below. Nothing else to set up, and nothing to
install on the runner's side. When the reading is finished, set it back
to *Restricted* — the link is what grants access, so revoking it is the
whole of the cleanup.

**Runner.** Paste the link, pick a `BATCH`, run. Pages are fetched with
`gdown`; no Drive account or permission of your own is needed for them.

**Results go to the runner's own Drive**, not to the runtime. A free
session drops after a few hours and the corpus takes 8-15, so results
that live only on the runtime are lost with it. In a mounted Drive
folder they survive, and re-running simply resumes where it stopped.
Zip that folder and send it back when the corpus is done, or use
section 9.

`SOURCE = 'upload'` is fine for one batch or a trial.

**Start with the test batch.** `prepare_test_batch.py` writes
`batch_test.zip` - 30 pages, 7.3MB, ~25 minutes on a T4. Every page in
it has known recorded behaviour: 21 that broke on the last run and 9
controls that came back correct and must still come back correct. Set
`SOURCE = 'upload'`, or put it in the Drive folder and set
`BATCH = 'batch_test.zip'`. Check sections 6b, 7 and 8 against
`data/read/batches/TEST_BATCH.csv` before spending hours on the full corpus.
"""),
    code("""
import zipfile, pathlib, subprocess, sys

# Where the pages come from. The corpus can stay in its owner's Drive
# while somebody else's GPU quota does the reading.
#
#   'link'    a Drive link set to "anyone with the link", fetched with
#             gdown. The runner needs no account and no permissions.
#   'drive'   a folder already mounted in the runner's own Drive.
#   'upload'  files.upload(), fine for the 7MB test batch.
SOURCE = 'link'

LINK = ''                   # the owner's Drive folder or file link
DRIVE_DIR = '/content/drive/MyDrive/answer_scripts'
BATCH = 'batch_00.zip'      # None extracts every batch present

# Results go to the RUNNER's own Drive. A free session drops after a
# few hours and the corpus takes 8-15, so anything written only to the
# runtime is lost with it; in Drive the run resumes instead of starting
# again. Set False only for a short test.
SAVE_TO_DRIVE = True
RESULTS_DIR = '/content/drive/MyDrive/answer_scripts_out'

IN = pathlib.Path('/content/pages'); IN.mkdir(exist_ok=True)
OUT_ROOT = pathlib.Path('/content')


# Extract the wanted batch, or the only one there is. A comment and not
# a docstring: cell source is held in a triple-quoted string here, so a
# docstring inside a cell closes the cell early.
def stage(zips):
    if not zips:
        raise SystemExit('no batch zip found - check the path or link')

    if BATCH is None or len(zips) == 1:
        picked = zips
    else:
        picked = [z for z in zips if z.name == BATCH]
        if not picked:
            raise SystemExit(f'{BATCH} not among {[z.name for z in zips]}')

    for z in picked:
        with zipfile.ZipFile(z) as archive:
            archive.extractall(IN)
        print('extracted', z.name)


if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT_ROOT = pathlib.Path(RESULTS_DIR)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f'results -> {OUT_ROOT} (survives a dropped session)')
else:
    print('results -> this runtime only; they are lost if it drops')

if SOURCE == 'link':
    if not LINK:
        raise SystemExit('set LINK to the shared Drive link')

    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'],
                   check=True)
    import gdown

    got = pathlib.Path('/content/zips'); got.mkdir(exist_ok=True)
    if '/folders/' in LINK:
        gdown.download_folder(LINK, output=str(got), quiet=True,
                              use_cookies=False)
    else:
        gdown.download(LINK, output=f'{got}/', quiet=True, fuzzy=True)

    found = sorted(got.rglob('*.zip'))
    if not found:
        raise SystemExit(
            'nothing downloaded. Check the folder is shared as "anyone '
            'with the link" - gdown cannot open a restricted one.')
    print('fetched:', [z.name for z in found])
    stage(found)

elif SOURCE == 'drive':
    if not SAVE_TO_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
    src = pathlib.Path(DRIVE_DIR)
    if not src.exists():
        raise SystemExit(f'{src} not found')
    stage(sorted(src.glob('batch_*.zip')))

else:
    from google.colab import files
    up = files.upload()
    for name in up:
        with zipfile.ZipFile(name) as archive:
            archive.extractall(IN)
        print('extracted', name)

imgs = sorted(p for p in IN.rglob('*')
              if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
print(f'\\n{len(imgs)} image(s) staged')

# Covers should never have left the local machine. Checked again here
# because the cost of being wrong is student identity data sitting on
# a hosted GPU, and that is worth two lines of paranoia.
covers = [p for p in imgs if p.stem.endswith('_p01')]
if covers:
    raise SystemExit(f'COVER PAGES PRESENT ({len(covers)}) - remove them '
                     f'first: {[p.name for p in covers[:5]]}')
print('no cover pages present')
"""),

    md("""
## 4. Load the model

`3B` is comfortable on a T4 and fast. `7B` is more accurate but needs
4-bit to fit — try 3B first and only move up if the score demands it.

**The pixel budget is set on the PROCESSOR here, and that is the part
that matters.** Qwen's default cap is 16384 image patches. A
1598x2177 scan is 3.48M pixels, which at 28x28 patches is ~4,400
visual tokens, and attention over 4,400 tokens is what asks a T4 for
18.85 GiB and dies. Capping at 1024 patches resizes the page to ~800k
pixels first, which is ~19x less attention memory and still well above
what this handwriting needs to stay legible.

Setting it on the processor makes it apply no matter how the image is
passed in later. Putting the same numbers only inside the chat message
does NOT: the message is a template, and a raw PIL image handed
straight to `processor(images=...)` sails past it.
"""),
    code("""
# --- pick one, run the whole notebook, then come back and pick the
# --- other. ENGINE names the output folder, so the two runs land in
# --- separate directories and ocr_bench can score them side by side.

RUN = "7b"          # "3b" or "7b"

if RUN == "3b":
    MODEL, FOUR_BIT, ENGINE = "Qwen/Qwen2.5-VL-3B-Instruct", False, "qwen3b_v2"
else:
    MODEL, FOUR_BIT, ENGINE = "Qwen/Qwen2.5-VL-7B-Instruct", True, "qwen7b"

# in 28x28 patches. Lower MAX_PATCHES if you hit OOM.
MIN_PATCHES, MAX_PATCHES = 256, 1024

# Greedy decoding sometimes falls into a repetition loop and then runs
# to this ceiling, so the ceiling sets what a loop costs: at ~7.5 tok/s
# a page that loops burns MAX_NEW/7.5 seconds and is then retried, and
# on the test batch two pages cost 440s each that way.
#
# The largest page here that was read correctly is ~1400 characters,
# roughly 400 tokens, so 1024 leaves about 2.5x headroom over anything
# genuine while cutting a loop from 203s to ~135s. A page that truly
# needs more than this does not exist in the corpus; a page that asks
# for it is looping.
#
# The penalty applies ONLY on the retry - a table with legitimately
# repeated cell values must not be penalised on the first, clean pass.
MAX_NEW, REP_PENALTY = 1024, 1.15

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
if FOUR_BIT:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, **kw)
model.eval()

processor = AutoProcessor.from_pretrained(
    MODEL,
    min_pixels=MIN_PATCHES * 28 * 28,
    max_pixels=MAX_PATCHES * 28 * 28,
)

print('engine :', ENGINE)
print('model  :', MODEL, '(4-bit)' if FOUR_BIT else '(bf16)')
print('max visual tokens per image:', MAX_PATCHES)
"""),

    md("""
## 5. The prompt

`USE` picks between two, and the choice is a real one.

`batch00` is the prompt whose behaviour over 250 pages is actually
known: 219 of them structurally clean, 0.099 CER on the benchmark
pages. Its faults are known too - it never declines, and its figure
coordinates are guesses - but they are measured, and `check_batch.py`
finds them.

`v2` targets those faults. It has not been run over a corpus.

If a run starts producing something obviously wrong, change this one
line rather than stopping to work out what happened.
"""),
    code('USE = "v2"        # "v2" or "batch00" - see the markdown above\n\n'
         'PROMPT_V2 = """' + PROMPT + '"""\n\n'
         'PROMPT_BATCH00 = """' + PROMPT_BATCH00 + '"""\n\n'
         'PROMPT = PROMPT_V2 if USE == "v2" else PROMPT_BATCH00\n'
         'print(f"using {USE}, {len(PROMPT)} chars")'),

    md("""
## 6. Read every page

`process_vision_info` is what actually resizes the image to the
processor's pixel budget. Passing a raw PIL image to
`processor(images=...)` skips that step, which is how a page becomes
~4,400 visual tokens and asks for 18.85 GiB.

**Markers carry no coordinates, deliberately.** Asking for pixels
produced round numbers in the model's own resized space - `105, 210,
630, 840` is `50, 100, 300, 400` at 768x1046 - and of six boxes drawn
back onto their pages only one enclosed its figure; one boxed prose and
missed the diagram entirely. What the reader does get right is the
reading ORDER: the marker lands where the figure belongs in the answer.
So position in the output locates the figure, and
`reading/assemble/segment.py` supplies the pixels.

A page that still OOMs is retried once at half the patch budget rather
than being lost, and the message says so - a page silently written as
an empty file would score as a total miss and look like a reading
failure rather than a memory one.
"""),
    code("""
import time, pathlib, gc, re, json
from PIL import Image
from qwen_vl_utils import process_vision_info

OUT = OUT_ROOT / ENGINE; OUT.mkdir(parents=True, exist_ok=True)

# A marker carries a description, never coordinates. Where a figure
# sits on the page is measured by reading/assemble/segment.py,
# which is precise about geometry; asking the model for pixels produced
# round numbers in its own resized space that mostly missed the figure.
MARK = re.compile(r'!\\[(diagram|table):\\s*([^\\]]*)\\]')

def read(path, max_patches=MAX_PATCHES, penalty=1.0):
    image = Image.open(path).convert('RGB')

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image,
         "min_pixels": MIN_PATCHES * 28 * 28,
         "max_pixels": max_patches * 28 * 28},
        {"type": "text", "text": PROMPT}]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    # THIS is the resize step. Without it the pixel budget is ignored.
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    tokens = int(inputs.input_ids.shape[1])

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW,
                             do_sample=False, repetition_penalty=penalty)

    trimmed = out[0][tokens:]

    # Hitting the ceiling means the text is cut off mid-sentence. It is
    # NOT empty, so the resume rule below ("non-empty means done") would
    # otherwise keep it forever. A ceiling hit is a failure to report,
    # not a page that merely took a long time.
    truncated = int(trimmed.shape[0]) >= MAX_NEW

    body = processor.decode(trimmed, skip_special_tokens=True).strip()

    del inputs, out
    return body, tokens, truncated

t0 = time.time()
failed, truncated = [], []

# RESUME. A free Colab session is capped around 12 hours and drops on
# idle, and a full-corpus pass does not fit in one sitting. Anything
# already written is skipped, so re-running after a disconnect picks up
# where it stopped instead of starting again. An empty file counts as
# NOT done - that is how a failed page is retried rather than kept.
done = {f.stem for f in OUT.glob('*.md') if f.stat().st_size > 0}
todo = [p for p in imgs if p.stem not in done]

if done:
    print(f'resuming: {len(done)} already read, {len(todo)} to go\\n')

for i, p in enumerate(todo, 1):
    started = time.time()
    body, tokens, cut = '', 0, False

    for attempt, budget in enumerate([MAX_PATCHES, MAX_PATCHES // 2]):
        try:
            body, tokens, cut = read(p, budget)
            if attempt:
                print(f'    (recovered at {budget} patches)')
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            if attempt:
                failed.append(p.stem)
                print(f'  {p.stem}: OOM even at {budget} patches')
        except Exception as e:
            failed.append(p.stem)
            print(f'  {p.stem}: FAILED {type(e).__name__}: {e}')
            break

    # One retry with a mild repetition penalty: enough to break a loop,
    # low enough that a page which is genuinely dense still comes out.
    if cut:
        torch.cuda.empty_cache(); gc.collect()
        again, tokens, cut = read(p, MAX_PATCHES, penalty=REP_PENALTY)
        if cut:
            truncated.append(p.stem)
        else:
            print('    (broke a repetition loop on retry)')
        body = again

    (OUT / (p.stem + '.md')).write_text(body, encoding='utf-8')
    (OUT / '_truncated.json').write_text(json.dumps(truncated, indent=1),
                                         encoding='utf-8')

    flag = '  *** TRUNCATED' if cut else ''
    print(f'[{i}/{len(todo)}] {p.stem}  {len(body)} chars  '
          f'{tokens} in-tokens  {time.time()-started:.0f}s{flag}',
          flush=True)

    torch.cuda.empty_cache(); gc.collect()

print(f'\\nTotal {time.time()-t0:.0f}s')
if failed:
    print(f'FAILED ({len(failed)}): {failed}')
if truncated:
    print(f'TRUNCATED ({len(truncated)}) - cut off at the {MAX_NEW}-token '
          f'ceiling even after a retry. They are written but incomplete, '
          f'and listed in _truncated.json so assembly can quarantine '
          f'them: {truncated}')
if not failed and not truncated:
    print('all pages read')
"""),

    md("""
## 6b. Redo pages that hit the token ceiling

Run this on a corpus read by an EARLIER version of this notebook, which
wrote a cut-off page as if it were finished. Greedy decoding can fall
into a repetition loop, run to the ceiling, and stop mid-sentence; the
file is non-empty, so the resume rule counts it as done and never
revisits it.

A page cannot be judged cut off by its ending, because answers legitimately
run onto the next page and so a clean page often ends mid-sentence. Length
is the honest signal: a normal page here is 200-1400 characters, and a
ceiling hit is ~2400 upwards. Anything long is re-read with the repetition
penalty, and the retry is kept only if it comes back shorter - a page that
is simply dense will read the same both ways and is left alone.
"""),
    code("""
SUSPECT_CHARS = 1800

suspects = [f for f in sorted(OUT.glob('*.md'))
            if len(f.read_text(encoding='utf-8')) >= SUSPECT_CHARS]

print(f'{len(suspects)} page(s) long enough to suspect a loop')
print()

fixed = []
for f in suspects:
    before = f.read_text(encoding='utf-8')
    img = next((q for q in imgs if q.stem == f.stem), None)
    if img is None:
        print(f'  {f.stem}: image not in this batch, skipped')
        continue

    body, _, cut = read(img, MAX_PATCHES, penalty=REP_PENALTY)
    state = '(still cut)' if cut else '(clean)'

    if len(body) < len(before):
        f.write_text(body, encoding='utf-8')
        fixed.append(f.stem)
        print(f'  {f.stem}: {len(before)} -> {len(body)} chars  {state}')
    else:
        print(f'  {f.stem}: {len(before)} chars, unchanged - genuinely dense')

    torch.cuda.empty_cache(); gc.collect()

print()
print(f'rewrote {len(fixed)}: {fixed}')
"""),

    md("""
## 7. Did the prompt actually take?

The previous run scored 0.141 CER overall but fabricated a whole
Dijkstra table on the one page that had one, and used `[?]` exactly
zero times across every page - including that one. So the useful check
is not the score, it is whether the model is now willing to decline.

`[?]` and `![table]` counts of zero mean the new instructions did
nothing, whatever the CER says.
"""),
    code("""
import re, collections

marks = collections.Counter()
unboxed = 0

for f in sorted(OUT.glob('*.md')):
    body = f.read_text(encoding='utf-8')
    marks['[?]'] += len(re.findall(r'\\[\\?\\]', body))
    marks['![diagram:] markers'] += len([m for m in MARK.findall(body)
                                         if m[0] == 'diagram'])
    marks['![table:] markers'] += len([m for m in MARK.findall(body)
                                       if m[0] == 'table'])
    marks['md table rows'] += len([l for l in body.splitlines()
                                   if l.strip().startswith('|')])
    # Anything image-shaped that is not one of the two markers: a
    # bare ![diagram], a coordinate tuple, an invented URL. All
    # three appeared in the previous run, and each one is a figure
    # the pipeline cannot place.
    unboxed += len(re.findall(r'!\\[[^\\]]*\\]\\(', body))
    unboxed += len(re.findall(r'!\\[(?:diagram|table)\\](?!:)', body))

marks['MALFORMED markers'] = unboxed

print(f'{"signal":<28}{"count":>7}')
for k in ['[?]', '![diagram:] markers', '![table:] markers',
          'MALFORMED markers', 'md table rows']:
    print(f'{k:<28}{marks[k]:>7}')

print()
if marks['[?]'] == 0:
    print('WARNING: still never declines. The anti-fabrication clause '
          'is not working.')
else:
    print('Good: it is declining where it cannot read.')

if unboxed:
    print(f'WARNING: {unboxed} malformed marker(s). The only two valid '
          f'forms are ![diagram: ...] and ![table: ...] - anything with '
          f'parentheses is a coordinate tuple or a URL the model made up.')

if marks['md table rows'] and not marks['[?]']:
    print('NOTE: tables were transcribed with no [?] anywhere. Check '
          'those cells against the page before trusting them - this is '
          'exactly where the 3B invented values last time.')
"""),

    md("""
## 8. Are the markers in the right place?

There are no coordinates to check any more. What has to be right is the
marker's POSITION IN THE READING ORDER - the line above it and the line
below it should be the writing that sits either side of the figure on
the page. That is what lets the geometry stage match a marker to a
region.

This prints each marker with its neighbours, next to the page itself.
Read the two together: if the marker claims to follow "* Eg:" then
"* Eg:" should be the last thing above the drawing on the scan.
"""),
    code("""
shown = 0
for f in sorted(OUT.glob('*.md')):
    body = f.read_text(encoding='utf-8')
    lines = body.splitlines()
    hits = [i for i, l in enumerate(lines) if MARK.search(l)]
    if not hits:
        continue

    src = [q for q in imgs if q.stem == f.stem]
    if not src:
        continue

    print(f'=== {f.stem} : {len(hits)} marker(s) ===')
    for i in hits:
        above = next((lines[j] for j in range(i - 1, -1, -1)
                      if lines[j].strip()), '(top of page)')
        below = next((lines[j] for j in range(i + 1, len(lines))
                      if lines[j].strip()), '(bottom of page)')
        print(f'   above: {above[:64]}')
        print(f'   MARK : {lines[i].strip()[:64]}')
        print(f'   below: {below[:64]}')
        print()

    im = Image.open(src[0]).convert('RGB')
    im.thumbnail((700, 700))
    display(im)

    shown += 1
    if shown >= 4:
        break

if not shown:
    print('No markers at all. Either these pages carry no figures, or '
          'the marker instruction did not take - check a page with a '
          'diagram on it before running the rest of the corpus.')
"""),

    md("""
## 9. Download

Unzip into `data/read/<ENGINE>/` in the repo (one `.md` per page), then
locally:

```
python pipeline.py read --engine qwen7b              # coverage: what is read, what is not
python reading/benchmark/ocr_bench.py --engine qwen7b --verbose
python pipeline.py run --engine qwen7b               # assemble -> handoff -> mark
```

Same pages, same scorer, so engines line up directly against the 0.573
baseline.
"""),
    code("""
import shutil
from google.colab import files
shutil.make_archive(f'/content/{ENGINE}', 'zip', OUT)
files.download(f'/content/{ENGINE}.zip')
"""),

    md("""
## What to look for

- **`[?]` markers** — the model admitting it cannot read. Zero of these
  across a whole booklet is a red flag, not a good sign.
- **`![table]`** on the Dijkstra page. Last run it invented
  `5 | 6 | 7 | 8` where the page says `2,A | 5,A | inf | inf`.
- **`![diagram]`** where a figure is, rather than prose describing it.
- **Question headings** (`### 2a)`) — the assembly step groups on these,
  so they matter more than the prose around them.
- **A LOWER CER is not automatically better.** If the model starts
  emitting `![table]` where it used to invent one, CER may barely move
  while the output becomes far more trustworthy. Read the diff, not
  just the number.
"""),
]


def main():

    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    # Compile every code cell before writing. Cells are stored as a
    # LIST OF LINES, so a "\n" that survives into the generator's own
    # string literal splits one source line into two and breaks the
    # cell - which happened, and only showed up on Colab. Escaping is
    # easy to get wrong again; noticing is not.
    broken = 0

    for index, cell in enumerate(CELLS):

        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])

        if source.lstrip().startswith("!"):
            continue                      # shell magic, not Python

        try:
            compile(source, f"<cell {index}>", "exec")
        except SyntaxError as error:
            broken += 1
            print(f"BROKEN cell {index}, line {error.lineno}: {error.msg}")
            for line in source.split("\n")[max(0, error.lineno - 3):
                                            error.lineno + 1]:
                print(f"    {line}")

    # Structural guard, separate from the compile check above. A cell
    # whose lines have lost their newlines still compiles once joined,
    # so only this catches the file collapsing to one line per cell.
    for index, cell in enumerate(CELLS):
        for line in cell["source"][:-1]:
            if not line.endswith("\n"):
                raise SystemExit(
                    f"cell {index} line {line[:40]!r} has no newline - "
                    f"the notebook would open as one line per cell")

    if broken:
        raise SystemExit(f"\n{broken} cell(s) will not compile - not written.")

    # --check: compile, and confirm the committed notebook is what this
    # generator would write. An edited prompt that was never regenerated
    # is a notebook reading pages with a prompt nobody reviewed.
    if "--check" in sys.argv[1:]:
        current = (json.loads(OUT.read_text(encoding="utf-8"))
                   if OUT.exists() else None)
        if current != json.loads(json.dumps(notebook)):
            raise SystemExit(f"{OUT.name} is out of date - regenerate it: "
                             "python reading/read/make_colab_notebook.py")
        print(f"{len(CELLS)} cells, all code cells compile, "
              f"{OUT.name} is current")
        return 0

    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"{len(CELLS)} cells, all code cells compile")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
