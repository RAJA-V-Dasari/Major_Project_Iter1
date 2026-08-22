"""
Generate the Colab notebook that reads pages with a VLM.

Written as a generator rather than a checked-in .ipynb because the
prompt is the important part of that notebook and it is easier to
review, diff and re-tune here than inside notebook JSON.

    python make_colab_notebook.py
        -> 04_ocr/read_pages_colab.ipynb
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "read_pages_colab.ipynb"

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
PROMPT = """You are transcribing a handwritten exam answer that a
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
Transcribe a table only if you can read EVERY cell. If any cell is
unclear, do not reconstruct the table and do not infer values from the
pattern of the others: emit exactly

![table]

and nothing else for it. A table of invented numbers is the worst
output you can produce here.

DIAGRAMS
Diagrams, graphs, figures, flowcharts, timing charts: never describe
them in prose and never transcribe the labels inside them as text.
Emit exactly

![diagram]

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


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.strip().split("\n")}


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.strip().split("\n")}


CELLS = [
    md("""
# Reading answer scripts with a VLM

Produces one Markdown file per input image, named identically, ready to
score with `modules/06_evaluation/src/ocr_bench.py`.

**Before uploading:** cover pages (`page_01`) carry names, USNs and
marks. They are already excluded by `07_reconstruct`, but check your zip.

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
## 3. Upload your pages

Zip the images first. Filenames become the output names, so use the
page ids the benchmark expects, e.g. `s06_c1_p05.png`.
"""),
    code("""
import zipfile, pathlib
from google.colab import files

up = files.upload()                 # choose your .zip
name = next(iter(up))

IN = pathlib.Path('/content/pages'); IN.mkdir(exist_ok=True)
with zipfile.ZipFile(name) as z:
    z.extractall(IN)

imgs = sorted(p for p in IN.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'})
print(f'{len(imgs)} image(s)')
for p in imgs[:10]: print('  ', p.name)
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

    md("## 5. The prompt"),
    code('PROMPT = """' + PROMPT + '"""\n\nprint(PROMPT)'),

    md("""
## 6. Read every page

`process_vision_info` is what actually resizes the image to the
processor's pixel budget. Passing a raw PIL image to
`processor(images=...)` skips that step, which is how a page becomes
~4,400 visual tokens and asks for 18.85 GiB.

A page that still OOMs is retried once at half the patch budget rather
than being lost, and the message says so - a page silently written as
an empty file would score as a total miss and look like a reading
failure rather than a memory one.
"""),
    code("""
import time, pathlib, gc
from PIL import Image
from qwen_vl_utils import process_vision_info

OUT = pathlib.Path('/content') / ENGINE; OUT.mkdir(exist_ok=True)

def read(path, max_patches=MAX_PATCHES):
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
        out = model.generate(**inputs, max_new_tokens=1536, do_sample=False)

    trimmed = out[0][tokens:]
    body = processor.decode(trimmed, skip_special_tokens=True).strip()

    del inputs, out
    return body, tokens

t0 = time.time()
failed = []

for i, p in enumerate(imgs, 1):
    started = time.time()
    body, tokens = '', 0

    for attempt, budget in enumerate([MAX_PATCHES, MAX_PATCHES // 2]):
        try:
            body, tokens = read(p, budget)
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

    (OUT / (p.stem + '.md')).write_text(body, encoding='utf-8')
    print(f'[{i}/{len(imgs)}] {p.stem}  {len(body)} chars  '
          f'{tokens} in-tokens  {time.time()-started:.0f}s', flush=True)

    torch.cuda.empty_cache(); gc.collect()

print(f'\\nTotal {time.time()-t0:.0f}s')
if failed:
    print(f'FAILED ({len(failed)}): {failed}')
else:
    print('all pages read')
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
for f in sorted(OUT.glob('*.md')):
    body = f.read_text(encoding='utf-8')
    marks['[?]'] += len(re.findall(r'\\[\\?\\]', body))
    marks['![table]'] += len(re.findall(r'!\\[table\\]', body))
    marks['![diagram]'] += len(re.findall(r'!\\[diagram\\]', body))
    marks['md table rows'] += len([l for l in body.splitlines()
                                   if l.strip().startswith('|')])

print(f'{"signal":<16}{"count":>7}')
for k in ['[?]', '![table]', '![diagram]', 'md table rows']:
    print(f'{k:<16}{marks[k]:>7}')

print()
if marks['[?]'] == 0:
    print('WARNING: still never declines. The anti-fabrication clause '
          'is not working.')
else:
    print('Good: the model is declining where it cannot read.')

if marks['md table rows'] and not marks['![table]']:
    print('NOTE: it transcribed tables and never once refused one. '
          'Check those cells against the page by eye.')
"""),

    md("## 8. Spot-check the page that failed last time"),
    code("""
target = 's10_c2_p10'
hit = [f for f in OUT.glob('*.md') if f.stem == target]
print((hit[0] if hit else sorted(OUT.glob('*.md'))[0]).read_text(encoding='utf-8')[:2000])
"""),

    md("""
## 9. Download

Unzip into `modules/06_evaluation/predictions/<ENGINE>/`, then locally:

```
python modules/06_evaluation/src/ocr_bench.py --engine qwen7b --verbose
python modules/06_evaluation/src/ocr_bench.py --engine qwen3b --verbose   # the old run
```

Same pages, same scorer, so the three engines line up directly against
the 0.573 baseline.
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

    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"{len(CELLS)} cells")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
