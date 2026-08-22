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
PROMPT = """Transcribe this handwritten exam answer page to Markdown.

Rules:
- Transcribe EXACTLY what is written. Do not correct spelling, grammar,
  arithmetic or factual errors. Student mistakes are the data.
- If you cannot read something, write [?] instead of guessing a
  plausible word. Never invent text.
- Question numbers are written in the left margin (1, 2a, 2b, 2c, 3a,
  3b, 4a, 4b). Emit each as a heading: ### 2a)
- Sub-parts (i, ii, iii ... or a, b, c ...) become: #### i)
- Mathematics: inline LaTeX between $ ... $
- Tables: a Markdown table.
- Diagrams, graphs, figures, flowcharts: do NOT describe them in prose.
  Emit exactly this line and nothing else for them: ![diagram]
- Struck-out or cancelled text: wrap in ~~ ~~
- Keep the line breaks as written.

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
"""),
    code("""
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"   # or "Qwen/Qwen2.5-VL-7B-Instruct"
FOUR_BIT = False                         # set True for the 7B on a T4

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
if FOUR_BIT:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, **kw)
processor = AutoProcessor.from_pretrained(MODEL)
print('loaded', MODEL)
"""),

    md("## 5. The prompt"),
    code('PROMPT = """' + PROMPT + '"""\n\nprint(PROMPT)'),

    md("""
## 6. Read every page

`min_pixels` / `max_pixels` matter: too small and the handwriting is
unreadable, too large and a T4 runs out of memory on a 1598x2177 scan.
"""),
    code("""
import time, pathlib
from PIL import Image

OUT = pathlib.Path('/content/markdown'); OUT.mkdir(exist_ok=True)

MIN_PX, MAX_PX = 512*28*28, 1280*28*28

def read(path):
    image = Image.open(path).convert('RGB')
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image,
         "min_pixels": MIN_PX, "max_pixels": MAX_PX},
        {"type": "text", "text": PROMPT}]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image],
                       return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1536, do_sample=False)

    trimmed = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()

t0 = time.time()
for i, p in enumerate(imgs, 1):
    started = time.time()
    try:
        body = read(p)
    except Exception as e:
        body = ''
        print(f'  {p.name}: FAILED {e}')
    (OUT / (p.stem + '.md')).write_text(body, encoding='utf-8')
    print(f'[{i}/{len(imgs)}] {p.stem}  {len(body)} chars  '
          f'{time.time()-started:.0f}s', flush=True)

print(f'\\nTotal {time.time()-t0:.0f}s')
"""),

    md("## 7. Spot-check one before downloading everything"),
    code("""
print((OUT / (imgs[0].stem + '.md')).read_text(encoding='utf-8')[:1500])
"""),

    md("""
## 8. Download

Unzip into `modules/06_evaluation/predictions/qwen_3b/`, then locally:

```
python modules/06_evaluation/src/ocr_bench.py --engine qwen_3b --verbose
```

That scores it against the same pages as the 0.573 CER baseline.
"""),
    code("""
import shutil
from google.colab import files
shutil.make_archive('/content/markdown_out', 'zip', OUT)
files.download('/content/markdown_out.zip')
"""),

    md("""
## What to look for

- **`[?]` markers** — the model admitting it cannot read. Good; that is
  the behaviour TrOCR lacked.
- **Fluent text that is not on the page** — fabrication. Compare a
  couple of outputs against the images by eye before trusting the CER.
- **`![diagram]`** where a figure is, rather than a prose description
  of it.
- **Question headings** (`### 2a)`) — these are what the reconstruction
  step groups on, so they matter more than the prose around them.
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
