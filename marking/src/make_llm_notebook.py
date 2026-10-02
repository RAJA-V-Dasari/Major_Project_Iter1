"""
Generate the Colab notebook that runs the model tier.

    data/marking/queue_llm.jsonl  ->  marking/notebooks/grade_llm.ipynb
                            ->  (on Colab) verdicts.jsonl

WHY A NOTEBOOK AND NOT AN API CALL
----------------------------------
This project pays for no inference. The model tier is Qwen2.5-7B-Instruct
in 4-bit on a free T4, which is why it arrives as a notebook you upload
the queue to rather than as a function this repo can call. The queue
records are self-contained by design - answer, rubric item, marks and
the chain note in one line - so the notebook needs no access to this
repo and no access to the corpus.

THE RULE THE PROMPT ENFORCES
----------------------------
**The model may not award a mark it cannot quote the student's own words
to support.** The quote is a required field, and `apply_verdicts.py`
re-checks every returned quote against the answer and rejects any award
whose quote is not actually there. A model that invents a justification
therefore cannot inflate the corpus - the check is downstream of it and
does not trust it.

That matters more than it sounds. The failure mode of a generous judge
is not obvious wrongness, it is plausible wrongness at scale: 740 items
each awarded on a reason that reads fine and cites nothing. Requiring a
verbatim span makes that failure mechanically detectable.

WHAT THE MODEL IS ALLOWED TO SAY
--------------------------------
Three verdicts, and declining is a first-class answer:

  award    the answer contains the claim - with the quote that shows it
  zero     the answer addresses this question but not this claim
  decline  cannot be determined from the text

`decline` exists because 740 of these items sit behind answers that
carry drawings the model cannot see. Guessing there would be inventing;
declining sends the item to a human holding the actual crop.

CARRY-FORWARD
-------------
Chain questions ship their `chain` note into the prompt. A student who
takes a wrong block size at step one produces later values that are
correct relative to their own error, and the model is told to mark those
against the student's own preceding values rather than against the
scheme's absolutes.

Run:
    python marking/src/make_llm_notebook.py
    python marking/src/make_llm_notebook.py --check     # compile the cells only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inspect

import llm_prompt
import paths

# The prompt is NOT defined here any more. It lives in llm_prompt.py,
# which llm_local.py imports too, because this repository now runs this
# tier in two places and a marking prompt with two definitions is two
# prompts the moment either is edited.
#
# The notebook cannot import that module - it runs on a Colab VM with no
# access to this repo - so the prompt is serialised INTO the cell below,
# along with the SOURCE of build_prompt() and parse_reply(). Serialising
# the source rather than retyping the functions is the point: the
# notebook cannot drift from the local runner, because there is nothing
# to drift from.
MODEL_ID = llm_prompt.REFERENCE_MODEL
SYSTEM_PROMPT = llm_prompt.SYSTEM_PROMPT
USER_TEMPLATE = llm_prompt.USER_TEMPLATE


def build_cells(queue_path):
    """The notebook, as a list of cells. Code is compiled before writing."""

    setup = '''\
# --- Qwen2.5-7B-Instruct, 4-bit, on a free T4 -----------------------
# Runtime -> Change runtime type -> T4 GPU, before running this.
!pip -q install -U "transformers>=4.44" accelerate bitsandbytes

import torch, json, os, re, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "''' + MODEL_ID + '''"

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=quant, device_map="auto", torch_dtype=torch.float16
)
model.eval()
print(torch.cuda.get_device_name(0))
'''

    upload = '''\
# --- the queue ------------------------------------------------------
# Upload data/marking/queue_llm.jsonl from the repo.
from google.colab import files
uploaded = files.upload()
QUEUE = list(uploaded)[0]

items = [json.loads(line) for line in open(QUEUE, encoding="utf-8")]
print(len(items), "items to mark")

# Checkpoint to Drive so a dropped session resumes instead of restarting.
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/grade_llm_verdicts.jsonl"
except Exception:
    OUT = "verdicts.jsonl"
print("writing to", OUT)
'''

    # Serialised out of llm_prompt, not retyped here. `inspect.getsource`
    # is what keeps this notebook and llm_local.py running the same code
    # rather than merely the same intentions: there is one definition, and
    # the notebook is a copy of it made at generation time.
    prompts = (
        "SYSTEM_PROMPT = " + json.dumps(SYSTEM_PROMPT) + "\n\n"
        "USER_TEMPLATE = " + json.dumps(USER_TEMPLATE) + "\n\n"
        + inspect.getsource(llm_prompt.build_prompt) + "\n"
        + "JSON_RE = re.compile(" + repr(llm_prompt.JSON_RE.pattern)
        + ", re.S)\n\n"
        + inspect.getsource(llm_prompt.parse_reply) + "\n"
    )

    run = '''\
# --- mark, with resume ----------------------------------------------
def done_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    for line in open(path, encoding="utf-8"):
        try:
            v = json.loads(line)
            keys.add((v["booklet_id"], v["question"], v["item_index"]))
        except Exception:
            pass
    return keys

def mark(record):
    # JSON_RE and parse_reply come from the cell above, serialised out of
    # the repo's llm_prompt.py. Do not redefine them here.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(record)},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=300, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(
        out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return parse_reply(reply)

already = done_keys(OUT)
print(len(already), "already done; resuming")

started = time.time()
with open(OUT, "a", encoding="utf-8") as handle:
    for n, record in enumerate(items, 1):
        key = (record["booklet_id"], record["question"], record["item_index"])
        if key in already:
            continue
        verdict = mark(record)
        handle.write(json.dumps({
            "booklet_id": record["booklet_id"],
            "question": record["question"],
            "item_index": record["item_index"],
            "marks_available": record["marks_available"],
            "verdict": verdict.get("verdict", "decline"),
            "marks": verdict.get("marks", 0),
            "quote": verdict.get("quote", ""),
            "reason": verdict.get("reason", ""),
            "model": MODEL_ID,
        }) + "\\n")
        handle.flush()
        if n % 25 == 0:
            rate = (time.time() - started) / n
            print(f"{n}/{len(items)}  {rate:.1f}s/item  "
                  f"~{(len(items)-n)*rate/60:.0f} min left", flush=True)

print("done ->", OUT)
'''

    download = '''\
# --- bring it home --------------------------------------------------
from collections import Counter
verdicts = [json.loads(l) for l in open(OUT, encoding="utf-8")]
print(len(verdicts), "verdicts")
print(Counter(v["verdict"] for v in verdicts))
print("awards with an empty quote (these WILL be rejected):",
      sum(1 for v in verdicts if v["verdict"] == "award" and not v["quote"]))

files.download(OUT)
# Then, in the repo:
#     python marking/src/apply_verdicts.py verdicts.jsonl
'''

    markdown = f'''\
# Model tier — marking the queued rubric items

Qwen2.5-7B-Instruct in 4-bit on a free T4. Set **Runtime → Change runtime
type → T4 GPU** first.

This marks only the items the cheap tiers could not settle honestly.
Each queue record is self-contained, so this notebook needs no access to
the repository or to the exam corpus.

**The rule:** the model may not award a mark it cannot quote the
student's own words to support. `apply_verdicts.py` re-checks every
quote against the answer and discards any award whose quote is not
actually there — so a model that invents a justification cannot inflate
the marks.

Checkpoints to Drive after every item; re-running resumes where it
stopped.
'''

    return [
        ("markdown", markdown),
        ("code", setup),
        ("code", upload),
        ("code", prompts),
        ("code", run),
        ("code", download),
    ]


def to_notebook(cells):
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None}
                   if kind == "code" else {}),
            }
            for kind, source in cells
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compile the code cells, write nothing")
    args = parser.parse_args()

    cells = build_cells(paths.OUT_DIR / "queue_llm.jsonl")

    # Refuse to write a notebook whose cells do not compile. A syntax
    # error found on Colab costs a GPU session and an upload; found here
    # it costs nothing.
    #
    # Notebook magics (!pip, %cd) are valid in a cell and not valid
    # Python, so they are blanked before compiling rather than being
    # allowed to defeat the check.
    for index, (kind, source) in enumerate(cells):
        if kind != "code":
            continue
        python_only = "\n".join(
            "" if line.lstrip().startswith(("!", "%")) else line
            for line in source.splitlines()
        )
        try:
            compile(python_only, f"<cell {index}>", "exec")
        except SyntaxError as error:
            raise SystemExit(f"cell {index} does not compile: {error}")
    print(f"{sum(1 for k, _ in cells if k == 'code')} code cells compile")

    if args.check:
        return

    out_dir = paths.ROOT / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "grade_llm.ipynb"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(to_notebook(cells), handle, indent=1)

    queue = paths.OUT_DIR / "queue_llm.jsonl"
    count = sum(1 for _ in open(queue, encoding="utf-8")) if queue.exists() else 0
    print(f"{out}")
    print(f"upload {queue} ({count} items) into it")


if __name__ == "__main__":
    main()
