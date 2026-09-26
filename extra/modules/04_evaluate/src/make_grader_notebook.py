"""
Generate the Colab notebook for the LLM tier of the marking ladder.

`grade.py` settles what keyword coverage and sentence similarity can
honestly settle, and queues the rest to
`04_evaluate/output/marks/review_llm.jsonl`. That file is deliberately
self-contained - each line carries the question, the model solution, the
student's answer and the rubric items still outstanding - so the tier
that reads it needs nothing from this repo but the file itself.

    python make_grader_notebook.py
        -> 04_evaluate/output/grade_llm.ipynb
        -> upload that and review_llm.jsonl to Colab, run, download
           verdicts.jsonl, then: python grade.py --apply verdicts.jsonl

WHY COLAB AND NOT THE LAPTOP
----------------------------
The same asymmetry that put reading on a hosted GPU applies here. torch
is allowed locally now, but 399 generative calls over a 7B model on this
CPU is hours; on a T4 it is minutes. Nothing else in the ladder needs a
GPU, so this stays a detachable step rather than a dependency.

WHY THE MODEL IS ASKED FOR EVIDENCE, NOT A SCORE
------------------------------------------------
A model asked "how many marks is this worth" will produce a plausible
number for anything, including an answer that is confidently wrong -
which is the exact failure the semantic tier was demoted for. So it is
asked, per rubric item, for a quote from the student's answer that
supports the point, and awarded marks only when it can produce one. A
verdict with no quote is not a low score; it is an abstention, and it
goes to the human queue. That keeps the expensive tier honest in the
same way the cheap ones were made honest.
"""

import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
OUT = STAGE_DIR / "output" / "grade_llm.ipynb"

MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPT = """\
You are marking one question from a Computer Networks exam, against the
official marking scheme.

You will be given the question, the scheme's model solution, the
student's answer as transcribed from their handwriting, and a list of
rubric items that are still to be decided. Each item names one thing the
scheme awards marks for.

For each item, decide whether the student's answer actually contains
that point.

The one rule that matters: award marks only when you can quote the words
from the student's answer that earn them. Copy the quote exactly as it
appears. If you cannot find such a quote, do not award the marks and do
not invent a reason - say the answer does not contain the point, or that
you are not sure.

Judge the point, not the presentation. The answer was read from
handwriting, so spelling, spacing and broken tables are the reader's
faults and never the student's. A correct idea in the student's own
words earns full marks for that item. A partly correct idea earns part
of the item's marks. An answer that is about the right topic but states
something different from the scheme earns nothing for that item -
being on the subject is not the same as being right.
"""

# Appended only for the questions the key marks as multi-step. Without
# it the tier charges a single early slip once per rubric item, which on
# a five-subnet question is five times for one mistake.
METHOD_RULE = """\

MARKING THE METHOD ON THIS QUESTION

This question is a chain: the student computes a value, then computes
the next one from it, and the scheme awards marks at every link. You
will be given the steps the marks attach to, and a note saying what may
be carried forward.

Mark it the way a human examiner does. If the student makes one mistake
early and then works correctly from their own wrong value, charge that
mistake ONCE and award the later steps on the student's own figures. A
number that is wrong only because it inherited an earlier wrong number,
and is otherwise correctly derived, still earns its step.

This is not licence to be generous. Award nothing for a step whose
method is wrong, nothing for a value that appears with no working behind
it, and nothing where the student has simply written a different answer.
The question is always "did they do this step correctly, given what they
had" - not "does this match the scheme".

When you carry a value forward, say so in the note: which value you
accepted and which step it came from.
"""

# Last in every prompt, method question or not, so the output contract
# is the final thing the model reads.
OUTPUT_RULE = """\

Answer with one JSON object per line, and nothing else:

{"item": <the item number>, "awarded": <number>, "quote": "<exact words \
from the student's answer, or empty if none>", "note": "<one short \
sentence>"}

If you are genuinely unable to tell, set "awarded" to null. That is a
better answer than a guess.
"""


def lines_of(source):
    """Split into nbformat's `source` list.

    Every element except the last MUST keep its trailing newline. A
    notebook reader concatenates the list with "".join, so a list of
    bare lines is reassembled into one enormous line and every cell
    opens as a single unreadable row. Most local tooling is forgiving
    about this; Colab is not.
    """
    return source.strip().splitlines(keepends=True)


def code(source):
    return {"cell_type": "code", "metadata": {}, "source": lines_of(source),
            "execution_count": None, "outputs": []}


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": lines_of(source)}


CELLS = [
    md("""
# LLM tier - marking

Third rung of the ladder in `04_evaluate/src/grade.py`. Everything the
keyword and similarity tiers could settle is already settled; this marks
only what they could not.

**Runtime -> Change runtime type -> T4 GPU** before running.

Upload `review_llm.jsonl` (from `04_evaluate/output/marks/`) when the
next cell asks. Download `verdicts.jsonl` at the end and run:

```
python modules/04_evaluate/src/grade.py --apply verdicts.jsonl
```
"""),

    code("""
!pip -q install transformers accelerate bitsandbytes
"""),

    code("""
from google.colab import files
import pathlib, json

up = files.upload()                      # choose review_llm.jsonl
QUEUE = pathlib.Path(next(iter(up)))

groups = [json.loads(line) for line in
          QUEUE.read_text(encoding='utf-8').splitlines() if line.strip()]

items = sum(len(g['items']) for g in groups)
print(f'{len(groups)} questions, {items} rubric items to decide')
"""),

    code(f"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = {MODEL!r}

# 4-bit for the same reason reading uses it: a 7B model does not fit a
# free T4 in 16-bit, and marking is not a task where the last decimal of
# precision changes a verdict.
quant = BitsAndBytesConfig(load_in_4bit=True,
                           bnb_4bit_quant_type='nf4',
                           bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=quant, device_map='auto',
    torch_dtype=torch.float16)
model.eval()
print('loaded')
"""),

    code(f"""
PROMPT = {PROMPT!r}
METHOD_RULE = {METHOD_RULE!r}
OUTPUT_RULE = {OUTPUT_RULE!r}

# A chain question needs room to show its working before it can justify
# carrying a value forward; a prose question does not.
MAX_NEW, MAX_NEW_METHOD = 700, 1100


def build(group):
    method = group.get('method')

    lines = [PROMPT]
    if method:
        lines.append(METHOD_RULE)
    lines.append(OUTPUT_RULE)

    lines += ['', '--- QUESTION ---', group['question_text'], '',
              '--- MODEL SOLUTION (the scheme) ---', group['model_solution']]

    if method:
        lines += ['', '--- THE STEPS THE MARKS ATTACH TO ---']
        lines += [f'{{n}}. {{s}}' for n, s in enumerate(method['steps'], 1)]
        lines += ['', '--- WHAT MAY BE CARRIED FORWARD ---',
                  method['carry_forward']]

    lines += ['', '--- STUDENT ANSWER (read from handwriting) ---',
              group['answer'] or '(nothing was filed under this question)',
              '', '--- RUBRIC ITEMS TO DECIDE ---']

    for it in group['items']:
        lines.append(f"item {{it['item']}} "
                     f"(worth {{it['marks_available']}} marks): {{it['point']}}")

    return '\\n'.join(lines)


def ask(group):
    text = tok.apply_chat_template(
        [{{'role': 'user', 'content': build(group)}}],
        tokenize=False, add_generation_prompt=True)

    batch = tok([text], return_tensors='pt').to(model.device)

    budget = MAX_NEW_METHOD if group.get('method') else MAX_NEW

    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=budget,
                             do_sample=False, repetition_penalty=1.05)

    return tok.decode(out[0][batch.input_ids.shape[1]:],
                      skip_special_tokens=True)
"""),

    code("""
import re, time, pathlib

OUT = pathlib.Path('verdicts.jsonl')

# Resume: a disconnect should cost the calls not yet made, not all of
# them. Same rule the reading notebook uses.
done = set()
if OUT.exists():
    for line in OUT.read_text(encoding='utf-8').splitlines():
        if line.strip():
            v = json.loads(line)
            done.add((v['booklet'], v['question'], v['item']))

start = time.time()

with OUT.open('a', encoding='utf-8') as fh:
    for n, g in enumerate(groups, 1):

        want = {(g['booklet'], g['question'], i['item']) for i in g['items']}
        if want <= done:
            continue

        try:
            raw = ask(g)
        except Exception as exc:
            print(f"  !! {g['booklet']} {g['question']}: {exc}")
            continue

        found = 0
        for line in raw.splitlines():
            line = line.strip().strip('`')
            if not line.startswith('{'):
                continue
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = (g['booklet'], g['question'], v.get('item'))
            if key in done or v.get('item') is None:
                continue

            v.update(booklet=g['booklet'], question=g['question'])
            fh.write(json.dumps(v, ensure_ascii=False) + '\\n')
            done.add(key)
            found += 1

        fh.flush()

        if n % 20 == 0 or found == 0:
            rate = (time.time() - start) / n
            print(f'{n}/{len(groups)}  {found} verdicts  '
                  f'{rate:.1f}s/question  '
                  f'~{rate * (len(groups) - n) / 60:.0f} min left')

print(f'\\ndone: {len(done)} verdicts in {(time.time() - start) / 60:.1f} min')
"""),

    code("""
# Sanity before downloading: a tier that awards marks without quoting the
# student is the failure this whole design is built against, so count it.
import collections

vs = [json.loads(l) for l in
      pathlib.Path('verdicts.jsonl').read_text(encoding='utf-8').splitlines()
      if l.strip()]

noquote = [v for v in vs if v.get('awarded') and not (v.get('quote') or '').strip()]
abstain = [v for v in vs if v.get('awarded') is None]

print(f'verdicts       : {len(vs)}')
print(f'awarded > 0    : {sum(1 for v in vs if v.get("awarded"))}')
print(f'awarded zero   : {sum(1 for v in vs if v.get("awarded") == 0)}')
print(f'abstained      : {len(abstain)}  (these go to the human queue)')
print(f'marks with no quote : {len(noquote)}   <-- should be 0')

from google.colab import files
files.download('verdicts.jsonl')
"""),
]


def main():

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
        "cells": CELLS,
    }

    # Same guard the reading notebook carries: a `source` list whose
    # non-final lines have lost their newline reassembles into one line
    # and the cell is unreadable in Colab.
    for index, cell in enumerate(notebook["cells"]):
        body = cell["source"]
        for line in body[:-1]:
            if not line.endswith("\n"):
                raise SystemExit(
                    f"cell {index}: line without a trailing newline would "
                    f"collapse the cell in Colab: {line!r}")
        source = "".join(body)

        # Shell magics are not Python and never will be; the reading
        # notebook's checker skips them the same way.
        if cell["cell_type"] == "code" and not source.lstrip().startswith("!"):
            compile(source, f"<cell {index}>", "exec")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"{len(CELLS)} cells, model {MODEL}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
