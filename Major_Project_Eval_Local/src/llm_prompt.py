"""
The model tier's prompt, and the parsing of what comes back.

This module holds no model and talks to no backend. It exists so the
prompt has exactly ONE definition in this repository.

WHY THIS FILE EXISTS AT ALL
---------------------------
In the Colab original the prompt lived inside `make_llm_notebook.py` and
was serialised into a notebook. That was fine while there was one place
that ran it. This copy runs the tier locally as well, and two copies of
a marking prompt is two prompts: edit one, and the verdicts produced
either side of the edit are no longer the same experiment, while every
file still claims they are.

So `llm_local.py` and `make_llm_notebook.py` both import from here. The
prompt is verbatim the one the recorded Colab run used, character for
character, which is what makes `output/grade_llm_verdicts.jsonl` and a
fresh local run comparable at all. **Changing anything below invalidates
that comparison** - so if you change it, say so in
`docs/LOCAL_SETUP.md`, and do not pretend a local run reproduces the
recorded one.

THE RULE THE PROMPT ENFORCES
----------------------------
The model may not award a mark it cannot quote the student's own words
to support. That is a request, not a guarantee, and it is not enforced
here: `apply_verdicts.py` re-checks every returned quote against the
answer and discards any award whose quote is not actually there. On the
recorded run 15.4% of the model's awards failed that check. The
enforcement is downstream of the model and does not trust it, and moving
it into this file would be moving it inside the thing it checks.
"""

import json
import re

# The Colab original ran Qwen2.5-7B-Instruct in 4-bit on a T4. Locally
# the same weights arrive as a GGUF quant through a different runtime,
# so the identifier is backend-specific and lives with each backend.
# This is the reference the local names are meant to match.
REFERENCE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = """\
You are marking one rubric item of a university Computer Networks exam \
answer, against the department's official scheme.

You are given: the question, the scheme's model solution, ONE rubric \
item with its marks, and the student's answer as transcribed from their \
handwriting.

Decide ONE of three things:

  "award"   - the answer contains this specific claim. You MUST supply a \
verbatim quote from the student's answer that shows it.
  "zero"    - the answer engages with this question but does not make \
this claim.
  "decline" - you cannot tell from the text alone.

RULES

1. Quote or do not award. The "quote" field must be copied EXACTLY from \
the student's answer, character for character. It is checked against the \
answer automatically and an award whose quote is not found is discarded. \
Never paraphrase, never repair spelling, never quote the question or the \
model solution.

2. Mark the claim, not the topic. Being about the right subject is not \
the same as making the claim. Wrong subnets are still about subnetting.

3. Spelling and grammar are the student's own and are never penalised. \
The text is a transcription of handwriting, so "recieve" and "tunelling" \
are the student's words, not errors to mark down.

4. If the answer's evidence would be in a diagram you cannot see, \
DECLINE. Do not infer what a drawing probably showed. You are told when \
an answer carries drawings.

5. Partial credit only where the item's marks allow it. Award the item's \
full marks, or zero, unless the item is worth more than 1 mark and the \
student has clearly made part of the claim.

6. On a chain question you are given a carry-forward note. Mark later \
steps against the student's OWN earlier values, not the scheme's. One \
arithmetic slip is charged once, at the step where it happens.

Reply with ONLY a JSON object:

{"verdict": "award" | "zero" | "decline",
 "marks": <number>,
 "quote": "<verbatim from the student's answer, or empty>",
 "reason": "<one short sentence>"}
"""

USER_TEMPLATE = """\
QUESTION ({cie}, {question}, worth {q_marks} marks overall)
{question_text}

SCHEME'S MODEL SOLUTION
{model_solution}

THE ONE RUBRIC ITEM YOU ARE MARKING  [{marks} marks]
{point}
{chain_note}{figure_note}
STUDENT'S ANSWER
{answer}
"""


def build_prompt(record):
    """One queue record -> the user turn.

    The record is self-contained by design: answer, rubric item, marks
    and the chain note arrive in one line, so nothing here needs the
    corpus, the keys, or any other module.
    """

    chain_note = ""
    if record.get("chain"):
        steps = "; ".join(record["chain"].get("steps", []))
        chain_note = (
            "\nCARRY-FORWARD (this question is a chain)\n"
            + record["chain"].get("carry_forward", "")
            + "\nThe scheme's steps: " + steps + "\n"
        )

    figure_note = ""
    if record.get("answer_has_figures"):
        figure_note = (
            "\nNOTE: this answer also contains "
            + str(record["answer_has_figures"])
            + " drawing(s) that are NOT shown to you. If the evidence for "
              "this item would be in a drawing, reply decline.\n"
        )

    return USER_TEMPLATE.format(
        cie="CIE " + str(record["cie"]),
        question=record["question"],
        q_marks=record.get("question_marks", "?"),
        question_text=record["question_text"],
        model_solution=record["model_solution"],
        marks=record["marks_available"],
        point=record["point"],
        chain_note=chain_note,
        figure_note=figure_note,
        answer=record["answer"],
    )


JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_reply(reply):
    """The model's raw text -> a verdict dict.

    Unparseable output becomes a `decline`, never a zero and never an
    award. A backend that returns rubbish must not be able to settle an
    item in either direction: declining sends it to a person, which is
    the correct destination for "we do not know".
    """

    match = JSON_RE.search(reply or "")
    if not match:
        return {"verdict": "decline", "marks": 0, "quote": "",
                "reason": "model produced no JSON"}
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return {"verdict": "decline", "marks": 0, "quote": "",
                "reason": "model produced unparseable JSON"}
    if not isinstance(parsed, dict):
        return {"verdict": "decline", "marks": 0, "quote": "",
                "reason": "model produced JSON that is not an object"}
    return parsed


def verdict_row(record, verdict, model):
    """The line written to the verdicts file.

    Shaped for `apply_verdicts.py`, which is unchanged from the Colab
    version and does not care which backend produced the row - only
    that the quote is really in the answer.
    """

    return {
        "booklet_id": record["booklet_id"],
        "question": record["question"],
        "item_index": record["item_index"],
        "marks_available": record["marks_available"],
        "verdict": verdict.get("verdict", "decline"),
        "marks": verdict.get("marks", 0),
        "quote": verdict.get("quote", ""),
        "reason": verdict.get("reason", ""),
        "model": model,
    }


def record_key(row):
    """What identifies an item, for resume and for comparison."""

    return (row["booklet_id"], row["question"], row["item_index"])
