"""
The model tier, read in session by Claude instead of a local 7B.

    python marking/src/claude_tier.py --status
    python marking/src/claude_tier.py --next 12            # the next answers to mark
    python marking/src/claude_tier.py --ingest batch.jsonl  # record the verdicts

    data/marking/queue_llm.jsonl  ->  data/marking/claude_verdicts.jsonl
                            ->  python marking/src/apply_verdicts.py data/marking/claude_verdicts.jsonl

WHY THIS EXISTS
---------------
The model tier was always a queue on disk rather than an in-process
call, so WHO reads the queue is swappable. This project already swapped
it once for page reading (08_pageread/src/record.py) and once for the
examiner's covers (gold/gold_notes.md: "Read by: Claude, in session").

Here the alternatives were worse on both axes that matter:

  * Colab uploads real students' answers to a hosted notebook.
  * The faithful local model, qwen2.5:7b Q4, needs 5.6 GB resident
    against a 4.9 GB budget on this machine. setup_local.py rates it
    SWAPS - 5-20x slower while looking merely slow - over 740 items.

WHAT DID NOT CHANGE
-------------------
The rules. Every verdict here is produced under llm_prompt.SYSTEM_PROMPT,
the same prompt the recorded Qwen run used, and every row goes through
apply_verdicts.py unchanged: an award whose quote is not verbatim in
the answer is discarded, and a zero on a drawing-backed or chain item
is refused (zero_blocked). Those checks were written not to trust the
model, and they do not trust this one either.

WHAT DID CHANGE, AND MUST BE SAID
---------------------------------
The model. Rows carry `model = claude-opus-5-5 (in session)`, so these
verdicts are NOT a reproduction of the recorded Qwen run and their
numbers must not be reported as one. They are a different, stronger
reader held to the same discipline.

WHY BY QUESTION, NOT BY BOOKLET
-------------------------------
`--next` walks the queue question by question across students. Reading
one model solution and then twenty answers to it is how a careful
examiner marks - each answer judged against the others rather than
against a memory of yesterday - and it means the solution is read once
per batch instead of once per answer.

THE SECOND PASS: SHOWN THE DRAWINGS
-----------------------------------
    python marking/src/claude_tier.py --vision-next 8
    python marking/src/claude_tier.py --vision-ingest batch.jsonl

The first pass is text-only, like every model-tier reader before it,
and declines wherever the evidence would be in a drawing. The second
pass takes what is still pending after `apply_verdicts.py`, renders each
answer's crops into contact sheets under `data/marking/claude_sheets/`, and is
read with the sheets open. Its rows are labelled
`claude-opus-5-5 (in session, shown the drawings)` and carry `shown`,
the number of crops the reader was given, which is the only thing that
lets `apply_verdicts.py` accept a zero on a drawing-backed answer or
decide an item whose rubric point is a drawing (`shown_all`).

An award whose evidence is IN a drawing has no text to quote. It names
the crop and says what it shows (`crop`, `crop_reads`); the audit counts
those apart, because the verbatim check could not run on them. The lost
page and the chain carve-outs are untouched by this pass - seeing pixels
does not bring a page back or settle carry-forward.

Page 1 of every booklet is the identity block and is never listed here.
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths                                               # noqa: E402
from apply_verdicts import quote_supports                  # noqa: E402
from llm_prompt import verdict_row                         # noqa: E402

MODEL = "claude-opus-5-5 (in session)"
VISION_MODEL = "claude-opus-5-5 (in session, shown the drawings)"

QUEUE = paths.OUT_DIR / "queue_llm.jsonl"
HUMAN_QUEUE = paths.OUT_DIR / "queue_human.jsonl"
VERDICTS = paths.OUT_DIR / "claude_verdicts.jsonl"
SHEETS = paths.OUT_DIR / "claude_sheets"

SHEET_WIDTH = 1000         # crops are 1445 wide; this keeps handwriting legible
SHEET_MAX_HEIGHT = 1500    # and the sheet under the size a viewer downscales


def load_queue():
    records = [json.loads(line) for line in QUEUE.read_text().splitlines()
               if line.strip()]
    return {(r["booklet_id"], r["question"], r["item_index"]): r
            for r in records}


def done_keys():
    if not VERDICTS.exists():
        return set()
    out = set()
    for line in VERDICTS.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out.add((row["booklet_id"], row["question"], row["item_index"]))
    return out


def groups(queue, done):
    """Pending items, grouped by answer, ordered by question then student."""

    grouped = OrderedDict()
    for key in sorted(queue, key=lambda k: (queue[k]["cie"], k[1], k[0], k[2])):
        if key in done:
            continue
        record = queue[key]
        grouped.setdefault((record["cie"], key[1], key[0]), []).append(record)
    return grouped


def show(batch, vision=False):
    """Print a batch compactly: each question's solution once."""

    last_question = None

    for (cie, question, booklet), records in batch:

        first = records[0]

        if (cie, question) != last_question:
            print("\n" + "#" * 78)
            print(f"# CIE {cie}  QUESTION {question}")
            print("#" * 78)
            print("QUESTION:", first["question_text"])
            print("\nMODEL SOLUTION:", first["model_solution"])
            if first.get("chain"):
                print("\nCHAIN / CARRY-FORWARD:",
                      first["chain"].get("carry_forward", ""))
            last_question = (cie, question)

        print("\n" + "=" * 78)
        print(f"b={booklet}  q={question}"
              + (f"   [{first['answer_has_figures']} DRAWING(S)"
               + (" - see the sheets]" if vision else " NOT SHOWN]")
               if first.get("answer_has_figures") else ""))
        print("ITEMS:")
        for record in records:
            print(f"  i={record['item_index']}  [{record['marks_available']}]"
                  f"  {record['point']}")
        print("ANSWER:", first["answer"] or "(empty)")


def ingest(path):
    """
    Record verdicts, after the same quote check apply_verdicts will run.

    Pre-checking here is not a second enforcement - apply_verdicts is the
    enforcement and is untouched. It exists so a quote I transcribed
    wrongly is caught while the answer is still in front of me, rather
    than silently turned into a discarded award later.
    """

    queue = load_queue()
    written, bad = 0, []

    with open(VERDICTS, "a", encoding="utf-8") as out:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            v = json.loads(line)
            key = (v["b"], str(v["q"]), int(v["i"]))
            record = queue.get(key)
            if record is None:
                bad.append((key, "not in the queue"))
                continue

            verdict = {"verdict": v["v"], "marks": v.get("m", 0),
                       "quote": v.get("quote", ""), "reason": v.get("r", "")}

            if verdict["verdict"] not in ("award", "zero", "decline"):
                bad.append((key, f"unknown verdict {verdict['verdict']!r}"))
                continue

            if verdict["verdict"] == "award":
                if float(verdict["marks"]) > float(record["marks_available"]):
                    bad.append((key, "awards more than the item is worth"))
                    continue
                ok, why = quote_supports(verdict["quote"], record["answer"])
                if not ok:
                    bad.append((key, why + f": {verdict['quote'][:50]!r}"))
                    continue

            out.write(json.dumps(verdict_row(record, verdict, MODEL)) + "\n")
            written += 1

    print(f"wrote {written} verdict(s) -> {VERDICTS}")
    for key, why in bad:
        print(f"  REJECTED {key}: {why}")
    status()


def status():
    queue = load_queue()
    done = done_keys() & set(queue)
    left = groups(queue, done)
    print(f"{len(done)}/{len(queue)} items marked, "
          f"{len(left)} answers left")


# ------------------------------------------------------------ second pass


def pending_now():
    """Every item the marks still leave undecided, as a queue record.

    Read from the marks rather than the queues: the queues are written by
    `grade.py` before any verdict is applied, so they cannot say what is
    still open. Run `apply_verdicts.py` first.
    """

    records = load_queue()
    for line in HUMAN_QUEUE.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("kind") == "figure":
                records[(record["booklet_id"], record["question"],
                         record["item_index"])] = record

    out = {}
    for path in sorted(paths.MARKS_DIR.glob("*.json")):
        report = json.loads(path.read_text())
        for question in report["questions"]:
            if not question["counted"]:
                continue
            for index, item in enumerate(question["items"]):
                key = (report["booklet_id"], question["id"], index)
                if item["awarded"] is None and key in records:
                    record = dict(records[key])
                    record["declined"] = item.get("llm_declined")
                    record["gaps"] = question.get("gaps")
                    out[key] = record
    return out


def crops_for(booklet, question):
    from serve import diagrams_for       # pulls in grade; only when needed
    return diagrams_for(booklet, question)


def pages_for(booklet, question):
    from serve import sections_for
    section = sections_for(booklet).get(question) or {}
    out = []
    for page_id in section.get("pages", []):
        number = page_id.rsplit("_p", 1)[-1]
        if number == "01":                  # the identity block
            continue
        out.append(paths.HANDOFF_DATA / booklet / "pages"
                   / f"page_{number}.png")
    return out


def build_sheets(booklet, question):
    """Every crop of one answer, labelled and stacked, as few PNGs."""

    from PIL import Image, ImageDraw

    crops = crops_for(booklet, question)
    SHEETS.mkdir(parents=True, exist_ok=True)
    for old in SHEETS.glob(f"{booklet}__{question}__*.png"):
        old.unlink()

    sheets, current, height = [], [], 0
    for number, crop in enumerate(crops):
        image = Image.open(crop["path"]).convert("L")
        image = image.resize((SHEET_WIDTH, max(
            1, round(image.height * SHEET_WIDTH / image.width))))
        if current and height + image.height + 28 > SHEET_MAX_HEIGHT:
            sheets.append(current)
            current, height = [], 0
        current.append((number, crop["page_id"], image))
        height += image.height + 28
    if current:
        sheets.append(current)

    out = []
    for n, parts in enumerate(sheets):
        sheet = Image.new("L", (SHEET_WIDTH,
                                sum(im.height + 28 for _, _, im in parts)), 255)
        draw = ImageDraw.Draw(sheet)
        y = 0
        for number, page_id, image in parts:
            draw.rectangle([0, y, SHEET_WIDTH, y + 26], fill=0)
            draw.text((8, y + 8), f"crop {number}   ({page_id})", fill=255)
            sheet.paste(image, (0, y + 28))
            y += image.height + 28
        path = SHEETS / f"{booklet}__{question}__{n}.png"
        sheet.save(path)
        out.append(path)
    return len(crops), out


def latest_rows():
    out = {}
    if VERDICTS.exists():
        for line in VERDICTS.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[(row["booklet_id"], row["question"], row["item_index"])] = row
    return out


def vision_next(count):
    pending = pending_now()
    latest = latest_rows()
    settled_here = set()
    for key, record in pending.items():
        row = latest.get(key, {})
        # Already read with the drawings: still pending only because a
        # refusal (chain, lost page) sent it on to a person.
        if row.get("model") == VISION_MODEL:
            settled_here.add(key)
        # A chain zero on an answer with nothing drawn: seeing pixels
        # cannot change it and apply_verdicts will refuse it again.
        elif (row.get("verdict") == "zero" and record.get("chain")
              and not record.get("answer_has_figures")):
            settled_here.add(key)
    grouped = groups(pending, settled_here)
    batch = list(grouped.items())[:count]
    show(batch, vision=True)
    print("\n" + "#" * 78 + "\n# WHAT TO LOOK AT\n" + "#" * 78)
    for (cie, question, booklet), records in batch:
        shown, sheets = build_sheets(booklet, question)
        print(f"\nb={booklet} q={question}: {shown} crop(s)")
        for record in records:
            if record.get("declined"):
                print(f"  i={record['item_index']} first pass: "
                      f"{record['declined'][:140]}")
        for sheet in sheets:
            print(f"  sheet {sheet}")
        if not sheets:
            for page in pages_for(booklet, question):
                print(f"  page  {page}")
    print(f"\n({len(batch)} of {len(grouped)} answers still pending)")


def vision_ingest(path):
    """Record second-pass verdicts: same checks, plus `shown` and crops."""

    pending = pending_now()
    shown_cache = {}
    written, bad = 0, []

    with open(VERDICTS, "a", encoding="utf-8") as out:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            v = json.loads(line)
            key = (v["b"], str(v["q"]), int(v["i"]))
            record = pending.get(key)
            if record is None:
                bad.append((key, "not pending"))
                continue
            if (key[0], key[1]) not in shown_cache:
                shown_cache[(key[0], key[1])] = len(crops_for(key[0], key[1]))
            shown = shown_cache[(key[0], key[1])]

            verdict = {"verdict": v["v"], "marks": v.get("m", 0),
                       "quote": v.get("quote", ""), "reason": v.get("r", "")}
            if verdict["verdict"] not in ("award", "zero", "decline"):
                bad.append((key, f"unknown verdict {verdict['verdict']!r}"))
                continue

            row = verdict_row(record, verdict, VISION_MODEL)
            row["shown"] = shown
            if v.get("pages"):
                # Only for an answer with nothing cropped: the reader
                # opened the pages themselves (never page 1).
                row["pages_seen"] = [p.stem for p in
                                     pages_for(key[0], key[1])]

            if verdict["verdict"] == "award":
                if float(verdict["marks"]) > float(record["marks_available"]):
                    bad.append((key, "awards more than the item is worth"))
                    continue
                ok, why = quote_supports(verdict["quote"], record["answer"])
                if not ok:
                    crop, reads = v.get("crop"), (v.get("reads") or "").strip()
                    page = v.get("page")
                    if isinstance(crop, int) and 0 <= crop < shown and reads:
                        row.update(quote="", crop=crop, crop_reads=reads)
                    elif page and page in row.get("pages_seen", []) and reads:
                        row.update(quote="", page=page, crop_reads=reads)
                    else:
                        bad.append((key, why + " and no crop or seen page "
                                                "named"))
                        continue

            out.write(json.dumps(row) + "\n")
            written += 1

    print(f"wrote {written} verdict(s) -> {VERDICTS}")
    for key, why in bad:
        print(f"  REJECTED {key}: {why}")
    print("now: python marking/src/apply_verdicts.py data/marking/claude_verdicts.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--next", type=int, metavar="N")
    parser.add_argument("--ingest", type=Path)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--vision-next", type=int, metavar="N")
    parser.add_argument("--vision-ingest", type=Path)
    args = parser.parse_args()

    if args.vision_next:
        return vision_next(args.vision_next)
    if args.vision_ingest:
        return vision_ingest(args.vision_ingest)
    if args.ingest:
        return ingest(args.ingest)
    if args.next:
        queue = load_queue()
        pending = list(groups(queue, done_keys()).items())
        show(pending[:args.next])
        print(f"\n({min(args.next, len(pending))} of {len(pending)} answers left)")
        return
    status()


if __name__ == "__main__":
    main()
