"""
The human tier's working tool: one booklet at a time, with the drawings.

    data/marking/marks/*.json + the handoff's crops  ->  http://127.0.0.1:8000

WHY A BOOKLET AND NOT A QUESTION
--------------------------------
The first version paged one question at a time and it was the wrong unit.
A booklet's questions share pages, share handwriting and share the
examiner's running judgement, so marking Q2b means holding what Q2a
already established - and a question-per-page tool turns that into
navigation. A booklet is also the unit the examiner worked in and the
unit the cover's total is written in, so it is the only unit where "do we
agree?" can be answered on the screen where the marking happened.

So: every counted question of one booklet on one page, one form, one
save, with the examiner's mark beside each question and their total in
the header.

WHY A SETTLED ITEM CAN BE OVERRIDDEN
------------------------------------
The human tier is the top of the ladder, not a fallback for the items
below it. On `student_01_cie_2` every cheap tier and the model agreed Q2a
was worth 5/5, all five awards quote-backed; the examiner gave 3. A tool
that displays that disagreement and then refuses to let the person
resolve it is showing them a problem while withholding the fix.

Overriding is deliberate rather than easy: every item defaults to "keep",
a settled one shows which tier decided it and on what evidence, and an
override is logged with the tier it displaced.

THE MARKS FILES ARE THE ONLY STATE
----------------------------------
There is no queue file. `data/marking/queue_human.jsonl` was written before the
model ran and has been wrong ever since. A second copy of the truth is a
second thing to get stale, so this reads `data/marking/marks/*.json` directly.

Every decision is appended to `data/marking/human_marks.jsonl` before the marks
file is touched. That log is what makes the pipeline replayable: a re-run
of grade.py wipes the marks back to the ladder's own verdicts, and
without the log a day of human work would go with it.

MOVING THROUGH THE WORK
-----------------------
The index is a worklist: filter to the booklets still undecided, or to
the ones whose examiner total falls outside our range, and "Continue
reviewing" opens the first one with work left. On a booklet, "Save & next
undecided" saves and jumps to the next booklet that still needs a person;
`n` / `p` do the same from the keyboard and `h` hides questions with
nothing left to decide. Drawings open full size on click.
/export/marks.csv and /export/questions.csv give the marks as they stand
now, including every decision made here since the last pipeline run.

WHAT IT WILL NOT DO
-------------------
  * bind anything but 127.0.0.1 - the pages quote real students
  * serve a path from the URL. A crop is addressed by position in data we
    loaded ourselves, so there is no filename to traverse with

Run:
    python marking/src/serve.py
    python marking/src/serve.py --port 8800
    python marking/src/serve.py --check      # no socket, just prove the data is there
"""

import argparse
import csv
import html
import io
import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
import load_handoff
import grade
from agreement import examiner_mark

HUMAN_LOG = paths.OUT_DIR / "human_marks.jsonl"

_BOOKLET_CACHE = {}
_ALIGNMENT = None


def sections_for(booklet_id):
    """{question id: merged content}, resolved exactly as the grader did.

    This goes through `grade.build_sections` rather than the booklet's own
    labels on purpose. Some parts reach a question only through `align.py`
    - a bare "2" that the paper's own text places at 2a - and a lookup by
    raw label silently finds nothing for them. Two questions in this
    corpus are in that position, both with drawings, and showing a human
    "no crops" on a question that has two would be the tool lying about
    the one thing it exists to show.
    """

    global _ALIGNMENT
    if _ALIGNMENT is None:
        _ALIGNMENT = grade.load_alignment()
    if booklet_id not in _BOOKLET_CACHE:
        booklet = load_handoff.load_booklet(booklet_id)
        with open(paths.KEYS_DIR / f"cie{booklet.cie}.json",
                  encoding="utf-8") as handle:
            valid = {q["id"] for q in json.load(handle)["questions"]}
        _BOOKLET_CACHE[booklet_id] = grade.build_sections(
            booklet, _ALIGNMENT, valid)
    return _BOOKLET_CACHE[booklet_id]


def diagrams_for(booklet_id, question_id):
    """The crops the student drew under this question, in page order."""

    section = sections_for(booklet_id).get(question_id)
    return list(section["diagrams"]) if section else []


def examiner_total(gold_row):
    if not gold_row or gold_row.get("confidence") == "no_gold":
        return None
    for field in ("total_box", "total_grid"):
        value = (gold_row.get(field) or "").strip()
        if value:
            try:
                return float(value)
            except ValueError:
                pass
    return None


def build_booklets():
    """All 50, in order, whether or not anything is still undecided."""

    with open(paths.GOLD_CSV, encoding="utf-8", newline="") as handle:
        gold = {r["booklet_id"]: r for r in csv.DictReader(handle)}

    booklets = []
    for path in sorted(paths.MARKS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        row = gold.get(report["booklet_id"])
        booklets.append({
            "booklet": report["booklet_id"],
            "cie": report["cie"],
            "path": path,
            "examiner_total": examiner_total(row),
            "marks": [examiner_mark(row, q["id"]) if q["counted"] else None
                      for q in report["questions"]],
        })
    return booklets


def read_report(entry):
    with open(entry["path"], encoding="utf-8") as handle:
        return json.load(handle)


def counted(report):
    """[(question index, question)] for the questions that score."""

    return [(i, q) for i, q in enumerate(report["questions"]) if q["counted"]]


def open_items(report):
    return sum(1 for _, q in counted(report)
               for i in q["items"] if i["awarded"] is None)


def next_open(booklets, index):
    """The next booklet after `index` that still holds undecided items.

    Wraps round, so the last booklet leads back to the first one left.
    None when nothing anywhere is undecided.
    """

    total = len(booklets)
    for step in range(1, total + 1):
        candidate = (index + step) % total
        if open_items(read_report(booklets[candidate])):
            return candidate
    return None


def standing(report, examiner):
    """How our range sits against the examiner's total, as a word."""

    ours, pending = report["marks_settled"], report["marks_pending"]
    if examiner is None:
        return "none"
    if not pending and abs(ours - examiner) < 1e-9:
        return "agreed"
    if ours - 1e-9 <= examiner <= ours + pending + 1e-9:
        return "reachable"
    return "above" if examiner > ours + pending else "below"


def retotal(report):
    """Recompute the booklet's settled/pending after a decision lands."""

    for question in report["questions"]:
        question["marks_settled"] = round(sum(
            i["awarded"] for i in question["items"]
            if i["awarded"] is not None), 2)
        question["marks_pending"] = round(sum(
            i["marks_available"] for i in question["items"]
            if i["awarded"] is None), 2)
    report["marks_settled"] = round(sum(
        q["marks_settled"] for q in report["questions"] if q["counted"]), 2)
    report["marks_pending"] = round(sum(
        q["marks_pending"] for q in report["questions"] if q["counted"]), 2)


def record(entry, decisions, note):
    """Write one booklet's decisions to the log, then to the marks.

    The log first, deliberately. If the process dies between the two, a
    replay can still recover the judgement; the reverse order would lose
    it.
    """

    report = read_report(entry)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = []
    for (qindex, index), marks in sorted(decisions.items()):
        question = report["questions"][qindex]
        item = question["items"][index]
        marks = max(0.0, min(float(marks), item["marks_available"]))
        if item["awarded"] is not None and item["awarded"] == marks:
            continue                           # nothing actually changed
        displaced = item["tier"] if item["awarded"] is not None else None
        lines.append(json.dumps({
            "booklet_id": entry["booklet"], "question": question["id"],
            "item_index": index, "marks": marks,
            "marks_available": item["marks_available"],
            "overrode": displaced, "note": note, "at": stamp,
        }) + "\n")
        why = note or "marked by a human, with the drawings"
        if displaced:
            why = f"{why} (overrides the {displaced} tier)"
        item.update(awarded=marks, tier="human", why=why, evidence=[])
        for stale in ("quote", "llm_declined", "llm_model", "crop",
                      "page", "crop_reads"):
            item.pop(stale, None)              # apply_human drops the same

    applied = len(lines)
    if applied:
        paths.OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(HUMAN_LOG, "a", encoding="utf-8") as log:
            log.writelines(lines)
        retotal(report)
        with open(entry["path"], "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return applied


# ---------------------------------------------------------------- HTML

CSS = """
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;
     color:#1a1a1a;background:#f4f4f2}
header{position:sticky;top:0;background:#1a1a1a;color:#fff;padding:9px 18px;
       display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;z-index:9}
header b{font-size:17px}
header .sp{margin-left:auto}
header a{color:#8ecbff;text-decoration:none;margin-left:10px}
nav{background:#2b2b2b;color:#bbb;padding:7px 18px;position:sticky;top:42px;
    z-index:8;font-size:14px}
nav a{color:#8ecbff;text-decoration:none;margin-right:13px}
nav a.open{color:#ffd479;font-weight:600}
.badge{padding:2px 9px;border-radius:11px;font-size:13px;background:#2f6f3e;
       color:#fff}
.badge.none{background:#666}
.badge.warn{background:#9a5b12}
main{max-width:1560px;margin:0 auto;padding:18px}
.q{background:#fff;border:1px solid #ddd;border-radius:8px;margin-bottom:20px;
   overflow:hidden;scroll-margin-top:90px}
.qhead{background:#fafaf8;border-bottom:1px solid #e5e5e5;padding:11px 16px;
       display:flex;gap:13px;align-items:baseline;flex-wrap:wrap}
.qhead b{font-size:16px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:16px}
@media(max-width:1150px){.cols{grid-template-columns:1fr}}
h3{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#777;
   margin:0 0 9px}
.answer{white-space:pre-wrap}
.empty{color:#999;font-style:italic}
figure{margin:13px 0 0}
figure img{width:100%;border:1px solid #ccc;border-radius:5px;background:#fff}
figcaption{font-size:12px;color:#777;margin-top:3px}
.item{border-top:1px solid #eee;padding:12px 0}
.item:first-of-type{border-top:0}
.pt{font-weight:600}
.meta{font-size:13px;color:#666;margin-top:3px}
.settled{border-left:3px solid #2f6f3e;padding-left:11px;background:#f6faf6}
.quote{font-size:13px;color:#444;background:#f7f7f5;border-left:2px solid #bbb;
       padding:5px 9px;margin-top:5px}
.controls{margin-top:7px;display:flex;gap:5px;align-items:center;
          flex-wrap:wrap}
.controls label{font-size:13px;display:flex;align-items:center;gap:4px;
                background:#f2f2f0;border:1px solid #ddd;border-radius:5px;
                padding:3px 9px;cursor:pointer}
.controls label:hover{background:#e8e8e4}
.controls input[type=number]{width:60px;padding:3px}
.bar{position:sticky;bottom:0;background:#fff;border-top:2px solid #1a1a1a;
     padding:12px 18px;display:flex;gap:12px;align-items:center;z-index:7}
button{background:#1a1a1a;color:#fff;border:0;border-radius:6px;
       padding:10px 22px;font-size:15px;cursor:pointer}
input[type=text]{flex:1;padding:8px;border:1px solid #ccc;border-radius:5px}
table{border-collapse:collapse;width:100%;font-size:14px;background:#fff}
td,th{border-bottom:1px solid #eee;padding:7px 10px;text-align:left}
th{background:#fafaf8;font-size:12px;text-transform:uppercase;color:#777}
tr.done td{color:#999}
a.row{color:#1a4f8a;text-decoration:none;font-weight:600}
.ok{color:#2f6f3e;font-weight:600}
.off{color:#9a2f2f;font-weight:600}
.flash{background:#e7f3e9;border:1px solid #9cc9a5;border-radius:6px;
       padding:8px 14px;margin-bottom:16px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.stat{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 16px}
.stat b{display:block;font-size:22px}
.stat span{font-size:12px;color:#777;text-transform:uppercase;
           letter-spacing:.06em}
.filters a{margin-right:14px}
.filters a.on{color:#ffd479;font-weight:600}
a.go{background:#2f6f3e;color:#fff;border-radius:6px;padding:8px 16px;
     text-decoration:none;font-weight:600;display:inline-block}
button.alt{background:#fff;color:#1a1a1a;border:1px solid #1a1a1a}
.keys{font-size:12px;color:#777}
kbd{border:1px solid #bbb;border-radius:3px;padding:0 4px;font-size:11px;
    background:#fafafa}
body.hide-decided .q.decided{display:none}
figure a{display:block}
"""

# Keyboard: n / p move between booklets that still need work, h hides the
# questions with nothing left to decide. None of it fires while typing.
SCRIPT = """
<script>
(function(){
  var b=document.body, key='hideDecided';
  try{ if(localStorage.getItem(key)==='1') b.classList.add('hide-decided'); }catch(e){}
  window.toggleDecided=function(){
    b.classList.toggle('hide-decided');
    try{ localStorage.setItem(key, b.classList.contains('hide-decided')?'1':'0'); }catch(e){}
  };
  document.addEventListener('keydown',function(ev){
    var t=ev.target.tagName;
    if(t==='INPUT'||t==='TEXTAREA'||ev.ctrlKey||ev.metaKey||ev.altKey) return;
    var link=document.querySelector('[data-key="'+ev.key+'"]');
    if(link){ ev.preventDefault(); link.click(); }
    if(ev.key==='h' && window.toggleDecided){ toggleDecided(); }
  });
})();
</script>
"""


def page(title, body):
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style>"
            f"</head><body>{body}{SCRIPT}</body></html>").encode("utf-8")


def mark_controls(field, avail, current):
    """The radio set for one rubric item.

    Everything defaults to "keep" so that saving a page you only scrolled
    through changes nothing - which matters far more now that one save
    covers a whole booklet instead of a single question.
    """

    label = "keep" if current is not None else "leave"
    out = [f"<label><input type='radio' name='v{field}' value='keep' checked>"
           f"{label}</label>"]
    options = [0.0, avail] if avail <= 1 else [0.0, avail / 2, avail]
    for value in options:
        tag = f"{value:g}" + (" (full)" if value == avail else "")
        out.append(f"<label><input type='radio' name='v{field}' "
                   f"value='{value:g}'>{tag}</label>")
    out.append(f"<label>or <input type='number' name='n{field}' step='0.5' "
               f"min='0' max='{avail:g}' placeholder='--'></label>")
    return "<div class='controls'>" + "".join(out) + "</div>"


def render_booklet(index, total, entry, report, next_index=None, saved=None):
    e = html.escape
    booklet = entry["booklet"]
    rows = counted(report)
    ours = sum(q["marks_settled"] for _, q in rows)
    pending = sum(q["marks_pending"] for _, q in rows)
    available = sum(q["marks_available"] for _, q in rows)
    ex_total = entry["examiner_total"]

    if ex_total is None:
        verdict = "<span class='badge none'>no examiner total</span>"
    elif ours - 1e-9 <= ex_total <= ours + pending + 1e-9:
        verdict = (f"<span class='badge'>examiner {ex_total:g}"
                   + (" &middot; agreed" if not pending else
                      " &middot; reachable") + "</span>")
    else:
        side = "above" if ex_total > ours + pending else "below"
        verdict = (f"<span class='badge warn'>examiner {ex_total:g} "
                   f"&middot; {side} our range</span>")

    head = (
        f"<header><b>{e(booklet)}</b><span>CIE {report['cie']}</span>"
        f"<span>ours <b>{ours:g}</b>"
        + (f" (+{pending:g} pending)" if pending else "")
        + f" / {available:g}</span>{verdict}"
        f"<span class='sp'>booklet {index + 1} of {total}</span>"
        f"<a href='/b/{max(0, index - 1)}' data-key='p'>&larr; prev</a>"
        f"<a href='/b/{min(total - 1, index + 1)}'>next &rarr;</a>"
        + (f"<a href='/b/{next_index}' data-key='n'>next undecided "
           "&rarr;</a>" if next_index is not None else "")
        + "<a href='/?show=todo'>worklist</a></header>"
    )

    jump = ["<nav>"]
    for qindex, question in rows:
        left = sum(1 for i in question["items"] if i["awarded"] is None)
        cls = " class='open'" if left else ""
        jump.append(f"<a href='#q{qindex}'{cls}>Q{e(question['id'])}"
                    + (f" ({left})" if left else "") + "</a>")
    jump.append("<a href='#' onclick='toggleDecided();return false' "
                "style='float:right'>show / hide decided questions</a>"
                "<span class='keys' style='float:right;margin-right:14px'>"
                "<kbd>n</kbd> next undecided &middot; <kbd>p</kbd> prev "
                "&middot; <kbd>h</kbd> hide decided</span></nav>")

    body = [f"<form method='post' action='/decide/{index}'><main>"]
    if saved is not None:
        body.append("<div class='flash'>"
                    + (f"Saved {saved} change(s)." if saved else
                       "Saved - nothing had changed.")
                    + "</div>")

    for qindex, question in rows:
        examiner = entry["marks"][qindex]
        avail = question["marks_available"]
        settled = question["marks_settled"]
        left = question["marks_pending"]
        if examiner is None:
            compare = "<span class='empty'>no examiner mark</span>"
        elif not left and abs(settled - examiner) < 1e-9:
            compare = f"<span class='ok'>examiner {examiner:g} &middot; agreed</span>"
        elif settled - 1e-9 <= examiner <= settled + left + 1e-9:
            compare = f"<span>examiner {examiner:g}</span>"
        else:
            compare = f"<span class='off'>examiner {examiner:g}</span>"

        body.append(
            f"<div class='q{'' if left else ' decided'}' id='q{qindex}'>"
            "<div class='qhead'>"
            f"<b>Q{e(question['id'])}</b>"
            f"<span>part {e(str(question['part']))} &middot; {avail:g} marks"
            "</span>"
            f"<span>ours <b>{settled:g}</b>"
            + (f" (+{left:g} pending)" if left else "")
            + f"</span>{compare}</div><div class='cols'><div>")

        answer = (e(question["answer"]) if question["answer"].strip() else
                  "<span class='empty'>No transcribed text. If anything was "
                  "written, it is in the drawings below.</span>")
        body.append("<h3>The student's answer</h3>"
                    f"<div class='answer'>{answer}</div>")

        crops = diagrams_for(booklet, question["id"])
        if crops:
            body.append(f"<h3 style='margin-top:18px'>{len(crops)} drawing(s)"
                        "</h3>")
            for n, crop in enumerate(crops):
                src = f"/crop/{index}/{qindex}/{n}"
                body.append(
                    f"<figure><a href='{src}' target='_blank' "
                    f"title='open full size'><img src='{src}' "
                    f"loading='lazy'></a><figcaption>drawing {n + 1} "
                    f"&middot; page {e(str(crop.get('page_id')))} "
                    "&middot; click to enlarge</figcaption></figure>")
        elif question["figures"]:
            body.append("<p class='empty'>Recorded as having "
                        f"{question['figures']} drawing(s), but the crops "
                        "could not be located in the handoff.</p>")
        if question["gaps"]:
            body.append("<p class='empty'>A page of this part was lost "
                        "before it was ever read. Judge it as incomplete, "
                        "not wrong.</p>")

        body.append("</div><div><h3>The scheme's rubric</h3>")
        for n, item in enumerate(question["items"]):
            avail_i = item["marks_available"]
            field = f"{qindex}_{n}"
            reader = (f" ({e(item['llm_model'])})"
                      if item.get("llm_model") else "")
            if item["awarded"] is not None:
                quote = item.get("quote")
                if not quote and item.get("crop_reads"):
                    where = (f"drawing {item['crop'] + 1}"
                             if item.get("crop") is not None
                             else f"page {item.get('page')}")
                    quote = (f"[{where}, not quote-checked] "
                             f"{item['crop_reads']}")
                body.append(
                    "<div class='item settled'>"
                    f"<div class='pt'>{e(item['point'])}</div>"
                    f"<div class='meta'><b>{item['awarded']:g}/{avail_i:g}"
                    f"</b> &middot; {e(item['tier'])} tier{reader} &middot; "
                    f"{e(str(item.get('why') or ''))}</div>"
                    + (f"<div class='quote'>{e(quote)}</div>" if quote else "")
                    + mark_controls(field, avail_i, item["awarded"])
                    + "</div>")
            else:
                declined = item.get("llm_declined")
                note = (f"the model{reader} was refused here: "
                        f"{e(str(declined))}"
                        if declined else
                        f"undecided &middot; reached the {e(item['tier'])} "
                        "tier")
                body.append(
                    f"<div class='item'><div class='pt'>{e(item['point'])}"
                    f"</div><div class='meta'><b>{avail_i:g} marks</b></div>"
                    f"<div class='meta'>{note}</div>"
                    + mark_controls(field, avail_i, None) + "</div>")
        body.append("</div></div></div>")

    body.append("</main><div class='bar'><input type='text' name='note' "
                "placeholder='why (optional, recorded with every mark you "
                "change on this page)'>"
                "<button type='submit' name='go' value='stay' class='alt'>"
                "Save</button>"
                "<button type='submit' name='go' value='next'>"
                "Save &amp; next undecided</button>"
                f"<span>{open_items(report)} item(s) still undecided</span>"
                "</div></form>")

    return page(booklet, head + "".join(jump) + "".join(body))


FILTERS = [("all", "all booklets"), ("todo", "still undecided"),
           ("off", "outside the examiner's total"), ("done", "fully decided")]

STATUS_HTML = {
    "none": "<span class='empty'>no examiner total</span>",
    "agreed": "<span class='ok'>agreed exactly</span>",
    "reachable": "reachable",
    "above": "<span class='off'>above our range</span>",
    "below": "<span class='off'>below our range</span>",
}


def render_index(booklets, show="all"):
    e = html.escape
    show = show if show in dict(FILTERS) else "all"
    counts = {"done": 0, "agreed": 0, "off": 0, "items_left": 0,
              "items": 0, "settled": 0.0, "pending": 0.0}
    rows, first_open = [], None
    for index, entry in enumerate(booklets):
        report = read_report(entry)
        left = open_items(report)
        items = sum(len(q["items"]) for _, q in counted(report))
        state = standing(report, entry["examiner_total"])
        counts["items"] += items
        counts["items_left"] += left
        counts["settled"] += report["marks_settled"]
        counts["pending"] += report["marks_pending"]
        counts["done"] += not left
        counts["agreed"] += state == "agreed"
        counts["off"] += state in ("above", "below")
        if left and first_open is None:
            first_open = index

        if show == "todo" and not left:
            continue
        if show == "done" and left:
            continue
        if show == "off" and state not in ("above", "below"):
            continue

        ex = entry["examiner_total"]
        progress = 100 * (items - left) / items if items else 100
        rows.append(
            f"<tr{' class=done' if not left else ''}>"
            f"<td><a class='row' href='/b/{index}'>{e(entry['booklet'])}</a>"
            f"</td><td>{entry['cie']}</td>"
            f"<td>{report['marks_settled']:g}</td>"
            f"<td>{report['marks_pending']:g}</td>"
            f"<td>{'-' if ex is None else format(ex, 'g')}</td>"
            f"<td>{left or ''}</td><td>{progress:.0f}%</td>"
            f"<td>{STATUS_HTML[state]}</td></tr>")

    decided = counts["items"] - counts["items_left"]
    head = ("<header><b>Marking review</b>"
            f"<span>{len(booklets)} booklets</span><span class='sp'></span>"
            "<a href='/export/marks.csv'>download marks CSV</a>"
            "<a href='/export/questions.csv'>per-question CSV</a></header>"
            "<nav class='filters'>"
            + "".join(f"<a href='/?show={key}'"
                      + (" class='on'" if key == show else "")
                      + f">{label}</a>" for key, label in FILTERS)
            + "</nav>")

    stats = ("<div class='stats'>"
             f"<div class='stat'><b>{decided} / {counts['items']}</b>"
             "<span>rubric items decided</span></div>"
             f"<div class='stat'><b>{counts['items_left']}</b>"
             "<span>items waiting for you</span></div>"
             f"<div class='stat'><b>{counts['done']}</b>"
             "<span>booklets fully decided</span></div>"
             f"<div class='stat'><b>{counts['agreed']}</b>"
             "<span>agree with examiner exactly</span></div>"
             f"<div class='stat'><b>{counts['off']}</b>"
             "<span>outside examiner's total</span></div>"
             "</div>")
    start = (f"<p><a class='go' href='/b/{first_open}' data-key='n'>"
             "Continue reviewing &rarr;</a> <span class='keys'>or press "
             "<kbd>n</kbd></span></p>" if first_open is not None else
             "<p class='ok'>Every rubric item is decided.</p>")

    table = ("<table><tr><th>booklet</th><th>CIE</th><th>ours</th>"
             "<th>pending</th><th>examiner</th><th>undecided</th>"
             "<th>done</th><th>status</th></tr>" + "".join(rows)
             + "</table>" if rows else
             "<p class='empty'>Nothing matches this filter.</p>")
    return page("Marking review",
                head + "<main>" + stats + start + table + "</main>")


def export_csv(booklets, per_question):
    """The marks as they stand right now, for a spreadsheet.

    Built from the marks files on each request, so it includes every
    decision saved in this tool - unlike data/marking/summary.csv, which only
    changes when the pipeline is re-run.
    """

    out = io.StringIO()
    writer = csv.writer(out)
    if per_question:
        writer.writerow(["booklet_id", "cie", "question", "part",
                         "marks_available", "ours_settled", "pending",
                         "examiner", "undecided_items"])
    else:
        writer.writerow(["booklet_id", "cie", "ours_settled", "pending",
                         "max_marks", "examiner_total", "status",
                         "undecided_items"])
    for entry in booklets:
        report = read_report(entry)
        if per_question:
            for qindex, question in counted(report):
                examiner = entry["marks"][qindex]
                writer.writerow([
                    entry["booklet"], report["cie"], question["id"],
                    question["part"], question["marks_available"],
                    question["marks_settled"], question["marks_pending"],
                    "" if examiner is None else examiner,
                    sum(1 for i in question["items"]
                        if i["awarded"] is None)])
        else:
            rows = counted(report)
            writer.writerow([
                entry["booklet"], report["cie"], report["marks_settled"],
                report["marks_pending"],
                sum(q["marks_available"] for _, q in rows),
                "" if entry["examiner_total"] is None
                else entry["examiner_total"],
                standing(report, entry["examiner_total"]),
                open_items(report)])
    return out.getvalue().encode("utf-8")


# ------------------------------------------------------------- handler

class Handler(BaseHTTPRequestHandler):
    booklets = []

    def log_message(self, *args):
        pass                                   # the console stays readable

    def send(self, body, kind="text/html; charset=utf-8", code=200):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def redirect(self, where):
        self.send_response(303)
        self.send_header("Location", where)
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        route = url.path.strip("/").split("/")
        query = parse_qs(url.query)

        if route == [""]:
            show = (query.get("show") or ["all"])[0]
            return self.send(render_index(self.booklets, show))

        if route[0] == "b" and len(route) == 2 and route[1].isdigit():
            index = int(route[1])
            if not 0 <= index < len(self.booklets):
                return self.send(b"no such booklet", "text/plain", 404)
            entry = self.booklets[index]
            saved = (query.get("saved") or [None])[0]
            return self.send(render_booklet(
                index, len(self.booklets), entry, read_report(entry),
                next_open(self.booklets, index),
                int(saved) if saved and saved.isdigit() else None))

        if route[0] == "export" and len(route) == 2 and route[1] in (
                "marks.csv", "questions.csv"):
            body = export_csv(self.booklets, route[1] == "questions.csv")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition",
                             f"attachment; filename={route[1]}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route[0] == "crop" and len(route) == 4:
            return self.serve_crop(*route[1:])

        return self.send(b"not found", "text/plain", 404)

    def serve_crop(self, index, qindex, number):
        """A crop is addressed by position, never by name.

        All three are integers indexed into data we loaded ourselves, so
        there is no path in the URL for anyone to walk out of.
        """

        if not all(part.isdigit() for part in (index, qindex, number)):
            return self.send(b"bad crop id", "text/plain", 400)
        index, qindex, number = int(index), int(qindex), int(number)
        if not 0 <= index < len(self.booklets):
            return self.send(b"no such booklet", "text/plain", 404)
        entry = self.booklets[index]
        report = read_report(entry)
        if not 0 <= qindex < len(report["questions"]):
            return self.send(b"no such question", "text/plain", 404)
        crops = diagrams_for(entry["booklet"],
                             report["questions"][qindex]["id"])
        if not 0 <= number < len(crops):
            return self.send(b"no such crop", "text/plain", 404)
        path = Path(crops[number]["path"])
        if not path.exists():
            return self.send(b"crop missing from the handoff",
                             "text/plain", 404)
        self.send(path.read_bytes(), "image/png")

    def do_POST(self):
        route = urlparse(self.path).path.strip("/").split("/")
        if not (len(route) == 2 and route[0] == "decide"
                and route[1].isdigit()):
            return self.send(b"not found", "text/plain", 404)
        index = int(route[1])
        if not 0 <= index < len(self.booklets):
            return self.send(b"no such booklet", "text/plain", 404)

        length = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        entry = self.booklets[index]
        report = read_report(entry)

        decisions = {}
        for qindex, question in counted(report):
            for n in range(len(question["items"])):
                field = f"{qindex}_{n}"
                typed = (form.get(f"n{field}") or [""])[0].strip()
                chosen = (form.get(f"v{field}") or ["keep"])[0]
                value = typed if typed else (None if chosen == "keep"
                                             else chosen)
                if value is None:
                    continue
                try:
                    decisions[(qindex, n)] = float(value)
                except ValueError:
                    continue

        note = (form.get("note") or [""])[0].strip()
        applied = record(entry, decisions, note)
        if (form.get("go") or ["stay"])[0] == "next":
            following = next_open(self.booklets, index)
            if following is None:
                return self.redirect("/?show=all")
            return self.redirect(f"/b/{following}?saved={applied}")
        self.redirect(f"/b/{index}?saved={applied}")


def check(booklets):
    """Prove the data is really there before anyone opens a browser."""

    items = missing = withcrops = questions = todo = 0
    for entry in booklets:
        report = read_report(entry)
        left = open_items(report)
        items += left
        if left:
            todo += 1
        for _, question in counted(report):
            questions += 1
            crops = diagrams_for(entry["booklet"], question["id"])
            if crops:
                withcrops += 1
                missing += sum(1 for c in crops
                               if not Path(c["path"]).exists())
    print(f"{len(booklets)} booklets, {questions} counted questions")
    print(f"{todo} booklets still hold undecided items ({items} items)")
    print(f"{withcrops} questions have crops on disk")
    print(f"{missing} crop file(s) referenced but missing")
    if HUMAN_LOG.exists():
        marked = sum(1 for _ in open(HUMAN_LOG, encoding="utf-8"))
        print(f"{marked} human decision(s) recorded in {HUMAN_LOG}")
    return 1 if missing else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not paths.MARKS_DIR.exists():
        raise SystemExit("no marks yet - run: python marking/src/grade.py --all")

    booklets = build_booklets()
    if not booklets:
        raise SystemExit("no marks found in data/marking/marks/")

    if args.check:
        raise SystemExit(check(booklets))

    Handler.booklets = booklets
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    todo = sum(1 for e in booklets if open_items(read_report(e)))
    print(f"{len(booklets)} booklets, {todo} with undecided items")
    print(f"http://127.0.0.1:{args.port}/")
    print("ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
