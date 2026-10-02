# The handoff: what reading gives marking

The handoff is the one interface between the two halves. Reading writes
it (`reading/handoff/export.py`, run as `python pipeline.py handoff`).
Marking reads it (`marking/src/load_handoff.py`, the first step of
`python pipeline.py mark`). Everything on either side may be
restructured; this may not change without both sides agreeing.

Schema version **2.0**: the shape part 1 shipped to the marking half in
September 2026, kept so that handoff marks the same way as one built
here.

---

## 1. Layout

```
data/handoff/
  index.json                      every booklet, with totals
  student_07_cie_2/
    booklet.json                  questions -> parts -> answer[]
    pages/page_02.png ...         the prepared page images
    regions/s07_c2_p03_f01.png    every drawing crop an answer refers to
```

Every path inside a `booklet.json` is relative to that booklet's own
folder, so a booklet folder can be moved or shipped on its own.

Marking reads `data/handoff/` unless `MPE_HANDOFF` points elsewhere.
That can be the folder holding `index.json`, or an older handoff that
nests it under `data/` (the September `handoff1/`).

---

## 2. `index.json`

```json
{
  "schema_version": "2.0",
  "generated_by": "reading/handoff/export.py",
  "engine": "all_read",
  "booklets_total": 122,
  "totals": {"questions": 0, "parts": 0, "parts_labelled": 0, "items": 0,
             "diagrams": 0, "crossed": 0, "gaps": 0,
             "pages_read": 0, "pages_total": 0},
  "booklets": [{"booklet_id": "student_07_cie_2",
                "path": "student_07_cie_2/booklet.json", "...": "coverage"}]
}
```

`load_handoff.py --report` recomputes `booklets_total`, `parts`,
`parts_labelled` and `diagrams` from the booklets themselves and refuses
to let marking continue if any disagree. If the two halves read the
contract differently, every mark downstream inherits the difference, and
this check is the cheapest way to catch it.

`engine` names the reader. A handoff never mixes two: `export.py`
refuses booklets assembled from different reads.

---

## 3. `booklet.json`

```json
{
  "schema_version": "2.0",
  "booklet_id": "student_07_cie_2",
  "identity":  {"extracted": false, "fields": {"cie": {"value": 2}, "...": "..."}},
  "pages":     [{"page_id": "s07_c2_p02", "n": 2, "role": "answer",
                 "image": "pages/page_02.png"}],
  "questions": [{"number": "2", "parts": ["..."]}],
  "coverage":  {"parts_total": 6, "parts_labelled": 6, "gaps": 1, "...": "..."}
}
```

### `pages[]`: every page that exists here

| role | meaning |
|---|---|
| `cover` | page 1, present only if it was fetched with `--with-covers` |
| `answer` | something read on it was kept |
| `blank` | read, and nothing on it survived |
| `unread` | never transcribed. A `gap` item in the answers says so too |

### `questions[]` → `parts[]`

```json
{"number": "2", "parts": [
  {"label": "2a", "number": "2", "part": "a", "raw_marker": "2a)",
   "answer": ["... items ..."],
   "pages": ["s07_c2_p02", "s07_c2_p03"],
   "crosses_page_break": true,
   "text": "convenience join of the readable prose"}]}
```

- `label` is a question id on the paper: `1`, `2a`, `2b`, `2c`, `3a`,
  `3b`, `4a` or `4b`. It is `null` for writing that came before the
  first question number in a booklet when question 1 was labelled
  elsewhere. Marking resolves those with `align.py`, which never
  assigns one automatically.
- Parts appear in the order the student answered them, not sorted. A
  student who goes back to 2a gets one 2a part with both runs in page
  order.
- `text` is redundant by design. **The answer is the list.**

### `answer[]`: the items

| `type` | fields | marking does |
|---|---|---|
| `text` | `page_id`, `text`, `excluded: false`, `source` | marks it |
| `diagram` | `page_id`, `image`, `bbox`, `caption`, `flags` | routes evidence in it to a person, who sees `image` |
| `crossed` | `page_id`, `text`, `excluded: true` | skips it, and counts it so a cancelled answer reads differently from an empty one |
| `gap` | `page_id`, `reason` | treats the part as incomplete: no cheap tier may zero it |

A `diagram`'s `flags` may say `reading_failed` and `full_page`: the
reading of that page broke down partway, and the item is the whole page
so a person can mark from the image. `unreferenced` marks a drawn
region the reader never mentioned. That only happens when the diagram
pass has not run.

---

## 4. The three rules for anything that reads this

Each exists because ignoring it produces a wrong *mark* rather than an
error.

1. **`answer[]` is the source of truth, not `part.text`.** The join is a
   convenience whose construction you do not control.
2. **Skip anything with `excluded: true`.** Struck-out work is carried so
   a person can see it; scoring it awards marks the student withdrew. Use
   `item.get("excluded")`: a `gap` has no such key.
3. **A `gap` means incomplete, not short.** Content existed on a page
   nobody read; scoring the part as a thin answer penalises the student
   for our lost page.

---

## 5. Producing and checking it

```bash
python pipeline.py handoff                          # data/booklets -> data/handoff
python reading/handoff/export.py --check            # what it would write, nothing written
python marking/src/load_handoff.py --report         # marking reads it back
python marking/src/load_handoff.py --booklet student_07_cie_2 --verbose
```

`tests/test_pipeline.py` builds a synthetic booklet and checks every rule
above end to end.

### Differences from the September handoff

That handoff came from the other part 1 implementation (archived with the
old `main`). Marking reads both identically, because it depends only on
the fields above. Fields only that one carries, and that marking ignores:
a model-read `identity`, `geometry_source` on diagrams, and the page role
`unprocessed` where this one says `unread`.
