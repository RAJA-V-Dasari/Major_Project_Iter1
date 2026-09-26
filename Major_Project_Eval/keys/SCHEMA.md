# The rubric schema

One file per CIE: `cie1.json`, `cie2.json`, `cie3.json`. Hand-authored
by reading `scheme_pages/`, because the scheme PDFs carry **zero font
objects** — they are phone scans of a printed document and there is no
text layer to parse. `render_scheme.py --check` prints that font count;
it is the evidence behind doing this by hand.

Every question records the `scheme_page` it came from, so any claim in
here can be checked against the scan in about ten seconds.

## Shape

```jsonc
{
  "cie": 1,
  "course_code": "25CS5PCCON",
  "max_marks": 40,
  "source": "answer_keys/CIE 1 scheme CN.pdf",
  "parts": { "A": 5, "B": 15, "C": 20 },   // printed on the scheme
  "questions": [
    {
      "id": "2c",                   // matches the handoff's part labels
      "part": "B",
      "marks": 5,
      "choice_with": null,          // "2b" etc. on the Part-C pairs
      "scheme_page": "cie1_p03",
      "question": "...",            // as printed
      "solution": "...",            // the model answer, as printed
      "breakdown_source": "printed",
      "breakdown_text": "1 x 5 = 5 marks",
      "rubric": [ ... ],
      "chain": null,
      "notes": []
    }
  ]
}
```

## `rubric` items

```jsonc
{ "marks": 1.0,
  "point": "Resource identifiers: each document has a URL",
  "exact": ["/docs/page1.html"],
  "keywords": [["url", "uniform resource"], ["resource identifier"]] }
```

- **`marks`** — what this item is worth. The items must sum to the
  question's `marks`. `validate_keys.py` enforces it.
- **`point`** — the claim the student has to have made, in one sentence.
  This is what tier 2 embeds and what the tier-3 prompt asks about.
- **`exact`** — literal values that are right or wrong with no
  interpretation: `57088`, `0045`, `/26`, a CRC remainder. Tier 1
  decides on these alone and no model is consulted.
- **`keywords`** — groups of alternates. A group is satisfied if **any**
  alternate appears; coverage is the fraction of groups satisfied. So
  `[["ack","acknowledg"],["8000"]]` needs one of the ack spellings *and*
  the number.

### The rule for `exact`

**Only distinctive values go in `exact`.** `"57088"` is safe. `"2"`,
`"88"` and `"ok"` are not — they occur by accident in prose about
anything, and a tier that awards a mark because the digit 2 appeared
somewhere is worse than no tier. Values are matched case-insensitively
on word boundaries, so `8001` never matches inside `18001`.

If a correct answer has no distinctive literal form, leave `exact` empty
and let it go up the ladder. That is the ladder working, not a gap.

**Tier 1 requires the literal *and* the keywords, not the literal
alone.** This is the fix for the values that cannot be made distinctive.
CIE-1 4b has to accept `88` and `80` — they are the actual answers — and
on their own they would fire on any student who mentioned port 80. So
the tier awards only when every `exact` value is present *and* the
item's keyword groups are satisfied: `88` counts when it arrives next to
"length", not when it arrives next to anything at all.

`validate_keys.py` reports every short numeric `exact` as a warning, so
each one is a value a person decided to trust rather than one that
slipped through.

## `breakdown_source`

- `"printed"` — the scheme states the split (`1 x 5`, `2.5 x2`,
  `[ 4 marks]`, `6+4`). Copy it into `breakdown_text` verbatim.
- `"inferred"` — the scheme gives a total and a solution but no split,
  so the division into items is **our judgement, not the department's**.
  Every inferred question is a place where our marks can differ from the
  examiner's for a reason that is nobody's error. `agreement.py` reports
  printed and inferred questions separately for exactly this reason.

## `chain`

Present only where the marks hang off a sequence of steps, so that one
early slip would otherwise cost every later item:

```jsonc
"chain": { "steps": ["...", "..."],
           "carry_forward": "A wrong block size costs its own step; the
                             later subnets are then marked against the
                             student's own value." }
```

The examiner plainly awards partial credit on these — the covers show
7s and 8s on ten-mark subnetting questions. A grader that zeroes the
whole chain would disagree with them for a reason that has nothing to do
with whether the student understood the method.

## `notes`

Free text, for anything a later reader would otherwise rediscover the
hard way — including defects in the scheme itself. These are recorded,
never silently corrected. The scheme is the document of record even
where it is wrong, and a marker needs to know where it is wrong.
