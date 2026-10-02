"""Health-check a batch of pages read by the notebook.

Run this on every batch before it is assembled. It answers what the
per-page log cannot:

  - did a page fall into a repetition loop and get cut off?
  - is the reader willing to decline (`[?]`, `![table]`), or does it
    invent instead?
  - is every figure box usable for cropping, or is it the template
    text, a made-up label, or an invented URL?
  - will assembly group correctly, or has an enumerated list inside an
    answer been promoted to question headings?
  - did prompt text leak into a transcription?

A looped page is the one that hides. Its output is non-empty, so the
notebook's resume rule counts it as done and never revisits it; without
this check it enters the corpus silently truncated.

    python reading/read/qa/check_batch.py data/read/qwen7b
"""
import argparse, collections, io, pathlib, re, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

# The only two marker forms. Coordinates were dropped after they turned
# out to be round numbers in the model's own resized space that mostly
# missed the figure; position in the reading order locates it now, and
# segment.py supplies the pixels. A tuple or a URL is therefore no
# longer a badly-aimed box, it is a marker the pipeline cannot place.
BOX_OK = re.compile(r"!\[(diagram|table):\s*([^\]]*)\]$")
BOX_ANY = re.compile(r"!\[[^\]]*\](?:\([^)]*\))?")

HEADING = re.compile(r"^(#{1,6})\s*(.+?)\s*$", re.M)
NUMBER_ONLY = re.compile(r"^([0-9]{1,2})\)?$")
FENCE = re.compile(r"^```(\w*)", re.M)

# The schema these booklets are set from.
SCHEMA = {"1", "2a", "2b", "2c", "3a", "3b", "4a", "4b"}

# Long, distinctive prompt phrases only. Short ones ("TABLES") and the
# prompt's worked examples ("a b c a b") collide with real answers -
# these booklets are full of node labels - and raise only false alarms.
LEAKS = ["one rule that matters", "not answering this exam",
         "copied faithfully", "a serious error", "recognition is a trap"]

LOOP_RUN = 25           # a line repeated this often is a loop outright
LOOP_RUN_LONG = 10      # ...fewer repeats still count on a long page
LOOP_CHARS = 1800
LOOP_LINE = 600         # a single line this long is repetition within it

# Fences and table rules legitimately repeat, so they are never the
# evidence for a loop.
NOT_EVIDENCE = {"```", "|", "---"}


def loop_evidence(body):
    """Return (count, line) if this page looks looped, else (0, '')."""
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    if not lines:
        return 0, ""

    longest = max(lines, key=len)
    if len(longest) >= LOOP_LINE:
        return 1, longest

    for line, n in collections.Counter(lines).most_common():
        if line in NOT_EVIDENCE:
            continue
        if n >= LOOP_RUN or (n >= LOOP_RUN_LONG and len(body) >= LOOP_CHARS):
            return n, line
        break
    return 0, ""


def list_as_headings(body):
    """Bare-number headings forming an ascending run.

    `#### 0 / 1 / 2 / 3 / 4` is a numbered list inside one answer, not
    five questions. Assembly groups on headings, so left alone this
    splits one answer into fragments filed under invented question
    numbers.
    """
    nums = []
    for _, text in HEADING.findall(body):
        m = NUMBER_ONLY.match(re.sub(r"\s+", "", text))
        if m:
            nums.append(int(m.group(1)))

    best = run = 1 if nums else 0
    for a, b in zip(nums, nums[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return nums if best >= 3 else []


def fence_shape(body):
    """'wrapper', 'content', or '' - what the fences on a page are.

    A whole-page ```markdown wrapper is scaffolding to strip. A fence
    around part of a page is the student's own block content and must
    survive: at least one page here holds a Dijkstra working table that
    way, and blind stripping would delete it.
    """
    body = body.strip()
    kinds = FENCE.findall(body)
    if not kinds:
        return ""
    if len(kinds) == 2 and body.startswith("```") and body.endswith("```"):
        return "wrapper"
    return "content"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=pathlib.Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = sorted(args.batch.glob("*.md"))
    if not files:
        sys.exit(f"no .md files under {args.batch}")

    looped, clean, empty = [], [], []
    for f in files:
        body = f.read_text(encoding="utf-8")
        if not body.strip():
            empty.append(f.stem)
            continue
        n, line = loop_evidence(body)
        if n:
            looped.append((f.stem, len(body), n, line))
        else:
            clean.append((f.stem, body))

    print(f"{args.batch.name}: {len(files)} pages, {len(empty)} empty, "
          f"{len(looped)} looped, {len(clean)} clean")
    if empty:
        print("  EMPTY (retried on resume):", empty)

    if looped:
        print(f"\nLOOPED - cut off at the ceiling, re-read these "
              f"({len(looped)}):")
        for stem, size, n, line in sorted(looped, key=lambda r: -r[2]):
            print(f"   {stem:<13} {size:6d} chars  {n:4d}x  {line[:44]!r}")

    marks = collections.Counter()
    labels = collections.Counter()
    leaked, listy = [], []
    malformed = {}
    fences = collections.Counter()

    for stem, body in clean:
        marks["[?]"] += body.count("[?]")
        marks["pages with [?]"] += 1 if "[?]" in body else 0
        marks["table rows"] += sum(1 for l in body.splitlines()
                                   if l.strip().startswith("|"))

        bad = []
        for hit in BOX_ANY.findall(body):
            m = BOX_OK.match(hit)
            if m:
                marks[f"![{m.group(1)}] usable"] += 1
            else:
                bad.append(hit)
        if bad:
            malformed[stem] = bad

        fences[fence_shape(body)] += 1

        run = list_as_headings(body)
        if run:
            listy.append((stem, run))

        for _, text in HEADING.findall(body):
            key = re.sub(r"\s+", "", text).rstrip(")").lower()
            if re.fullmatch(r"[0-9]+[a-d]?", key):
                labels[key] += 1

        if any(p in body.lower() for p in LEAKS):
            leaked.append(stem)

    print(f"\nOn the {len(clean)} clean pages:")
    for k in ("[?]", "pages with [?]", "![diagram] usable",
              "![table] usable", "table rows"):
        print(f"   {k:<22} {marks[k]}")
    print(f"   {'``` whole-page wrap':<22} {fences['wrapper']}  (strip)")
    print(f"   {'``` around content':<22} {fences['content']}  "
          f"(KEEP - student's own block)")

    if malformed:
        n = sum(len(v) for v in malformed.values())
        print(f"\nUNUSABLE FIGURE BOXES - {n} on {len(malformed)} pages. "
              f"Each is a figure that cannot be cropped:")
        for stem, v in sorted(malformed.items(), key=lambda x: -len(x[1])):
            print(f"   {stem:<13} {len(v):3d}  {v[0][:50]}")

    if listy:
        print(f"\nENUMERATED LIST AS HEADINGS - {len(listy)} pages. Assembly "
              f"would split one answer into several questions:")
        for stem, run in listy:
            print(f"   {stem:<13} {run}")

    off = {k: v for k, v in labels.items() if k not in SCHEMA}
    if off:
        print(f"\n   off-schema labels: {sum(off.values())}  {off}")
    if leaked:
        print(f"   PROMPT LEAKED INTO OUTPUT: {leaked}")

    if args.verbose:
        print("\nlabels seen:", dict(sorted(labels.items())))

    if not marks["![table] usable"]:
        print("\n  ! zero ![table] boxes. A table the reader cannot read is "
              "rendered empty rather than declined - the same fault as the "
              "loops above, not a separate one.")


if __name__ == "__main__":
    main()
