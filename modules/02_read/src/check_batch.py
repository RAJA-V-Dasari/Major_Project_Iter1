"""Health-check a batch of pages read by the notebook.

Run this on every batch before it is assembled. It answers four
questions the per-page log cannot:

  - did any page fall into a repetition loop and get cut off?
  - is the reader willing to decline (`[?]`, `![table]`) or does it
    invent instead?
  - did any prompt text leak into a transcription?
  - are there enough question headings for assembly to group on?

A looped page is the important one. Its output is non-empty, so the
notebook's resume rule counts it as done and never revisits it; without
this check it enters the corpus silently truncated.

    python modules/02_read/src/check_batch.py modules/02_read/output/batch00
"""
import argparse, collections, io, pathlib, re, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

BOXED = re.compile(r"!\[(diagram|table)\]\(\s*\d")
BARE = re.compile(r"!\[(?:diagram|table)?\]\((?:x1|\s*\))")
QHEAD = re.compile(r"^#{1,6}\s*([0-9]+\s*[a-d]?)\s*\)?", re.M)

# The schema these booklets are set from. Anything else is a misread.
SCHEMA = {"1", "2a", "2b", "2c", "3a", "3b", "4a", "4b"}

# Phrases that exist only in the prompt. Finding one in a transcription
# means the model copied its instructions instead of the page.
LEAKS = ["ONE RULE THAT MATTERS", "not answering this exam",
         "copied faithfully", "serious error"]

# A loop shows up two ways: the same LINE over and over, or one line
# that never ends because the repetition is inside it. Both are caught,
# because either one means the page was cut off at the ceiling.
LOOP_RUN = 25          # a line repeated this often is a loop outright
LOOP_RUN_LONG = 10     # ...fewer repeats still count on a long page
LOOP_CHARS = 1800
LOOP_LINE = 600        # a single line this long is repetition within it

# Fences legitimately repeat - a page with several code blocks is not a
# loop - so they are never the evidence for one.
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
        print("  EMPTY (will be retried on resume):", empty)

    if looped:
        print(f"\nLOOPED - cut off at the token ceiling, re-read these "
              f"({len(looped)}):")
        for stem, size, n, line in sorted(looped, key=lambda r: -r[2]):
            print(f"   {stem:<13} {size:6d} chars   {n:4d}x  {line[:44]!r}")

    marks = collections.Counter()
    labels = collections.Counter()
    leaked, bare, fenced, headed = [], [], 0, 0
    for stem, body in clean:
        marks["[?]"] += body.count("[?]")
        marks["pages with [?]"] += 1 if "[?]" in body else 0
        for kind in BOXED.findall(body):
            marks[f"![{kind}] boxed"] += 1
        marks["table rows"] += sum(1 for l in body.splitlines()
                                   if l.strip().startswith("|"))
        if BARE.search(body):
            bare.append(stem)
        if "```" in body:
            fenced += 1
        found = QHEAD.findall(body)
        if found:
            headed += 1
            for x in found:
                labels[re.sub(r"\s+", "", x)] += 1
        if any(p.lower() in body.lower() for p in LEAKS):
            leaked.append(stem)

    print(f"\nOn the {len(clean)} clean pages:")
    for k in ("[?]", "pages with [?]", "![diagram] boxed", "![table] boxed",
              "table rows"):
        print(f"   {k:<18} {marks[k]}")
    print(f"   {'``` fenced':<18} {fenced}")
    print(f"   {'question headings':<18} {headed} pages "
          f"({len(clean) - headed} are continuations)")

    off = {k: v for k, v in labels.items() if k not in SCHEMA}
    if off:
        print(f"   {'off-schema labels':<18} {sum(off.values())}  {off}")
    if leaked:
        print(f"   {'PROMPT LEAKED':<18} {leaked}")
    if bare:
        print(f"   {'placeholder, no box':<18} {bare}")

    if args.verbose:
        print("\nlabels seen:", dict(sorted(labels.items())))

    # The reader declining is what keeps a wrong answer from looking
    # like a right one, so it is called out rather than left to be
    # noticed in the numbers above.
    if not marks["![table] boxed"]:
        print("\n  ! zero ![table] boxes. A table the reader cannot read is "
              "being rendered empty rather than declined - that is the same "
              "fault as the loops above, not a separate one.")


if __name__ == "__main__":
    main()
