"""Compare two runs of the same pages, case by case.

A prompt change is only worth keeping if the pages it targeted got
better and the pages that already worked did not get worse. Aggregate
counts hide both halves: a run can fix nine loops, break two controls,
and still look like an improvement.

Reads `upload/TEST_BATCH.csv` for what each page is supposed to
demonstrate, then reports the signals that decide it.

    python modules/02_read/src/compare_runs.py \\
        --before modules/02_read/output/batch00 \\
        --after  modules/02_read/output/batchtest
"""
import argparse, collections, csv, io, pathlib, re, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

MARK_NEW = re.compile(r"!\[(diagram|table):\s*[^\]]*\]")
MARK_OLD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
CASES = pathlib.Path("modules/02_read/upload/TEST_BATCH.csv")


def signals(body):
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    top = collections.Counter(
        l for l in lines if l not in {"```", "|", "---"}).most_common(1)
    return {
        "chars": len(body),
        "repeat": top[0][1] if top else 0,
        "marks": len(MARK_NEW.findall(body)),
        "oldbox": len(MARK_OLD.findall(body)),
        "q": body.count("[?]"),
        "rows": sum(1 for l in lines if l.startswith("|")),
    }


def verdict(before, after):
    """What changed, in the terms the test batch was built to check."""
    notes = []
    if before["repeat"] >= 25 and after["repeat"] < 25:
        notes.append("loop FIXED")
    if before["repeat"] < 25 and after["repeat"] >= 25:
        notes.append("loop INTRODUCED")
    if before["oldbox"] and not after["oldbox"] and not after["marks"]:
        notes.append("figure LOST")
    if after["marks"] and not before["marks"]:
        notes.append("marker gained")
    if before["rows"] and not after["rows"]:
        notes.append("table dropped")
    if after["rows"] > max(6, before["rows"] * 2):
        notes.append("tables INFLATED")
    if after["chars"] < before["chars"] * 0.25:
        notes.append("content LOST")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=pathlib.Path, required=True)
    ap.add_argument("--after", type=pathlib.Path, required=True)
    args = ap.parse_args()

    cases = {}
    if CASES.exists():
        with open(CASES, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                cases[row["page_id"]] = (row["was"], row["must_now"])

    rows = []
    for f in sorted(args.after.glob("*.md")):
        old = args.before / f.name
        if not old.exists():
            continue
        b = signals(old.read_text(encoding="utf-8"))
        a = signals(f.read_text(encoding="utf-8"))
        was, must = cases.get(f.stem, ("", ""))
        rows.append((f.stem, b, a, verdict(b, a),
                     was.startswith("CONTROL"), must))

    ctrl = [r for r in rows if r[4]]
    fail = [r for r in rows if not r[4]]

    for title, group in (("CONTROLS - these were right already", ctrl),
                         ("TARGETS - these were broken", fail)):
        print(f"\n=== {title} ===")
        print(f"{'page':<13} {'chars':>13} {'rep':>9} {'mark':>6} "
              f"{'[?]':>5} {'rows':>9}   notes")
        for stem, b, a, notes, _, must in sorted(group):
            print(f"{stem:<13} {b['chars']:>6}->{a['chars']:<6} "
                  f"{b['repeat']:>4}->{a['repeat']:<4} "
                  f"{b['marks'] + b['oldbox']:>2}->{a['marks']:<3} "
                  f"{b['q']:>2}->{a['q']:<2} "
                  f"{b['rows']:>4}->{a['rows']:<4}   {', '.join(notes)}")

    good = sum(1 for r in rows if "loop FIXED" in r[3])
    bad = sum(1 for r in rows
              if any(n.isupper() or n.split()[-1].isupper() for n in r[3])
              and "loop FIXED" not in r[3])

    print(f"\nloops fixed: {good}")
    print(f"pages with a regression: {bad}")
    print(f"[?] total: {sum(r[1]['q'] for r in rows)} -> "
          f"{sum(r[2]['q'] for r in rows)}")
    print(f"table rows total: {sum(r[1]['rows'] for r in rows)} -> "
          f"{sum(r[2]['rows'] for r in rows)}")


if __name__ == "__main__":
    main()
