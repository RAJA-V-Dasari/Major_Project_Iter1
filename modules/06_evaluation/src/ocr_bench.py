"""
Score any OCR engine against hand-transcribed pages.

    06_evaluation/bench_pages.json      which pages, and why those
    06_evaluation/ground_truth/*.md     what is actually written on them
    06_evaluation/predictions/<engine>/*.md
        -> CER / WER per page, per difficulty bucket, and overall

This is `plan.md` section 6 step 2, and it exists to settle one
question with a number instead of an argument: is reading a whole page
better than reading it one segmented line at a time?

WHY CER, AND WHY NORMALISED FIRST
---------------------------------
Character error rate is edit distance over length, so it degrades
gracefully - a recogniser that gets four characters in five right
scores 0.20 rather than simply "wrong", which is what you need to
compare two engines that are both imperfect.

Everything is normalised before scoring: whitespace collapsed, case
folded, and the markdown scaffolding (table pipes, heading hashes,
emphasis) stripped. Otherwise an engine is punished for formatting
choices rather than for reading, and a page-level engine that emits a
real markdown table would score worse than a line engine that emits
the same characters as loose text - which would invert the very
comparison this is here to make.

DIAGRAMS ARE EXCLUDED FROM THE SCORE
------------------------------------
A diagram has no ground-truth text, so any transcription of it is
noise in a CER. Ground truth marks them with an image placeholder
line, and both sides drop those lines before scoring. What a diagram
IS good for is a separate count - did the engine notice something was
there - reported alongside but never mixed into the character score.

Run:
    python ocr_bench.py --engine trocr_lines
    python ocr_bench.py --engine vlm_page --verbose
    python ocr_bench.py --list            # what is transcribed so far
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

PAGES = STAGE_DIR / "bench_pages.json"
TRUTH_DIR = STAGE_DIR / "ground_truth"
PRED_DIR = STAGE_DIR / "predictions"

# A ground-truth line that stands in for a drawing rather than text.
DIAGRAM_LINE = re.compile(r"^\s*!\[.*?\]|^\s*<!--\s*(diagram|edges)", re.I)

BUCKETS = ["neat", "medium", "messy"]


def strip_markdown(text):
    """Formatting out, characters in. See the module note."""

    # fenced blocks keep their content, lose their fences
    text = re.sub(r"^```.*$", "", text, flags=re.M)

    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)      # headings
    text = re.sub(r"[*_~`]+", "", text)                      # emphasis
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)     # bullets
    text = re.sub(r"^\s*\|?[\s:|-]+\|?\s*$", "", text, flags=re.M)  # table rules
    text = text.replace("|", " ")                            # table cells

    return text


def normalise(text, drop_diagrams=True):
    """Comparable form: no markdown, no case, single spaces."""

    lines = []

    for line in text.splitlines():
        if drop_diagrams and DIAGRAM_LINE.search(line):
            continue
        lines.append(line)

    text = strip_markdown("\n".join(lines))

    # NFKC folds the lookalikes an engine may legitimately differ on -
    # a full-width digit, a non-breaking space - so they do not read as
    # substitutions
    text = unicodedata.normalize("NFKC", text)

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def edit_distance(a, b):
    """Levenshtein, two rows rather than a full matrix."""

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),   # substitution
            ))
        previous = current

    return previous[-1]


def rates(truth, predicted):
    """(CER, WER) of `predicted` against `truth`."""

    cer = (edit_distance(truth, predicted) / len(truth)) if truth else (
        0.0 if not predicted else 1.0)

    t_words, p_words = truth.split(), predicted.split()

    wer = (edit_distance(t_words, p_words) / len(t_words)) if t_words else (
        0.0 if not p_words else 1.0)

    return min(cer, 1.0), min(wer, 1.0)


def page_key(entry):
    return (f"s{entry['student']:02d}_c{entry['cie']}"
            f"_p{entry['page']:02d}")


def load_pages():

    if not PAGES.exists():
        raise SystemExit(f"{PAGES} not found")

    with open(PAGES, encoding="utf-8") as handle:
        return json.load(handle)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    entries = load_pages()

    if args.list:
        print(f"{'page':<14}{'bucket':<9}{'truth':<8}")
        done = 0
        for entry in entries:
            key = page_key(entry)
            has = (TRUTH_DIR / f"{key}.md").exists()
            done += has
            print(f"{key:<14}{entry['bucket']:<9}{'yes' if has else '-':<8}")
        print(f"\n{done}/{len(entries)} transcribed")
        return 0

    if not args.engine:
        raise SystemExit("--engine is required (or --list)")

    engine_dir = PRED_DIR / args.engine

    if not engine_dir.exists():
        raise SystemExit(f"{engine_dir} not found - write predictions there, "
                         f"one <page_id>.md per page")

    per_bucket = {b: [] for b in BUCKETS}
    scored = []
    missing = []

    for entry in entries:

        key = page_key(entry)

        truth_path = TRUTH_DIR / f"{key}.md"
        pred_path = engine_dir / f"{key}.md"

        if not truth_path.exists() or not pred_path.exists():
            missing.append(key)
            continue

        truth = normalise(truth_path.read_text(encoding="utf-8"))
        predicted = normalise(pred_path.read_text(encoding="utf-8"))

        cer, wer = rates(truth, predicted)

        scored.append((key, entry["bucket"], cer, wer, len(truth)))
        per_bucket[entry["bucket"]].append((cer, wer))

    if not scored:
        raise SystemExit("nothing scored - no page has both truth and a "
                         "prediction")

    print(f"Engine : {args.engine}")
    print(f"Pages  : {len(scored)} scored"
          + (f", {len(missing)} missing" if missing else ""))
    print()

    if args.verbose:
        print(f"{'page':<14}{'bucket':<9}{'chars':>7}{'CER':>8}{'WER':>8}")
        for key, bucket, cer, wer, n in scored:
            print(f"{key:<14}{bucket:<9}{n:>7}{cer:>8.3f}{wer:>8.3f}")
        print()

    print(f"{'bucket':<9}{'pages':>6}{'CER':>9}{'WER':>9}")
    for bucket in BUCKETS:
        rows = per_bucket[bucket]
        if not rows:
            continue
        cer = sum(r[0] for r in rows) / len(rows)
        wer = sum(r[1] for r in rows) / len(rows)
        print(f"{bucket:<9}{len(rows):>6}{cer:>9.3f}{wer:>9.3f}")

    # character-weighted, so a long page counts for more than a sparse
    # one - the corpus-level number rather than the per-page average
    total_chars = sum(r[4] for r in scored)
    weighted = sum(r[2] * r[4] for r in scored) / total_chars

    overall_cer = sum(r[2] for r in scored) / len(scored)
    overall_wer = sum(r[3] for r in scored) / len(scored)

    print(f"{'OVERALL':<9}{len(scored):>6}{overall_cer:>9.3f}{overall_wer:>9.3f}")
    print(f"\nCharacter-weighted CER : {weighted:.3f}")

    if missing:
        print(f"\nNot scored ({len(missing)}): {' '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
