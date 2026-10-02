"""Audit every page in a batch against the image it was read from.

`check_batch.py` reads only the Markdown, so it can see that a page is
malformed but not that a page is *missing* content. This measures the
ink on the source image and compares it with the transcription, which
is what catches the quiet failure: a page covered in handwriting that
came back as three lines.

Ink per line is roughly constant within a corpus of the same paper and
pen, so characters-per-ink is stable across pages that were read fully.
A page far below the median gave back less than it should have; far
above and the reader was generating text the page does not support.

Writes a TSV of every page plus a summary of the outliers.

    python reading/read/qa/audit_pages.py data/read/qwen7b
"""
import argparse, collections, io, pathlib, re, statistics, sys

import cv2
import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents
                            if (p / "common" / "layout.py").exists())))
from common import layout                                  # noqa: E402

PAGES = layout.PAGES
STEM = re.compile(r"^s(\d+)_c(\d+)_p(\d+)$")

INK_LEVEL = 160         # cleaned pages are near-white; this is generous
RULE_LEN = 90           # a horizontal run this long is printed ruling
MIN_RUN = 6             # rows of ink this tall count as a line of text


def page_path(stem):
    m = STEM.match(stem)
    if not m:
        return None
    s, c, p = m.groups()
    return PAGES / f"student_{s}" / f"cie_{c}" / f"page_{p}.png"


def ink_stats(path):
    """Ink fraction and an estimated line count for one page.

    The paper is ruled, and the printed rules are ink too - on a nearly
    empty page they are most of it. Measuring them made a page holding
    one small routing table look densely written and its correct,
    complete transcription look like content loss. So the rules are
    removed first: anything surviving a long horizontal opening is
    printed ruling, not handwriting.
    """
    a = np.asarray(Image.open(path).convert("L"))
    dark = (a < INK_LEVEL).astype(np.uint8)

    rules = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (RULE_LEN, 1)))
    # dilate slightly so the rule's own thickness goes with it
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))
    dark = (dark & ~rules).astype(bool)

    ink = float(dark.mean())

    rows = dark.sum(axis=1)
    # A text line is a run of rows carrying more ink than the page's
    # own noise floor, so a faint scan is not read as a blank page.
    floor = max(3, rows.max() * 0.04)
    lines, run = 0, 0
    for r in rows:
        if r > floor:
            run += 1
        else:
            if run >= MIN_RUN:
                lines += 1
            run = 0
    if run >= MIN_RUN:
        lines += 1
    return ink, lines


def text_len(body):
    """Characters of actual transcription, ignoring scaffolding."""
    body = re.sub(r"^```\w*$", "", body, flags=re.M)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)
    body = re.sub(r"^#{1,6}\s*", "", body, flags=re.M)
    body = re.sub(r"[|\-\s]", "", body)
    return len(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=pathlib.Path)
    ap.add_argument("--tsv", type=pathlib.Path)
    args = ap.parse_args()

    files = sorted(args.batch.glob("*.md"))
    if not files:
        sys.exit(f"no .md under {args.batch}")

    rows, missing = [], []
    for f in files:
        img = page_path(f.stem)
        if img is None or not img.exists():
            missing.append(f.stem)
            continue
        body = f.read_text(encoding="utf-8")
        ink, lines = ink_stats(img)
        rows.append({
            "stem": f.stem,
            "ink": ink,
            "lines": lines,
            "chars": len(body),
            "text": text_len(body),
            # characters recovered per unit of ink - flat across pages
            # that were read fully, whatever the handwriting size
            "yield": text_len(body) / ink if ink > 1e-6 else 0.0,
        })

    if missing:
        print(f"NO SOURCE IMAGE for {len(missing)}: {missing[:8]}")

    ys = sorted(r["yield"] for r in rows)
    med = statistics.median(ys)
    lo, hi = med * 0.45, med * 1.9

    for r in rows:
        r["verdict"] = ("THIN" if r["yield"] < lo else
                        "DENSE" if r["yield"] > hi else "ok")

    tsv = args.tsv or args.batch.parent / f"{args.batch.name}_audit.tsv"
    with open(tsv, "w", encoding="utf-8", newline="") as fh:
        fh.write("stem\tink\tlines\tchars\ttext\tyield\tverdict\n")
        for r in sorted(rows, key=lambda x: x["yield"]):
            fh.write(f"{r['stem']}\t{r['ink']:.4f}\t{r['lines']}\t"
                     f"{r['chars']}\t{r['text']}\t{r['yield']:.0f}\t"
                     f"{r['verdict']}\n")

    counts = collections.Counter(r["verdict"] for r in rows)
    print(f"\n{args.batch.name}: {len(rows)} pages measured against source")
    print(f"  median yield {med:.0f} chars per unit ink   "
          f"(ok band {lo:.0f}-{hi:.0f})")
    for k in ("ok", "THIN", "DENSE"):
        print(f"  {k:<6} {counts[k]}")
    print(f"  written: {tsv}")

    for kind in ("THIN", "DENSE"):
        bad = [r for r in rows if r["verdict"] == kind]
        if not bad:
            continue
        print(f"\n{kind} - {len(bad)} pages "
              f"({'less text than the ink supports' if kind == 'THIN' else 'more text than the ink supports'}):")
        bad.sort(key=lambda r: r["yield"], reverse=(kind == "DENSE"))
        for r in bad:
            print(f"   {r['stem']:<13} ink {r['ink']*100:5.2f}%  "
                  f"{r['lines']:3d} lines  {r['text']:5d} chars  "
                  f"yield {r['yield']:6.0f}")


if __name__ == "__main__":
    main()
