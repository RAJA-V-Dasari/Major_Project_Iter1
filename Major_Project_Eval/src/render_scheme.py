"""
Render the three answer schemes to PNG, one file per page.

    answer_keys/CIE {1,2,3} *.pdf  ->  keys/scheme_pages/cie<N>_p<NN>.png

WHY THIS EXISTS AT ALL
----------------------
The schemes are phone scans. Probed with a byte scan, all three PDFs
contain image XObjects and *zero* font objects - there is no text layer
to extract, and no amount of pdftotext will produce one. The rubric has
to be read off the pixels by a person, so the first job is to get the
pixels out at a size a person can actually read.

200 dpi is the default because the marks column on the right edge
("5X1", "2+3", "CO1-PO1") is small, handwritten-adjacent print on a
skewed scan, and it is the part we most need to read correctly - it is
where a question's mark breakdown comes from. At 150 dpi it is legible
but uncomfortable; at 300 the files roughly double for no gain in what
a reader can resolve.

No deskewing, no thresholding, no cleanup. This is a faithful render
and nothing else: if a page is crooked, it is crooked in the PDF, and
straightening it here would mean the thing a human verifies against no
longer matches the document of record.

Run:
    python src/render_scheme.py
    python src/render_scheme.py --dpi 300 --cie 2
    python src/render_scheme.py --check
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths

DEFAULT_DPI = 200


def render(cie, dpi, out_dir):
    """Render one scheme. Returns the list of paths written."""

    import pymupdf

    pdf = paths.require(
        paths.SCHEME_PDFS / paths.SCHEME_FILES[cie], f"CIE {cie} scheme"
    )

    written = []
    with pymupdf.open(pdf) as doc:
        for number, page in enumerate(doc, start=1):
            pixmap = page.get_pixmap(dpi=dpi)
            out = out_dir / f"cie{cie}_p{number:02d}.png"
            pixmap.save(out)
            written.append(out)

    return written


def check(out_dir):
    """Report what has been rendered, and whether anything looks wrong."""

    import pymupdf

    ok = True
    for cie in sorted(paths.SCHEME_FILES):
        pdf = paths.SCHEME_PDFS / paths.SCHEME_FILES[cie]
        if not pdf.exists():
            print(f"cie{cie}: MISSING pdf {pdf}")
            ok = False
            continue

        with pymupdf.open(pdf) as doc:
            expected = doc.page_count
            # Worth stating out loud, because it is the whole reason the
            # rubric is hand-authored rather than parsed.
            fonts = sum(len(doc.get_page_fonts(n)) for n in range(expected))

        found = sorted(out_dir.glob(f"cie{cie}_p*.png"))
        smallest = min((p.stat().st_size for p in found), default=0)

        status = "ok" if len(found) == expected else "INCOMPLETE"
        if len(found) != expected:
            ok = False

        print(
            f"cie{cie}: {len(found)}/{expected} pages  "
            f"fonts={fonts}  smallest={smallest / 1024:.0f}KB  {status}"
        )

        # A page that renders to almost nothing rendered blank.
        for page in found:
            if page.stat().st_size < 20_000:
                print(f"    suspiciously small: {page.name}")
                ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--cie", type=int, choices=[1, 2, 3],
                        help="render only this one")
    parser.add_argument("--check", action="store_true",
                        help="report what is on disk, render nothing")
    args = parser.parse_args()

    out_dir = paths.SCHEME_PAGES
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        raise SystemExit(0 if check(out_dir) else 1)

    targets = [args.cie] if args.cie else sorted(paths.SCHEME_FILES)
    total = 0
    for cie in targets:
        written = render(cie, args.dpi, out_dir)
        total += len(written)
        print(f"cie{cie}: {len(written)} pages at {args.dpi} dpi")

    print(f"\n{total} pages -> {out_dir}")
    check(out_dir)


if __name__ == "__main__":
    main()
