r"""
Where everything lives.

Every marking script imports this module and nothing else for a path, so
this is the one place the marking half meets the repository's data
layout (common/layout.py):

    marking/keys/           the rubric, hand-authored          (tracked)
    marking/gold/           the examiner's marks, as integers  (tracked)
    data/handoff/           part 1's booklets - what is marked
    data/marking/           everything this half writes
    data/schemes/           the department's scheme PDFs
    data/schemes/pages/     their renders, for reading the rubric off

Nothing under data/ is tracked: the handoff is students' answers, every
output quotes them, and the schemes are the department's material.

To mark a handoff that lives elsewhere - the one part 1 shipped in
September as handoff1/, say - point MPE_HANDOFF at it:

    set MPE_HANDOFF=D:\somewhere\handoff1
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(next(p for p in ROOT.parents
                            if (p / "common" / "layout.py").exists())))

from common import layout                                  # noqa: E402

KEYS_DIR = ROOT / "keys"
GOLD_DIR = ROOT / "gold"
GOLD_CSV = GOLD_DIR / "gold_marks.csv"

OUT_DIR = layout.MARKING
MARKS_DIR = OUT_DIR / "marks"
COVER_CROPS = OUT_DIR / "covers"

HANDOFF_DATA = layout.handoff_data()
HANDOFF = HANDOFF_DATA
HANDOFF_INDEX = HANDOFF_DATA / "index.json"

SCHEME_PDFS = layout.SCHEMES
SCHEME_PAGES = layout.SCHEME_PAGES

# The scheme PDFs are named by hand, not by convention.
SCHEME_FILES = {
    1: "CIE 1 scheme CN.pdf",
    2: "CIE 2 Scheme CN.pdf",
    3: "CIE 3 Scheme CN.pdf",
}


def require(path, what):
    """Fail early and say which path, rather than a bare FileNotFoundError."""

    if not path.exists():
        raise SystemExit(
            f"{what} not found at:\n    {path}\n"
            "Build it with: python pipeline.py handoff - or set MPE_HANDOFF "
            "(an existing handoff) / MPE_SCHEMES (the scheme PDFs)."
        )
    return path
