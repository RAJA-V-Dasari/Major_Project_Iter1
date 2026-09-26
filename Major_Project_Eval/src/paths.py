r"""
Where everything lives.

The handoff is NOT copied into this project. It is 187 MB of real
students' scripts, and a second copy is a second thing to keep
gitignored, a second thing to leak, and 187 MB that can drift from the
original. It is referenced in place instead, and overridable so the
corpus can move without touching any other file:

    set MPE_HANDOFF=D:\somewhere\handoff1
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KEYS_DIR = ROOT / "keys"
SCHEME_PAGES = KEYS_DIR / "scheme_pages"
GOLD_DIR = ROOT / "gold"
COVER_CROPS = GOLD_DIR / "covers"
GOLD_CSV = GOLD_DIR / "gold_marks.csv"
OUT_DIR = ROOT / "output"
MARKS_DIR = OUT_DIR / "marks"

def _first_existing(candidates, fallback):
    """The first path that is actually there, else `fallback` for the error.

    This project has been moved once already - it started beside
    Major_Project_Iter1 and now lives inside it - and each layout puts
    the corpus somewhere different relative to this file. Rather than
    hard-code the winner and break the other, try both and let the
    `require()` message name the fallback when neither exists.
    """

    for path in candidates:
        if path.exists():
            return path
    return fallback


_HANDOFF_TAIL = ("handoff_Pranay", "home", "pranay", "dev",
                 "Major_Project_Iter1", "handoff1")

_DEFAULT_HANDOFF = _first_existing(
    [
        ROOT.parent.joinpath(*_HANDOFF_TAIL),                       # inside Iter1
        ROOT.parent / "Major_Project_Iter1" / Path(*_HANDOFF_TAIL),  # beside it
    ],
    ROOT.parent.joinpath(*_HANDOFF_TAIL),
)
HANDOFF = Path(os.environ.get("MPE_HANDOFF", _DEFAULT_HANDOFF))
HANDOFF_DATA = HANDOFF / "data"
HANDOFF_INDEX = HANDOFF_DATA / "index.json"

_DEFAULT_SCHEMES = _first_existing(
    [
        ROOT.parent / "answer_keys",                          # inside Iter1
        ROOT.parent / "Major_Project_Iter1" / "answer_keys",  # beside it
    ],
    ROOT.parent / "answer_keys",
)
SCHEME_PDFS = Path(os.environ.get("MPE_SCHEMES", _DEFAULT_SCHEMES))

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
            "Set MPE_HANDOFF / MPE_SCHEMES if the corpus lives elsewhere."
        )
    return path
