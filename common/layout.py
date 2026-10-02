"""
Where every stage of the pipeline reads and writes.

Everything derived from the corpus lives under ONE gitignored root,
`data/` at the top of the repo. One `.gitignore` line is then the whole
protection against committing student work: no glob has to be right
about a folder somebody adds later.

    data/                          $MP_DATA moves the whole tree
      raw/                         untouched scans (HF raw repo)       prepare's input
      prepare/01_deskew/           deskewed pages + angles.json
      prepare/02_crop/             cropped pages + measurements.json
      prepare/preview/<stage>/     before/after pairs from --preview
      pages/                       PREPARED pages (HF cleaned repo)    the reader's input
      read/<engine>/               one transcription per page          <page_id>.md
      read/batches/                zips + manifests for a hosted GPU
      read/kaggle/                 Kaggle staging (kaggle_run.py)
      figures/                     the diagram pass: batch, diagrams.json,
                                   review.html, rejects.json
      geometry/                    segment.py run on its own, to inspect
      booklets/<booklet>/          booklet.md, structure.json, figures/
      handoff/                     index.json + <booklet>/booklet.json,
                                   pages/, regions/ - what marking reads
      marking/                     marks/, summary.csv, queues, verdicts,
                                   the human log, every report
      schemes/                     the three scheme PDFs; pages/ = renders
      benchmark/ground_truth/      hand transcriptions the reader is scored on

A page id is `s<NN>_c<C>_p<PP>`: s01_c2_p03 is student_01/cie_2/page_03.png.
A booklet id is `student_<NN>_cie_<C>`.

CONFIGURATION
-------------
`.env` at the repo root is read once, here, on import. A variable that
is already set in the environment wins over the file, and no value is
ever printed. The variables:

    HF_TOKEN       read access to the private Hugging Face repos
    MP_DATA        the data root (default: <repo>/data)
    MP_ENGINE      the default reader name, i.e. the folder under read/
    MPE_HANDOFF    mark an existing handoff instead of data/handoff -
                   the folder holding index.json, or one holding
                   data/index.json (the shape part 1 shipped in September)
    MPE_SCHEMES    the folder holding the three scheme PDFs
"""

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_env(path=REPO / ".env"):
    """KEY=VALUE lines from `.env` into os.environ, never overriding."""

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


load_env()


def _path(variable, default):
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


DATA = _path("MP_DATA", REPO / "data")

RAW = DATA / "raw"
PREPARE = DATA / "prepare"
PAGES = DATA / "pages"
READ = DATA / "read"
BATCHES = READ / "batches"
FIGURES = DATA / "figures"
GEOMETRY = DATA / "geometry"
BOOKLETS = DATA / "booklets"
HANDOFF = DATA / "handoff"
MARKING = DATA / "marking"
SCHEMES = _path("MPE_SCHEMES", DATA / "schemes")
SCHEME_PAGES = DATA / "schemes" / "pages"
BENCHMARK = DATA / "benchmark"

# The name a read is filed under. The Colab notebook calls its 7B run
# "qwen7b", so unzipping its download into read/qwen7b/ needs no flags.
DEFAULT_ENGINE = os.environ.get("MP_ENGINE") or "qwen7b"

# page_01 of every booklet is the identity cover: name, USN, signature
# and the examiner's marks. Every stage that touches pages excludes it.
COVER_PAGE = 1

PAGE_ID = re.compile(r"^s(\d+)_c(\d+)_p(\d+)$")
BOOKLET_ID = re.compile(r"^student_(\d+)_cie_(\d+)$")
PAGE_PATH = re.compile(r"student_(\d+)[\\/]cie_(\d+)[\\/]page_(\d+)\.png$")


def hf_token():
    """The Hugging Face token, or None. Never print it."""

    return os.environ.get("HF_TOKEN") or None


def read_dir(engine=None):
    """Where one reader's page transcriptions live."""

    return READ / (engine or DEFAULT_ENGINE)


def page_id(student, cie, page):
    return f"s{int(student):02d}_c{int(cie)}_p{int(page):02d}"


def booklet_id(student, cie):
    return f"student_{int(student):02d}_cie_{int(cie)}"


def parse_page_id(text):
    """'s01_c2_p03' -> (1, 2, 3), or None."""

    match = PAGE_ID.match(text or "")
    return tuple(int(g) for g in match.groups()) if match else None


def parse_page_path(path):
    """.../student_01/cie_2/page_03.png -> (1, 2, 3), or None."""

    match = PAGE_PATH.search(str(path))
    return tuple(int(g) for g in match.groups()) if match else None


def page_image(text, root=None):
    """The prepared image for a page id, under `root` (default: PAGES)."""

    student, cie, page = parse_page_id(text)
    return ((root or PAGES) / f"student_{student:02d}" / f"cie_{cie}"
            / f"page_{page:02d}.png")


def handoff_data():
    """The folder holding the handoff's index.json and booklet folders.

    data/handoff by default. MPE_HANDOFF may name either that folder
    itself or an older handoff that nests it one level down, under
    data/ - the shape part 1 shipped in September as handoff1/.
    """

    root = _path("MPE_HANDOFF", HANDOFF)
    if not (root / "index.json").exists() and \
            (root / "data" / "index.json").exists():
        return root / "data"
    return root


def require(path, what, hint=None):
    """Fail with the path that was looked for, not a bare traceback."""

    if not Path(path).exists():
        message = f"{what} not found at:\n    {path}"
        if hint:
            message += f"\n{hint}"
        raise SystemExit(message)
    return path
