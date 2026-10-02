"""
Import answer scripts (or any private dataset repo) from Hugging Face.

    python scripts/fetch_hf.py --list                         # what is there
    python scripts/fetch_hf.py                                # cleaned pages
    python scripts/fetch_hf.py --students 1-5 --cie 2         # a subset
    python scripts/fetch_hf.py --repo raw --students 7        # raw scans
    python scripts/fetch_hf.py --repo <org>/<name> --out data/handoff

    HF repo  ->  data/ (gitignored)  ->  the pipeline

    cleaned  ->  data/pages/    prepared pages, what the reader reads
    raw      ->  data/raw/      untouched scans, what `prepare` reads
    other    ->  data/hf/<name>/

The two answer-script repos are FOLDER datasets, not Parquet tables:

    student_<NN>/cie_<M>/page_<PP>.png        (+ README.md, .gitattributes;
                                               the raw repo also has
                                               students.csv, the roster)

so importing them is downloading files, and `--students` / `--cie` just
narrow which paths are fetched. See docs/DATA.md for the format and the
other ways in.

PRIVACY, ENFORCED HERE RATHER THAN HOPED FOR
--------------------------------------------
* page_01 is every booklet's cover - name, USN, signature, the
  examiner's marks. It is skipped unless you pass --with-covers.
* students.csv (names and USNs) is skipped unless --with-roster.
* The token is read from HF_TOKEN or the repo's .env and never printed.
* The default destinations are under data/, which is gitignored. Anything
  under --out that you point elsewhere is your responsibility to keep out
  of git.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "common" / "layout.py").exists())))

from common import layout                                  # noqa: E402

REPOS = {
    "raw": "prss-majorproject-37/Handwritten-AnswerScripts-MajorProject",
    "cleaned": "prss-majorproject-37/cleaned-handwritten-answerscripts",
}
DEFAULT_OUT = {"raw": layout.RAW, "cleaned": layout.PAGES}


def parse_numbers(text):
    """'1,4,10-12' -> [1, 4, 10, 11, 12]"""

    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            out.extend(range(int(low), int(high) + 1))
        else:
            out.append(int(part))
    return out


def patterns(args, is_script_repo):
    """The allow/ignore globs handed to snapshot_download."""

    if not is_script_repo:
        return (args.include or None), None

    students = parse_numbers(args.students) or [None]
    cies = parse_numbers(args.cie) or [None]
    allow = []
    for s in students:
        for c in cies:
            allow.append(f"{'student_%02d' % s if s else 'student_*'}/"
                         f"{'cie_%d' % c if c else 'cie_*'}/*.png")
    allow.append("README.md")
    if args.with_roster:
        allow.append("students.csv")
    ignore = None if args.with_covers else ["*/page_01.png"]
    return allow, ignore


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--repo", default="cleaned",
                        help="raw | cleaned | any <org>/<name> dataset repo")
    parser.add_argument("--out", type=Path, help="where to put it")
    parser.add_argument("--students", help="e.g. 1,4,10-12 (default: all)")
    parser.add_argument("--cie", help="e.g. 2 or 1,3 (default: all)")
    parser.add_argument("--include", action="append",
                        help="glob to fetch, for repos other than the "
                             "answer-script ones (repeatable)")
    parser.add_argument("--with-covers", action="store_true",
                        help="also fetch page_01 (names, USNs, marks)")
    parser.add_argument("--with-roster", action="store_true",
                        help="also fetch students.csv (names and USNs)")
    parser.add_argument("--list", action="store_true",
                        help="describe the repo, download nothing")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise SystemExit("needs huggingface_hub: pip install -e . "
                         "(or pip install huggingface_hub)")

    repo_id = REPOS.get(args.repo, args.repo)
    is_script_repo = repo_id in REPOS.values()
    token = layout.hf_token()
    if not token:
        print("no HF_TOKEN in the environment or .env - private repos "
              "will refuse", file=sys.stderr)

    api = HfApi(token=token)
    if args.list:
        info = api.dataset_info(repo_id, files_metadata=True)
        files = info.siblings
        pngs = [f for f in files if f.rfilename.endswith(".png")]
        students = {f.rfilename.split("/")[0] for f in pngs}
        booklets = {"/".join(f.rfilename.split("/")[:2]) for f in pngs}
        size = sum(f.size or 0 for f in files) / 1e6
        print(f"{repo_id}  ({'private' if info.private else 'PUBLIC'})")
        print(f"  {len(files)} files, {size:.0f} MB, last changed "
              f"{info.last_modified:%Y-%m-%d}")
        if pngs:
            print(f"  {len(students)} students, {len(booklets)} booklets, "
                  f"{len(pngs)} pages")
        top = sorted(f.rfilename for f in files if "/" not in f.rfilename)
        print("  top level:", ", ".join(top))
        return

    out = args.out or DEFAULT_OUT.get(args.repo) or (
        layout.DATA / "hf" / repo_id.split("/")[-1])
    allow, ignore = patterns(args, is_script_repo)
    print(f"{repo_id} -> {out}")
    if is_script_repo and not args.with_covers:
        print("  skipping page_01 (covers); --with-covers to include")
    path = snapshot_download(repo_id=repo_id, repo_type="dataset",
                             local_dir=out, token=token,
                             allow_patterns=allow, ignore_patterns=ignore)
    pages = list(Path(path).glob("student_*/cie_*/*.png"))
    print(f"done: {len(pages)} page image(s) under {path}"
          if is_script_repo else f"done: {path}")


if __name__ == "__main__":
    main()
