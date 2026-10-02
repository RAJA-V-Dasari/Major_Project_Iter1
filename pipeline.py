"""
The whole project behind one command.

    python pipeline.py status        what exists, stage by stage, and what next
    python pipeline.py fetch         prepared pages from Hugging Face -> data/pages
    python pipeline.py prepare       raw scans -> prepared pages (only to rebuild)
    python pipeline.py read          pages -> Markdown, one per page    [the GPU step]
    python pipeline.py batches       package pages for a Colab/Kaggle read
    python pipeline.py figures       package read pages for the diagram pass
    python pipeline.py assemble      pages -> one booklet per student per CIE
    python pipeline.py handoff       booklets -> the contract marking reads
    python pipeline.py mark          handoff -> marks, and agreement with the examiner
    python pipeline.py review        the marking website, http://127.0.0.1:8000
    python pipeline.py run           assemble -> handoff -> mark, stopping on failure
    python pipeline.py check         every check that needs no model and no GPU

Options after a command go to that stage's script:

    python pipeline.py fetch --students 1-5 --cie 2
    python pipeline.py read --reader server --url http://gpu-box:8000/v1
    python pipeline.py assemble --engine all_read
    python pipeline.py mark --no-semantic
    python pipeline.py run --engine all_read --no-semantic

WHY A DRIVER AND NOT A LIST IN A README
--------------------------------------
Every stage is a script that runs on its own, and each prints the
command for the next. What the driver adds is the order, which has gone
wrong by hand before: a booklet assembled from one read and marked
against a handoff built from another, a `grade.py --all` that silently
discarded every model and human decision not replayed after it. `run`
owns the order for the stages that need no GPU; reading the pages is the
one step it cannot do for you, because it needs a GPU - see `read`.

Every script reads and writes under one data root (common/layout.py):
data/ at the top of the repo, or $MP_DATA.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from common import layout                                  # noqa: E402

STAGES = {
    "fetch": [("scripts/fetch_hf.py", [])],
    "prepare": [("reading/prepare/deskew.py", []),
                ("reading/prepare/crop.py", []),
                ("reading/prepare/tone.py", [])],
    "read": [("reading/read/read_pages.py", [])],
    "batches": [("reading/read/prepare_corpus_batches.py", [])],
    "figures": [("reading/figures/build_batch.py", [])],
    "assemble": [("reading/assemble/build_booklet.py", ["--all"])],
    "handoff": [("reading/handoff/export.py", [])],
    "mark": [("marking/src/run_all.py", [])],
    "review": [("marking/src/serve.py", [])],
}

RUN_ORDER = ("assemble", "handoff", "mark")


def child_env():
    """The environment every stage runs in.

    UTF-8 mode, because these scripts read and print students' answers -
    arrows, LaTeX, Greek - and a Windows console's default code page
    turns the first one into a UnicodeEncodeError.
    """

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    return env


def script(path, args, title=None):
    """Run one stage script with this interpreter; stop the run on failure."""

    print(f"\n=== {title or path}", flush=True)
    print("    " + " ".join(["python", path, *args]), flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, str(REPO / path), *args],
                            cwd=REPO, env=child_env())
    if result.returncode != 0:
        raise SystemExit(f"\n{path} failed (exit {result.returncode}) - "
                         "stopping; later stages would run on stale input")
    print(f"    ({time.time() - started:.0f}s)", flush=True)


def stage(name, extra):
    for path, defaults in STAGES[name]:
        script(path, [*defaults, *extra], f"{name}: {path}")


def take(args, flag, many=False, has_value=True):
    """Remove `flag` (and its value) from args; return what was taken."""

    taken, rest, i = [], [], 0
    while i < len(args):
        if args[i] == flag:
            if has_value and i + 1 < len(args):
                taken.append(args[i + 1])
                i += 2
                continue
            taken.append(True)
            i += 1
            continue
        if has_value and args[i].startswith(flag + "="):
            taken.append(args[i].split("=", 1)[1])
            i += 1
            continue
        rest.append(args[i])
        i += 1
    args[:] = rest
    if many:
        return taken
    return taken[-1] if taken else None


def run(args):
    """assemble -> handoff -> mark, in the one order that is right."""

    args = list(args)
    engine = take(args, "--engine")
    start = take(args, "--from") or RUN_ORDER[0]
    no_semantic = take(args, "--no-semantic", has_value=False)
    verdicts = take(args, "--verdicts", many=True)
    if args:
        raise SystemExit(f"run does not take {' '.join(args)} - pass "
                         "stage-specific options to that stage instead")
    if start not in RUN_ORDER:
        raise SystemExit(f"--from must be one of {', '.join(RUN_ORDER)}")

    plan = {
        "assemble": ["--engine", engine] if engine else [],
        "handoff": [],
        "mark": ((["--no-semantic"] if no_semantic else [])
                 + [part for v in verdicts for part in ("--verdicts", v)]),
    }
    for name in RUN_ORDER[RUN_ORDER.index(start):]:
        stage(name, plan[name])

    print("\nDone. Review the marks: python pipeline.py review")


def count(folder, pattern):
    return sum(1 for _ in folder.glob(pattern)) if folder.exists() else 0


def status(_args):
    """What exists under the data root, stage by stage."""

    print(f"data root  {layout.DATA}"
          + ("" if layout.DATA.exists() else "   (not created yet)"))
    print(f"HF token   {'set' if layout.hf_token() else 'MISSING - put HF_TOKEN in .env'}")
    print()

    rows = []
    raw = count(layout.RAW, "student_*/cie_*/page_*.png")
    rows.append(("raw scans", f"{raw} pages" if raw else "-",
                 "only needed to rebuild data/pages"))

    pages = count(layout.PAGES, "student_*/cie_*/page_*.png")
    covers = count(layout.PAGES, "student_*/cie_*/page_01.png")
    booklets = count(layout.PAGES, "student_*/cie_*")
    rows.append(("pages", f"{pages - covers} content pages, {booklets} "
                          f"booklets" if pages else "-",
                 "" if pages else "python pipeline.py fetch"))

    reads = []
    if layout.READ.exists():
        for folder in sorted(layout.READ.iterdir()):
            if folder.is_dir() and folder.name not in ("batches", "kaggle"):
                reads.append(f"{folder.name} ({count(folder, '*.md')})")
    rows.append(("read", ", ".join(reads) or "-",
                 "" if reads else "the GPU step - see docs/PIPELINE.md"))

    diagrams = (layout.FIGURES / "diagrams.json").exists()
    rejects = (layout.FIGURES / "rejects.json").exists()
    rows.append(("figures", ("diagram pass done" if diagrams else "-")
                 + (", reviewed" if rejects else ""),
                 "" if diagrams else "optional: python pipeline.py figures"))

    structures = sorted(layout.BOOKLETS.glob("*/structure.json")) \
        if layout.BOOKLETS.exists() else []
    engines = sorted({json.loads(p.read_text(encoding="utf-8"))["engine"]
                      for p in structures})
    rows.append(("booklets", f"{len(structures)} from {', '.join(engines)}"
                 if structures else "-",
                 "" if structures else "python pipeline.py assemble"))

    handoff = layout.handoff_data()
    index = handoff / "index.json"
    if index.exists():
        data = json.loads(index.read_text(encoding="utf-8"))
        where = "" if handoff == layout.HANDOFF else f"  [{handoff}]"
        rows.append(("handoff", f"{data.get('booklets_total', '?')} booklets"
                     f"{where}", ""))
    else:
        rows.append(("handoff", "-", "python pipeline.py handoff"))

    marks = count(layout.MARKING / "marks", "*.json")
    extras = [name for name in ("grade_llm_verdicts.jsonl",
                                "local_verdicts.jsonl",
                                "claude_verdicts.jsonl",
                                "human_marks.jsonl", "agreement.md")
              if (layout.MARKING / name).exists()]
    rows.append(("marks", f"{marks} booklets" + (f"; {', '.join(extras)}"
                                                 if extras else "")
                 if marks else "-",
                 "python pipeline.py review" if marks
                 else "python pipeline.py mark"))

    width = max(len(r[0]) for r in rows)
    for name, state, hint in rows:
        print(f"  {name:<{width}}  {state}" + (f"   -> {hint}" if hint else ""))


def check(_args):
    """Every verification that needs no model, no GPU and no network."""

    tests = REPO / "tests"
    if tests.exists():
        print("\n=== unit tests", flush=True)
        result = subprocess.run([sys.executable, "-m", "unittest", "discover",
                                 "-s", "tests", "-t", "."],
                                cwd=REPO, env=child_env())
        if result.returncode:
            raise SystemExit("unit tests failed")

    # The scheme PDFs are the department's, not in git or on the Hub, so a
    # fresh machine has no renders to check against. The rubric's own
    # arithmetic is checked either way.
    rendered = layout.SCHEME_PAGES.exists() and any(
        layout.SCHEME_PAGES.glob("*.png"))
    script("marking/src/validate_keys.py",
           [] if rendered else ["--no-renders"],
           "the rubric is consistent"
           + ("" if rendered else " (scheme renders not checked: no "
                                  "PDFs rendered into data/schemes/pages)"))
    script("marking/src/gold_check.py", [], "the examiner's arithmetic")
    script("marking/src/make_llm_notebook.py", ["--check"],
           "the model-tier notebook compiles")
    script("reading/read/make_colab_notebook.py", ["--check"],
           "the reading notebook compiles and is current")

    if layout.BOOKLETS.exists() and any(layout.BOOKLETS.glob("*/structure.json")):
        script("reading/handoff/export.py", ["--check"],
               "the booklets package cleanly")
    if (layout.handoff_data() / "index.json").exists():
        script("marking/src/load_handoff.py", ["--report"],
               "marking reads the handoff as it was written")
    if (layout.MARKING / "marks").exists():
        script("marking/src/serve.py", ["--check"],
               "every crop the website needs is on disk")

    print("\nAll checks passed.")


COMMANDS = {"run": run, "status": status, "check": check}


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11 or newer is needed")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    command, extra = sys.argv[1], sys.argv[2:]

    if command in COMMANDS:
        COMMANDS[command](extra)
        return 0

    if command in STAGES:
        stage(command, extra)
        return 0

    raise SystemExit(f"unknown command {command!r} - "
                     "python pipeline.py --help")


if __name__ == "__main__":
    raise SystemExit(main())
