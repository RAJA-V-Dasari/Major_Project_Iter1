"""
Part 2 end to end, on part 1's handoff, in the one order that is right.

    python src/run_all.py
    python src/run_all.py --verdicts output/claude_verdicts.jsonl
    python src/run_all.py --no-semantic

    handoff1/  ->  load_handoff  ->  align  ->  grade --all
               ->  apply_verdicts (each file)  ->  apply_human  ->  agreement

WHY A DRIVER AND NOT A LIST IN A README
--------------------------------------
Every step here already exists and each is correct on its own. What went
wrong was the order, twice:

  * `align.py --report` prints the resolved labels and returns before
    writing `output/alignment.json`. The run order in HANDOFF.md used
    `--report`, so `grade.py` ran without the resolved labels - 103
    unattempted instead of 90, and 9 location failures instead of 6 -
    and nothing said so.
  * `grade.py --all` rewrites every marks file from scratch. Any
    verdicts or human decisions not replayed after it are silently gone,
    and what you see is the cheap tiers alone.

Both are properties of the sequence, not of any one script, so the
sequence is what this file owns. It adds no marking logic of its own.

WHICH VERDICTS
--------------
The model tier has had three readers - Qwen on Colab
(`grade_llm_verdicts.jsonl`), Qwen via Ollama (`local_verdicts.jsonl`),
Claude in session (`claude_verdicts.jsonl`). `apply_verdicts.py` skips
an item another file already settled, so applying two of them means the
first one silently wins every item they share. With no `--verdicts`
this uses the one that is present, and refuses to guess when there is
more than one.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths                                               # noqa: E402

KNOWN_VERDICTS = ("grade_llm_verdicts.jsonl", "local_verdicts.jsonl",
                  "claude_verdicts.jsonl")


def step(title, *argv):
    print(f"\n=== {title}", flush=True)
    print("    " + " ".join(["python", *argv]), flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, str(HERE / argv[0]), *argv[1:]])
    if result.returncode != 0:
        raise SystemExit(f"\n{argv[0]} failed (exit {result.returncode}); "
                         "stopping - later steps would run on stale input")
    print(f"    ({time.time() - started:.0f}s)", flush=True)


def pick_verdicts(explicit):
    if explicit:
        for path in explicit:
            paths.require(Path(path), "verdicts file")
        return [Path(p) for p in explicit]

    present = [paths.OUT_DIR / name for name in KNOWN_VERDICTS
               if (paths.OUT_DIR / name).exists()]
    if len(present) > 1:
        raise SystemExit(
            "more than one model-tier verdicts file is present:\n    "
            + "\n    ".join(str(p) for p in present)
            + "\nthey come from different readers; name the one to apply "
              "with --verdicts")
    return present


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--verdicts", action="append", metavar="FILE",
                        help="model-tier verdicts to apply, in order")
    parser.add_argument("--no-semantic", action="store_true")
    args = parser.parse_args()

    paths.require(paths.HANDOFF_INDEX, "part 1's handoff (data/index.json)")
    print(f"handoff: {paths.HANDOFF}")
    verdicts = pick_verdicts(args.verdicts)

    step("part 1's handoff, read as part 2 sees it",
         "load_handoff.py", "--report")
    step("resolve parts with no usable question label", "align.py")
    step("the ladder: exact, keyword, semantic",
         "grade.py", "--all", *(["--no-semantic"] if args.no_semantic else []))

    if not verdicts:
        print("\n=== model tier: no verdicts file - its 740 items stay pending")
    for path in verdicts:
        step(f"model tier: {path.name}", "apply_verdicts.py", str(path))

    if (paths.OUT_DIR / "human_marks.jsonl").exists():
        step("human tier", "apply_human.py")
    else:
        print("\n=== human tier: no output/human_marks.jsonl yet "
              "(python src/serve.py writes it)")

    step("agreement with the examiner", "agreement.py")

    print(f"\nmarks:     {paths.MARKS_DIR}")
    print(f"summary:   {paths.OUT_DIR / 'summary.csv'}")
    print(f"agreement: {paths.OUT_DIR / 'agreement.md'}")


if __name__ == "__main__":
    main()
