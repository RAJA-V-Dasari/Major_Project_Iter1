r"""
Run the model tier on this machine instead of on a free Colab T4.

    output/queue_llm.jsonl  ->  output/local_verdicts.jsonl
                            ->  python src/apply_verdicts.py output/local_verdicts.jsonl

WHAT CHANGED, AND WHAT DELIBERATELY DID NOT
-------------------------------------------
Only the transport. The prompt is imported from `llm_prompt.py`,
byte-identical to the one the recorded Colab run used; the queue records
are unchanged; the verdict rows are the shape `apply_verdicts.py`
already reads; and every check that made the Colab tier trustworthy -
the quote re-check, `zero_blocked`, the refusal to overwrite a settled
item - is downstream of this file and untouched by it.

That is the whole point. This tier was always a queue written to disk
rather than an in-process call, so moving it off Colab costs one file
and no change to the ladder.

WHY NOT transformers + bitsandbytes
-----------------------------------
The notebook loads Qwen2.5-7B in 4-bit through bitsandbytes, which
needs CUDA. On a laptop with no NVIDIA GPU that path does not degrade,
it fails: bitsandbytes has no usable CPU 4-bit kernel, and the
unquantised model is ~28 GB in fp32. So the local tier goes through a
GGUF runtime (llama.cpp, directly or via Ollama), which is built for
quantised CPU inference and memory-maps the weights.

The consequence is that **the local model is not bit-identical to the
Colab one.** Same weights, different quantisation and a different
kernel. Verdicts will not match item for item, and this file does not
claim they will - it writes to `local_verdicts.jsonl`, beside the
recorded `grade_llm_verdicts.jsonl` rather than over it, so the two can
be compared. `--compare` does that comparison.

THIS IS SLOW, AND THE SLOWNESS IS THE POINT OF --limit
------------------------------------------------------
740 items, a ~1,200-token prompt each. On two cores that is measured in
tens of hours, not minutes. Run `python src/setup_local.py` first: it
times the actual machine on one real item and multiplies, rather than
guessing. `--limit` exists so you can get a defensible sample overnight
instead of a full run over a week.

Every verdict is flushed as it is produced and re-running resumes, so
stopping with Ctrl-C costs one item.

Run:
    python src/setup_local.py                  # check the machine first
    python src/llm_local.py --check            # backend + model, mark nothing
    python src/llm_local.py --limit 25         # a sample
    python src/llm_local.py                    # the whole queue, resumable
    python src/llm_local.py --compare          # local vs the recorded Colab run
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_prompt
import paths

# Qwen2.5-7B-Instruct, 4-bit, as the recorded run used - but as a GGUF
# quant through a CPU runtime. Q4_K_M is the closest widely-available
# analogue of the notebook's nf4 double-quant.
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"

# The longest prompt in the queue is ~2,030 tokens and the reply is
# capped at 300. Ollama's own default is 2048, which would silently
# truncate the long tail - and a truncated prompt loses the END of the
# user turn, which is the student's answer. The model would then mark an
# answer it was never shown and confidently return "zero". Set it
# explicitly; never inherit it.
NUM_CTX = 4096
MAX_NEW_TOKENS = 300

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# `do_sample=False` in the notebook. Greedy here too, so a re-run of the
# same item on the same build gives the same verdict and a disagreement
# means something.
TEMPERATURE = 0.0


# ------------------------------------------------------------- backends


class OllamaBackend:
    """Talk to a local Ollama server over HTTP.

    Deliberately urllib and not `requests`: this backend then adds no
    dependency at all to a project whose base install is three packages.
    """

    def __init__(self, model, host=OLLAMA_HOST, num_ctx=NUM_CTX):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx

    @property
    def name(self):
        return f"ollama/{self.model}"

    def _post(self, route, payload, timeout):
        request = urllib.request.Request(
            f"{self.host}{route}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def preflight(self):
        """Server up, and this model actually pulled."""

        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as r:
                tags = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise SystemExit(
                f"no Ollama server at {self.host}  ({error.reason})\n"
                "Start it with:  ollama serve\n"
                "Install it from https://ollama.com/download - see "
                "docs/LOCAL_SETUP.md"
            )
        have = [m.get("name", "") for m in tags.get("models", [])]
        # Ollama reports "qwen2.5:7b-instruct-q4_K_M"; accept the
        # ":latest" spelling too rather than failing on a cosmetic suffix.
        if not any(h == self.model or h == f"{self.model}:latest" or
                   h.split(":")[0] == self.model for h in have):
            raise SystemExit(
                f"model not pulled: {self.model}\n"
                f"Pull it with:  ollama pull {self.model}\n"
                + ("Models on this machine: " + ", ".join(have) if have
                   else "This machine has no models pulled yet.")
            )
        return f"{self.name}  (ctx {self.num_ctx}, host {self.host})"

    def complete(self, system, user, timeout=1800):
        reply = self._post("/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_NEW_TOKENS,
                "num_ctx": self.num_ctx,
            },
        }, timeout=timeout)
        return reply.get("message", {}).get("content", "")


class LlamaCppBackend:
    """Run a GGUF file in-process, with no server.

    The fallback for a machine where installing Ollama is not an option.
    Needs `pip install llama-cpp-python` and a GGUF file on disk.
    """

    def __init__(self, model_path, num_ctx=NUM_CTX, threads=None):
        self.model_path = Path(model_path)
        self.num_ctx = num_ctx
        self.threads = threads or max(1, (os.cpu_count() or 2))
        self._llm = None

    @property
    def name(self):
        return f"llama-cpp/{self.model_path.name}"

    def preflight(self):
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            raise SystemExit(
                "llama-cpp-python is not installed.\n"
                "    pip install llama-cpp-python\n"
                "or use the Ollama backend - see docs/LOCAL_SETUP.md"
            )
        if not self.model_path.exists():
            raise SystemExit(f"GGUF not found at:\n    {self.model_path}")
        return f"{self.name}  (ctx {self.num_ctx}, {self.threads} threads)"

    def _load(self):
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.num_ctx,
                n_threads=self.threads,
                verbose=False,
            )
        return self._llm

    def complete(self, system, user, timeout=None):
        out = self._load().create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        )
        return out["choices"][0]["message"]["content"]


def make_backend(args):
    if args.backend == "llama-cpp":
        if not args.gguf:
            raise SystemExit("--backend llama-cpp needs --gguf <path to .gguf>")
        return LlamaCppBackend(args.gguf, num_ctx=args.num_ctx,
                               threads=args.threads)
    return OllamaBackend(args.model, num_ctx=args.num_ctx)


# ------------------------------------------------------------------ io


def load_queue(path):
    paths.require(path, "the model queue (run src/grade.py --all first)")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_done(path):
    """Which items this verdicts file already holds, and by which models.

    Returns (keys, models). The model set matters: a verdicts file that
    mixes two models is not one experiment, and `apply_verdicts.py`
    would apply it as though it were.
    """

    keys, models = set(), Counter()
    if not path.exists():
        return keys, models
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                keys.add(llm_prompt.record_key(row))
                models[row.get("model", "?")] += 1
            except Exception:
                continue
    return keys, models


def fmt_duration(seconds):
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------- main


def check_destination(out_path, backend_name, allow_mixed):
    """Refuse to turn one verdicts file into two experiments.

    Deliberately called BEFORE the backend preflight. It needs no server
    and no model, so a run aimed at the wrong file should fail on this
    rather than after a 5 GB pull - and pointing --out at the recorded
    Colab run is exactly the mistake worth catching for free.
    """

    done, existing_models = load_done(out_path)
    if existing_models and backend_name not in existing_models:
        if not allow_mixed:
            raise SystemExit(
                f"{out_path.name} already holds {sum(existing_models.values())} "
                f"verdicts from: {', '.join(sorted(existing_models))}\n"
                f"This run would append verdicts from {backend_name}, making "
                "one file that is two experiments.\n"
                "Use --out to start a separate file, or --allow-mixed-models "
                "if mixing really is what you want."
            )
        print(f"! appending {backend_name} to a file that already holds "
              f"{', '.join(sorted(existing_models))}")
    return done


def run(args, backend, records, out_path, done):
    todo = [r for r in records if llm_prompt.record_key(r) not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(records)} queued, {len(done)} already in {out_path.name}, "
          f"{len(todo)} to mark now")
    if not todo:
        print("nothing to do")
        return

    if args.dry_run:
        first = todo[0]
        print("\n--- first prompt, as it would be sent "
              "-----------------------------")
        print(llm_prompt.build_prompt(first)[:1200])
        print("--- (truncated) ----------------------------------------------")
        return

    started = time.time()
    counts = Counter()
    marked = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(out_path, "a", encoding="utf-8") as handle:
            for n, record in enumerate(todo, 1):
                try:
                    reply = backend.complete(
                        llm_prompt.SYSTEM_PROMPT,
                        llm_prompt.build_prompt(record),
                    )
                    verdict = llm_prompt.parse_reply(reply)
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    # A backend failure is a decline, never a zero. The
                    # item goes to a person rather than being settled by
                    # a timeout.
                    verdict = {"verdict": "decline", "marks": 0, "quote": "",
                               "reason": f"backend error: {type(error).__name__}"}

                row = llm_prompt.verdict_row(record, verdict, backend.name)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                counts[row["verdict"]] += 1
                marked = n

                if n % args.progress_every == 0 or n == len(todo):
                    per = (time.time() - started) / n
                    left = (len(todo) - n) * per
                    print(f"  {n}/{len(todo)}  {per:.0f}s/item  "
                          f"{fmt_duration(left)} left  "
                          f"{dict(counts)}", flush=True)
    except KeyboardInterrupt:
        print(f"\nstopped after {marked} item(s) - re-run to resume")

    elapsed = time.time() - started
    print(f"\n{marked} marked in {fmt_duration(elapsed)} -> {out_path}")
    print("verdicts:", dict(counts))
    print("\nNext:")
    print(f"    python src/apply_verdicts.py {out_path} --dry-run")
    print(f"    python src/apply_verdicts.py {out_path}")


def compare(out_path, reference):
    """Local verdicts against the recorded Colab run, on shared items.

    Not a correctness check - neither run is ground truth. It is a
    measure of how much the quantisation and the runtime change the
    tier's behaviour, which is the thing this copy makes it possible to
    ask at all.
    """

    if not out_path.exists():
        raise SystemExit(f"no local verdicts at {out_path} - run the tier first")
    if not reference.exists():
        raise SystemExit(f"no recorded run at {reference} to compare against")

    def index(path):
        rows = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                rows[llm_prompt.record_key(row)] = row
        return rows

    mine, theirs = index(out_path), index(reference)
    shared = sorted(set(mine) & set(theirs))
    print(f"local {len(mine)}   recorded {len(theirs)}   shared {len(shared)}")
    if not shared:
        return

    agree = sum(1 for k in shared
                if mine[k]["verdict"] == theirs[k]["verdict"])
    print(f"same verdict on {agree}/{len(shared)} "
          f"({100 * agree / len(shared):.0f}%)\n")

    print(f"{'':<10}" + "".join(f"{v:>10}" for v in ("award", "zero", "decline"))
          + "   <- recorded")
    for mine_v in ("award", "zero", "decline"):
        row = [sum(1 for k in shared
                   if mine[k]["verdict"] == mine_v
                   and theirs[k]["verdict"] == their_v)
               for their_v in ("award", "zero", "decline")]
        print(f"{mine_v:<10}" + "".join(f"{c:>10}" for c in row))
    print("^ local")

    models = Counter(r.get("model", "?") for r in mine.values())
    print("\nlocal model(s):", ", ".join(sorted(models)))
    print("recorded model(s):",
          ", ".join(sorted({r.get("model", "?") for r in theirs.values()})))


def main():
    parser = argparse.ArgumentParser(
        description="Run the model tier locally instead of on Colab.")
    parser.add_argument("--backend", choices=["ollama", "llama-cpp"],
                        default="ollama")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL,
                        help=f"Ollama model tag (default {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--gguf", help="GGUF path, for --backend llama-cpp")
    parser.add_argument("--num-ctx", type=int, default=NUM_CTX)
    parser.add_argument("--threads", type=int, default=None,
                        help="llama-cpp only; defaults to all cores")
    parser.add_argument("--out", default=None,
                        help="verdicts file (default output/local_verdicts.jsonl)")
    parser.add_argument("--booklet", action="append", default=None,
                        metavar="BOOKLET_ID",
                        help="mark only this booklet; repeatable. Unlike "
                             "--limit, which takes whatever is first in the "
                             "queue, this keeps a run to whole booklets - "
                             "which is what serve.py and agreement.py report "
                             "on, so a part-marked booklet is harder to read "
                             "than an unmarked one.")
    parser.add_argument("--limit", type=int, default=None,
                        help="mark at most N unmarked items, then stop")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--allow-mixed-models", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="verify backend and model, mark nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the first prompt, mark nothing")
    parser.add_argument("--compare", action="store_true",
                        help="local verdicts vs the recorded Colab run")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else paths.OUT_DIR / "local_verdicts.jsonl"

    if args.compare:
        compare(out_path, paths.OUT_DIR / "grade_llm_verdicts.jsonl")
        return

    queue = paths.OUT_DIR / "queue_llm.jsonl"
    records = load_queue(queue)

    if args.booklet:
        wanted = set(args.booklet)
        present = {r["booklet_id"] for r in records}
        unknown = wanted - present
        if unknown:
            raise SystemExit(
                "not in the queue: " + ", ".join(sorted(unknown)) + "\n"
                "A booklet with nothing queued is not an error upstream - it "
                "means the cheap tiers settled all of it, or grade.py has not "
                "run since. Check output/queue_llm.jsonl."
            )
        records = [r for r in records if r["booklet_id"] in wanted]
        print(f"restricted to {len(wanted)} booklet(s): "
              f"{', '.join(sorted(wanted))}")

    backend = make_backend(args)

    # Cheap, local, no server needed - so it runs first.
    done = check_destination(out_path, backend.name, args.allow_mixed_models)

    if not args.dry_run:
        print(backend.preflight())

    if args.check:
        _, models = load_done(out_path)
        print(f"{len(records)} items queued in {queue.name}")
        print(f"{len(done)} already marked in {out_path.name}"
              + (f" by {', '.join(sorted(models))}" if models else ""))
        print(f"{len(records) - len(done)} remaining")
        return

    run(args, backend, records, out_path, done)


if __name__ == "__main__":
    main()
