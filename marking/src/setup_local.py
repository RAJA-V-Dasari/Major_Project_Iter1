r"""
Check whether this machine can actually run the model tier, and how long
it would take.

    python marking/src/setup_local.py            # inspect the machine, decide
    python marking/src/setup_local.py --time-it  # mark one real item, extrapolate

WHY THIS EXISTS
---------------
The Colab tier had one hardware question - "did you pick the T4?" - and
the notebook answered it in a markdown cell. A local tier has several,
and getting them wrong is expensive in a way that is not obvious for
hours: a 7B model that does not fit in RAM does not refuse to load, it
swaps, and a run that should take twenty hours takes a week while
looking exactly the same from the outside.

So this file measures rather than assumes. `--time-it` marks one real
item off the real queue with the real prompt, and multiplies by what is
left. An estimate from your machine on your queue beats any number
written in a README by someone with a different laptop.

WHAT IT WILL TELL YOU TO DO
---------------------------
Probably to use a smaller model than the Colab run used, and it will say
so in marks rather than in adjectives. Qwen2.5-7B-Instruct at Q4_K_M
needs roughly 5 GB resident. On a 8 GB laptop that is not comfortable,
and the honest options are a smaller Qwen or a long weekend - see
docs/LOCAL_SETUP.md, which lays out what each choice costs.

Nothing here writes a verdict or touches the marks.
"""

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_local
import llm_prompt
import paths

# (tag, download GB, resident GB, note). Download is the GGUF on disk.
# Resident is that plus the KV cache at NUM_CTX and the runtime's own
# overhead - which is the number to plan RAM around, and the one people
# skip. Both are approximate and rounded up.
MODELS = [
    ("qwen2.5:7b-instruct-q4_K_M", 4.7, 5.6,
     "matches the Colab run's weights; the faithful choice"),
    ("qwen2.5:3b-instruct-q4_K_M", 1.9, 2.6,
     "a different model, ~2.5x faster; say so if you report its numbers"),
    ("qwen2.5:1.5b-instruct-q4_K_M", 1.0, 1.5,
     "fast enough to iterate on the prompt; too small to trust for marks"),
]

# What Windows and a browser hold even when you are trying to be good.
# Used to estimate what would be free if you closed things, so the table
# does not condemn a model on the strength of your current tab count.
OS_RESERVE_GB = 2.5


# ------------------------------------------------------------- machine


def memory_gb():
    """(total, available) in GB, or (None, None) if we cannot tell.

    Windows first, because that is what this project runs on, then the
    Linux path so the file is not a lie on a different machine.
    """

    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return (status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30)
    except Exception:
        pass

    try:
        info = {}
        with open("/proc/meminfo") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                info[name] = float(rest.strip().split()[0]) / 2**20
        return (info["MemTotal"], info.get("MemAvailable", info["MemFree"]))
    except Exception:
        return (None, None)


def report_machine():
    cores = os.cpu_count() or 0
    total, available = memory_gb()
    free_disk = shutil.disk_usage(Path.home()).free / 2**30

    print("=== this machine ===")
    print(f"  logical CPUs    {cores}")
    if total:
        print(f"  RAM             {total:.1f} GB total, "
              f"{available:.1f} GB available now")
    else:
        print("  RAM             could not determine")
    print(f"  free disk       {free_disk:.0f} GB")

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    has_nvidia = shutil.which("nvidia-smi") is not None
    print(f"  NVIDIA GPU      {'yes' if has_nvidia else 'no'}"
          + (f" (CUDA_VISIBLE_DEVICES={gpu})" if gpu else ""))
    if not has_nvidia:
        print("                  -> CPU inference. This is the slow path, and")
        print("                     it is why the tier is queued and resumable.")
    return cores, total, available, free_disk


def report_models(total, available, free_disk):
    """Size each model against this machine.

    Judged against what would be free with your applications closed, not
    against what is free while you read this - otherwise the verdict is
    a function of how many browser tabs are open. Both numbers are
    printed so the difference is visible.
    """

    print("\n=== models, against this machine's RAM ===")
    if total is None:
        print("  (cannot size these without a RAM reading)")
        return

    budget = max(available, total - OS_RESERVE_GB)
    print(f"  budget: {budget:.1f} GB "
          f"(RAM {total:.1f} GB less ~{OS_RESERVE_GB:.1f} GB for the OS; "
          f"{available:.1f} GB free right now)")
    if budget > available + 0.5:
        print(f"  -> assumes you close your browser and editor first. Run it "
              f"now against\n     the {available:.1f} GB actually free and "
              f"every row below drops one grade.")
    print()

    for tag, download, resident, note in MODELS:
        headroom = budget - resident
        if headroom > 1.0:
            mark = "ok    "
        elif headroom > 0:
            mark = "tight "
        else:
            mark = "SWAPS "
        disk = "" if free_disk > download + 2 else "  ! not enough free disk"
        print(f"  [{mark}] {tag:<32} {resident:.1f} GB resident, "
              f"{download:.1f} GB download{disk}")
        print(f"           {note}")

    print("\n  'SWAPS' does not mean it refuses to run. It means it runs from")
    print("  the page file, which can be 5-20x slower while looking exactly")
    print("  like a model that is merely slow. That is what --time-it is for.")


def report_ollama():
    print("\n=== ollama ===")
    on_path = shutil.which("ollama")
    print(f"  installed       {'yes, ' + on_path if on_path else 'not on PATH'}")

    host = llm_local.OLLAMA_HOST
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as response:
            tags = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        print(f"  server          not responding at {host} ({error.reason})")
        print("                  start it with:  ollama serve")
        return []
    except Exception as error:
        print(f"  server          unexpected reply from {host}: {error}")
        return []

    names = sorted(m.get("name", "?") for m in tags.get("models", []))
    print(f"  server          up at {host}")
    print(f"  models pulled   {', '.join(names) if names else 'none'}")
    return names


# ------------------------------------------------------------- the queue


def report_queue():
    print("\n=== the queue ===")
    queue = paths.OUT_DIR / "queue_llm.jsonl"
    if not queue.exists():
        print(f"  {queue} does not exist yet")
        print("  run:  python marking/src/grade.py --all")
        return None, 0

    records = [json.loads(line) for line in open(queue, encoding="utf-8")
               if line.strip()]
    lengths = sorted(len(llm_prompt.SYSTEM_PROMPT)
                     + len(llm_prompt.build_prompt(r)) for r in records)
    median = lengths[len(lengths) // 2]

    out = paths.OUT_DIR / "local_verdicts.jsonl"
    done, models = llm_local.load_done(out)
    remaining = [r for r in records
                 if llm_prompt.record_key(r) not in done]

    print(f"  {len(records)} items queued")
    print(f"  {len(done)} already marked locally"
          + (f" by {', '.join(sorted(models))}" if models else ""))
    print(f"  {len(remaining)} remaining")
    print(f"  prompt size     ~{median // 4} tokens median, "
          f"~{lengths[-1] // 4} max")
    if lengths[-1] // 4 > llm_local.NUM_CTX - llm_local.MAX_NEW_TOKENS:
        print(f"  ! the longest prompt may not fit in num_ctx="
              f"{llm_local.NUM_CTX}; raise --num-ctx")
    return remaining, len(remaining)


def time_one(args, remaining):
    """Mark one real item and extrapolate. Writes no verdict."""

    if not remaining:
        print("\nnothing left unmarked to time against")
        return

    print(f"\n=== timing one real item on {args.model} ===")
    backend = llm_local.OllamaBackend(args.model, num_ctx=args.num_ctx)
    print(" ", backend.preflight())
    record = remaining[0]
    print(f"  item: {record['booklet_id']} {record['question']} "
          f"[{record['marks_available']} marks]")
    print("  (the first call also loads the model into RAM, so this is a "
          "pessimistic\n   single sample - the per-item rate settles lower)")

    started = time.time()
    reply = backend.complete(llm_prompt.SYSTEM_PROMPT,
                             llm_prompt.build_prompt(record))
    elapsed = time.time() - started
    verdict = llm_prompt.parse_reply(reply)

    print(f"\n  {elapsed:.0f}s for one item")
    print(f"  verdict: {verdict.get('verdict')}  "
          f"marks: {verdict.get('marks')}  "
          f"reason: {str(verdict.get('reason'))[:60]}")
    if verdict.get("verdict") == "decline" and "JSON" in str(verdict.get("reason")):
        print("  ! the model did not return usable JSON. If that repeats across")
        print("    a --limit 25 sample, the model is too small for this task.")

    total = elapsed * len(remaining)
    print(f"\n  {len(remaining)} remaining x {elapsed:.0f}s = "
          f"~{llm_local.fmt_duration(total)} for the full queue")
    for sample in (25, 100, 250):
        if sample < len(remaining):
            print(f"    --limit {sample:<4} ~"
                  f"{llm_local.fmt_duration(elapsed * sample)}")


def main():
    parser = argparse.ArgumentParser(description="Can this machine run the tier?")
    parser.add_argument("--time-it", action="store_true",
                        help="mark one real item and extrapolate")
    parser.add_argument("--model", default=llm_local.DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--num-ctx", type=int, default=llm_local.NUM_CTX)
    args = parser.parse_args()

    _, total, available, free_disk = report_machine()
    report_models(total, available, free_disk)
    report_ollama()
    remaining, count = report_queue()

    if args.time_it:
        time_one(args, remaining or [])
    else:
        print("\nNext:")
        print("  python marking/src/setup_local.py --time-it     # measure, do not guess")
        print(f"  python marking/src/llm_local.py --limit 25       # a sample of {count}")

    print("\nFull setup, including what a smaller model costs you: "
          "docs/LOCAL_SETUP.md")


if __name__ == "__main__":
    main()
