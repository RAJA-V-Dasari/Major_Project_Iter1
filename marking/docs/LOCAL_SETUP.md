# Running the model tier on this machine

The model tier has three routes: a free Colab T4, this machine, or
Claude in session. This page is the local route, which takes the one
step that did not run locally and runs it on the same machine as
everything else. The marking half was first built for Colab, and was
briefly kept as two folders, one per route. It is one package now.

If you only want the commands, skip to
[The short version](#the-short-version).

---

## What going local changes

| | Colab route | local route |
|---|---|---|
| where the model runs | free Colab T4 | this machine, on CPU |
| how it is reached | upload a notebook, download a file | `python marking/src/llm_local.py` |
| model | Qwen2.5-7B-Instruct, 4-bit (bitsandbytes/nf4) | Qwen2.5-7B-Instruct, GGUF Q4_K_M |
| prompt | defined in `make_llm_notebook.py` | defined in `llm_prompt.py`, **byte-identical** |
| verdicts land in | `data/marking/grade_llm_verdicts.jsonl` | `data/marking/local_verdicts.jsonl` |
| the checks after it | `apply_verdicts.py` | unchanged |

Three of those rows are the whole design:

**The prompt moved but did not change.** It lives in
[`src/llm_prompt.py`](../src/llm_prompt.py) now, and both the local
runner and the notebook generator import it — the notebook gets it by
`inspect.getsource`, so the two cannot drift. It was verified
byte-identical to the original and checked against all 740 queue
records, including the 275 that carry a chain note and the 438 that
carry a figure note.

**The verdicts land beside the recorded run, never on top of it.**
`data/marking/grade_llm_verdicts.jsonl` is the Colab run that the project's
published numbers come from. A local run writes `local_verdicts.jsonl`
instead, and `llm_local.py` refuses to append to a verdicts file written
by a different model unless you say `--allow-mixed-models`. That refusal
costs nothing and happens before the server is even contacted.

**Everything downstream is untouched.** `apply_verdicts.py` still
re-checks every quote against the student's answer, still refuses a zero
on a drawing-backed item (`zero_blocked`), and still will not overwrite
an item a cheaper tier already settled. Those checks were always
downstream of the model and do not care which machine produced the row.
Moving the tier was never going to weaken them — that was the point of
making it a queue on disk in the first place.

---

## Why not `transformers` and `bitsandbytes`

The notebook loads the model in 4-bit through bitsandbytes, which
requires CUDA. Without an NVIDIA GPU that path does not run slowly, it
does not run: bitsandbytes has no usable CPU 4-bit kernel, and the
unquantised 7B model is about 28 GB in fp32 — four times this laptop's
entire RAM.

So the local tier goes through a GGUF runtime instead (llama.cpp, either
via Ollama or directly). That is what quantised CPU inference is built
for: the weights are memory-mapped, and Q4_K_M is the closest widely
available analogue of the notebook's nf4 double-quant.

**The consequence is that local verdicts are not bit-identical to the
Colab ones.** Same weights, different quantisation, different kernel.
That is not a defect to hide — it is a thing worth measuring, and
`llm_local.py --compare` prints the confusion matrix between the two
runs on the items they share.

---

## What this machine can actually do

Run this first. It measures rather than assumes:

```bash
python marking/src/setup_local.py
```

On the laptop this copy was made on it reports:

```
  logical CPUs    4          (2 physical — i5-7200U)
  RAM             7.9 GB total
  NVIDIA GPU      no
  740 items queued, prompt ~1100 tokens median, ~1825 max
```

Which gives a RAM budget of about **5.4 GB** with the browser and editor
closed. Against that:

| model | resident | download | verdict on this machine |
|---|---|---|---|
| `qwen2.5:7b-instruct-q4_K_M` | 5.6 GB | 4.7 GB | **does not fit** — runs from the page file |
| `qwen2.5:3b-instruct-q4_K_M` | 2.6 GB | 1.9 GB | fits comfortably |
| `qwen2.5:1.5b-instruct-q4_K_M` | 1.5 GB | 1.0 GB | fits, but too small to trust for marking |

A model that does not fit **does not refuse to load.** It swaps, and the
run looks exactly like a run that is merely slow while being five to
twenty times slower than it should be. That is the failure this page
exists to warn about, and the reason `--time-it` exists:

```bash
python marking/src/setup_local.py --time-it
```

It marks one real item off the real queue and multiplies by what is
left. An estimate from your machine beats any number written here.

### The honest expectation

Two physical cores, a ~1,100-token prompt and 740 items is **tens of
hours, not minutes** — and that is with a model that fits. This is not a
defect in the port; it is what a 7B model costs on a 2017 ultrabook.
Three ways to live with it, in order of how much they cost you
scientifically:

1. **`--limit`.** Mark a sample overnight, apply it, and report the
   sample honestly as a sample. Resumable, so tomorrow's run continues.
2. **A smaller model.** `--model qwen2.5:3b-instruct-q4_K_M` is roughly
   2.5× faster and fits in RAM. It is a **different model**, so say so
   wherever you report its numbers — see [If you use a smaller
   model](#if-you-use-a-smaller-model).
3. **Keep Colab for the full run.** `make_llm_notebook.py` still works
   and is still the fastest path to all 740 items. Running locally and
   on Colab are not mutually exclusive, and `--compare` is more
   interesting when you have both.

---

## The short version

### 1. Install Ollama

Download from [ollama.com/download](https://ollama.com/download) and run
the installer. Then, in a terminal:

```bash
ollama --version
```

### 2. Check where the models will land

Ollama stores models in `%USERPROFILE%\.ollama\models` —
`C:\Users\<you>\.ollama\models`. **On this machine that is already
outside the synced tree and needs no change**, which was checked rather
than assumed:

```
USERPROFILE : C:\Users\Murty
Desktop     : C:\Users\Murty\OneDrive\Desktop   <- redirected into OneDrive
~/.ollama   : C:\Users\Murty\.ollama            <- NOT in OneDrive
```

This project lives under `OneDrive\Desktop` because Desktop is a
redirected known folder. `.ollama` is not a known folder, so it stays
local and a 4.7 GB model will not be uploaded anywhere.

Worth a look anyway if your setup differs, because the failure is
expensive and silent: a multi-gigabyte model inside a synced folder gets
uploaded, which is slow, can exhaust a quota, and is pointless — the
file is re-downloadable in one command. If your whole profile is
redirected, or you want the models on another drive, override it once:

```powershell
setx OLLAMA_MODELS "C:\ollama-models"
```

Then close and reopen your terminal so it takes effect.

### 3. Pull the model

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M      # matches the Colab run
ollama pull qwen2.5:3b-instruct-q4_K_M      # the practical choice here
```

### 4. Check before you commit hours to it

```bash
python marking/src/setup_local.py --time-it
python marking/src/llm_local.py --check
python marking/src/llm_local.py --dry-run           # see the exact prompt
```

### 5. Mark

```bash
python marking/src/llm_local.py --limit 25          # a sample first, always
python marking/src/llm_local.py                     # the rest, resumable
```

Every verdict is flushed as it is produced. Ctrl-C costs you one item,
and re-running picks up where it stopped.

### 6. Apply, exactly as before

```bash
python marking/src/apply_verdicts.py data/marking/local_verdicts.jsonl --dry-run
python marking/src/apply_verdicts.py data/marking/local_verdicts.jsonl
python marking/src/agreement.py
```

`--dry-run` first is not ceremony. It tells you how many awards will be
**rejected** for a quote that is not in the answer, and on the recorded
run that was 15.4% of them.

### 7. See what the change cost

```bash
python marking/src/llm_local.py --compare
```

---

## If you use a smaller model

A 3B model is not the model the project's published numbers came from,
and its verdicts are not interchangeable with them. If you run it:

- Keep them in their own file (`--out data/marking/local_3b_verdicts.jsonl`)
  so nothing silently mixes two models into one experiment.
- Say which model produced the marks wherever you report them. Every row
  carries a `model` field for exactly this reason, and
  `data/marking/verdict_audit.md` reports per-model.
- Expect the quote-rejection rate to be **worse**, not better. Quoting
  verbatim from a long passage is the part small models are worst at,
  and it is the thing `apply_verdicts.py` checks hardest. That rejection
  rate is the number to look at first — it is a property of the model,
  and the tier is only useful with the check around it.

The ladder's design already accounts for a weak model at this tier:
its awards must survive the quote check, and its zeros are refused
wherever it could not see the evidence. A smaller model makes the tier
settle less, not settle wrongly.

---

## The llama.cpp fallback

If installing Ollama is not an option, run a GGUF file in-process:

```bash
pip install llama-cpp-python
python marking/src/llm_local.py --backend llama-cpp --gguf C:\models\qwen2.5-7b-instruct-q4_k_m.gguf
```

Same prompt, same output file, same downstream. It is the fallback
rather than the default because `llama-cpp-python` builds from source on
Windows unless a matching wheel exists, and that is a longer detour than
installing Ollama.

---

## Troubleshooting

**`no Ollama server at http://localhost:11434`** — run `ollama serve`,
or start the Ollama app. Set `OLLAMA_HOST` if it listens elsewhere.

**`model not pulled`** — the error names the exact `ollama pull` command.

**It is far slower than `--time-it` predicted.** You are swapping. Close
the browser and re-run `python marking/src/setup_local.py` to see the real
budget, then drop to the 3B model.

**Every verdict comes back `decline` with "model produced no JSON".**
The model is not following the output format. Check `--dry-run` output
looks sane, then try a larger model — this is the characteristic failure
of going too small.

**Answers seem to be marked as absent when they are plainly present.**
Suspect truncation. `llm_local.py` sets `num_ctx=4096` explicitly
because Ollama's own default is 2048, and the longest prompt in this
queue is about 1,825 tokens plus a 300-token reply. A truncated prompt
loses the *end* of the user turn — which is the student's answer — so
the model marks an answer it was never shown. If you change models or
raise `MAX_NEW_TOKENS`, re-run `setup_local.py`; it warns when the
longest prompt no longer fits.

---

## See also

- [`PIPELINE.md`](PIPELINE.md) — every stage, and what each one proves
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the ladder, the asymmetry, the carve-outs
- [`../IMPROVEMENTS.md`](../IMPROVEMENTS.md) — what is still wrong
- [`src/llm_prompt.py`](../src/llm_prompt.py) — the prompt, and why it lives alone
