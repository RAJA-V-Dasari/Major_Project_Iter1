"""
The GPU, as a URL. Deploy once, then the laptop drives the whole pipeline.

    modal deploy modules/02_read/src/modal_vllm.py
    python modules/05_pipeline/src/run_booklet.py --all \
        --reader server --url https://<workspace>--mp-read-server.modal.run/v1

That is the entire change. `ServerReader` in `05_pipeline/src/readers.py`
already speaks OpenAI `/chat/completions`, and `run_booklet.py` already
takes `--url`, so nothing downstream of this file moves. The notebook
round trip - zip the corpus, upload to Drive, babysit a session, download
the markdown - is replaced by an HTTPS call.

WHY A SERVER AND NOT A MODAL FUNCTION PER PAGE
----------------------------------------------
Fanning out with `Function.map()` over 1,000 pages would be faster, but
it would put a second copy of the reading logic in this repo, and the
prompt would have to be restated here. The prompt lives once, in
`make_colab_notebook.py`, and `run_booklet.py` imports it. A vLLM server
is prompt-agnostic: it never sees the prompt until the client sends it,
so deploying this file cannot make the prompt drift from the one the
0.099 CER was measured with.

WHAT THIS COSTS
---------------
Nothing while idle. `scaledown_window` stops the container after ten
minutes of silence and Modal bills per second of actual run time, so the
meter only moves while pages are in flight. Modal's Starter tier carries
$30/month of credit that does not roll over. A T4 is about $0.59/hour
there, an L40S several times that; a corpus pass is hours, not days, so
the free credit covers the work left in this project with room to spare.

Cold start is the real cost: first deploy downloads ~16GB of weights into
the volume, and a wake-from-idle is tens of seconds once they are cached.
Batch the pages; do not send one and walk away.

WHY L40S AND NOT T4
-------------------
Because 48GB ends the quantization. Every number in DONE.md was measured
with the 7B in 4-bit, because that is what fits in a T4's 16GB, and 4-bit
is a quality ceiling nobody chose - it was the hardware. In bf16 the same
model is ~16.5GB of weights and has room for the KV cache and the vision
tower beside it. Dropping to `gpu="T4"` below still works and still needs
`--quantization` back; it is cheaper and it is not the same model.

THE PIXEL BUDGET IS LOAD-BEARING
--------------------------------
`--mm-processor-kwargs` carries the same 256-1024 patch budget as
`make_colab_notebook.py`. Read the Colab OOM in DONE.md before touching
it: the budget was once set on the chat message instead of the processor,
no resize happened, a page became ~4,437 visual tokens and attention
asked a T4 for 18.85 GiB. vLLM honours it here because it sits on the
processor. Changing it changes what the model sees, which invalidates the
CER the same way editing the prompt would.

PRIVACY
-------
A page is a student's handwriting and this sends it over the network, so
the endpoint is not left open. vLLM's own `--api-key` comes from a Modal
secret and `ServerReader` sends it as a bearer token:

    modal secret create mp-read-key VLM_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(32))")

then put the same value in `VLM_API_KEY` in the local environment, which
`ServerReader` already reads. `page_01` exclusion is unaffected - it is
enforced in `run_booklet.py`, upstream of every reader, and stays there.
"""

import json
import os

import modal

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
VLLM_PORT = 8000
MINUTES = 60

# The same budget as make_colab_notebook.py, in 28x28 patches.
MIN_PATCHES, MAX_PATCHES = 256, 1024

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.11.0",
        "huggingface_hub[hf_transfer]==0.34.4",
        "qwen-vl-utils==0.0.11",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

# Weights survive between deploys and between wakes, so the ~16GB
# download happens once rather than on every cold start.
hf_cache_vol = modal.Volume.from_name("mp-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("mp-vllm-cache", create_if_missing=True)

app = modal.App("mp-read")


@app.server(
    image=vllm_image,
    gpu="L40S",
    scaledown_window=10 * MINUTES,
    startup_timeout=20 * MINUTES,
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("mp-read-key")],
    port=VLLM_PORT,
    target_concurrency=8,
    unauthenticated=True,          # vLLM's own --api-key is the gate
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm", "serve", MODEL_NAME,
            # run_booklet.py defaults --model to this exact string
            "--served-model-name", MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--api-key", os.environ["VLM_API_KEY"],
            "--dtype", "bfloat16",
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
            "--limit-mm-per-prompt", json.dumps({"image": 1}),
            "--mm-processor-kwargs", json.dumps({
                "min_pixels": MIN_PATCHES * 28 * 28,
                "max_pixels": MAX_PATCHES * 28 * 28,
            }),
            "--uvicorn-log-level=warning",
        ]

        self.process = subprocess.Popen(cmd)
