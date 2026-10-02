"""
The one part of the pipeline that needs a GPU, behind one interface.

Every other stage - prepare, assemble, grade - is plain Python and runs
on a laptop in seconds. Reading the page is the only step that wants a
7B vision model, so it is the only step that is swappable. Everything
downstream consumes Markdown and neither knows nor cares which of these
produced it.

    CachedReader   markdown already on disk           instant, no model
    ServerReader   an OpenAI-compatible VLM endpoint  needs a URL
    LocalReader    llama.cpp + a GGUF model           needs no network

WHY THIS ABSTRACTION EXISTS AT ALL
-----------------------------------
The measurement that drove the whole rewrite - 0.099 CER against the
line pipeline's 0.463 - was taken with Qwen2.5-VL-7B on a hosted T4,
because this laptop is an i5-7200U with 2.8GB free and cannot run it.
That is fine for a corpus pass done once, and useless as an answer to
"what happens when I hand the system a new booklet".

So the pipeline does not depend on where the model runs. Point it at a
cache for a demo of work already done, at a server for a real run, or
at a local GGUF when there is no network. The CER only holds for the
model that produced the markdown, so every reader records which one it
was - see `provenance`.

WHAT EACH IS ACTUALLY FOR
-------------------------
CACHED is not a toy. A corpus pass is hours of GPU time and the result
does not change, so re-reading pages that have already been read is
waste. It is also what makes a demo instant and repeatable.

SERVER is the honest production answer. vLLM or llama.cpp's server on
any machine with a GPU - a lab box, a spot instance at about 20p an
hour - and this laptop drives it over HTTP.

LOCAL is the fallback with no network and no GPU. It works and it is
slow: a 3B at Q4 is roughly 2GB of RAM and minutes per page on this
CPU, against seconds on a T4. Use it for one booklet, not for 1,231
pages, and note the 3B scored 0.229 CER against the 7B's 0.099.
"""

import base64
import json
import os
import time
from pathlib import Path


class Reader:
    """Takes an image path and its page id, returns Markdown.

    `key` is passed rather than derived from the filename because they
    are not the same thing: on disk a page is `page_03.png`, and its
    identity in every manifest, cache and benchmark is `s01_c2_p03`.
    A cache keyed on the filename silently matches nothing and reports
    an empty booklet, which is how this was found.
    """

    name = "reader"

    def read(self, image_path, key):
        raise NotImplementedError

    def provenance(self):
        """What produced this text - recorded alongside every run.

        A CER is a property of a model, not of a pipeline, so output
        that does not say which model made it cannot be interpreted
        later.
        """
        return {"reader": self.name}


class CachedReader(Reader):
    """Markdown someone already produced, keyed by page id."""

    name = "cached"

    def __init__(self, cache_dir, engine="unknown"):
        self.cache_dir = Path(cache_dir)
        self.engine = engine

        if not self.cache_dir.exists():
            raise SystemExit(f"cache not found: {self.cache_dir}")

    def read(self, image_path, key):
        target = self.cache_dir / f"{key}.md"

        if not target.exists():
            return None            # caller decides: skip, or fall back

        return target.read_text(encoding="utf-8")

    def provenance(self):
        return {"reader": self.name, "cache": str(self.cache_dir),
                "engine": self.engine}


class ServerReader(Reader):
    """Any OpenAI-compatible /chat/completions endpoint that takes images.

    vLLM, llama.cpp --server, LM Studio, Ollama and the hosted APIs all
    speak this, so the same code points at a lab GPU or a rented one
    without changing.
    """

    name = "server"

    def __init__(self, url, model, prompt, api_key=None, timeout=300):
        self.url = url.rstrip("/")
        self.model = model
        self.prompt = prompt
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "")
        self.timeout = timeout

    def read(self, image_path, key=None):

        import urllib.request

        encoded = base64.b64encode(
            Path(image_path).read_bytes()).decode("ascii")

        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1536,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                {"type": "text", "text": self.prompt},
            ]}],
        }

        request = urllib.request.Request(
            f"{self.url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key else {})},
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read())

        return body["choices"][0]["message"]["content"].strip()

    def provenance(self):
        return {"reader": self.name, "url": self.url, "model": self.model}


class LocalReader(Reader):
    """llama.cpp with a GGUF vision model, in-process. No network.

    Needs `pip install llama-cpp-python` and both GGUFs - the model and
    its mmproj (the vision projector, a separate file). Expect minutes
    per page on a CPU like this one; that is the price of no GPU.
    """

    name = "local"

    def __init__(self, model_path, mmproj_path, prompt, n_ctx=4096,
                 n_threads=None):
        self.model_path = str(model_path)
        self.mmproj_path = str(mmproj_path)
        self.prompt = prompt
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, (os.cpu_count() or 2))
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return self._llm

        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        except ImportError:
            raise SystemExit(
                "llama-cpp-python is not installed.\n"
                "  pip install llama-cpp-python\n"
                "and download both GGUFs (model + mmproj) for the vision "
                "model you want to run.")

        handler = Qwen25VLChatHandler(clip_model_path=self.mmproj_path)

        self._llm = Llama(
            model_path=self.model_path,
            chat_handler=handler,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )

        return self._llm

    def read(self, image_path, key=None):

        llm = self._load()

        uri = Path(image_path).resolve().as_uri()

        response = llm.create_chat_completion(
            temperature=0,
            max_tokens=1536,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": self.prompt},
            ]}],
        )

        return response["choices"][0]["message"]["content"].strip()

    def provenance(self):
        return {"reader": self.name, "model": Path(self.model_path).name}


def build(kind, prompt, **options):
    """One place that knows every reader, so callers name a string."""

    if kind == "cached":
        return CachedReader(options["cache_dir"],
                            options.get("engine", "unknown"))

    if kind == "server":
        return ServerReader(options["url"], options.get("model", "default"),
                            prompt, options.get("api_key"))

    if kind == "local":
        return LocalReader(options["model_path"], options["mmproj_path"],
                           prompt)

    raise SystemExit(f"unknown reader: {kind}")
