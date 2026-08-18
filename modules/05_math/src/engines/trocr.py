"""
TrOCR: a handwriting recogniser, here as the control.

    microsoft/trocr-base-handwritten - ViT encoder, RoBERTa decoder.

WHY A TEXT MODEL IS IN A MATH MODULE
------------------------------------
Because it is not obvious that a formula model is the right tool for
THIS corpus, and the question is cheap to settle.

These are computer-networks scripts. The mathematics in them is
almost entirely linear - "header len = 6x4 = 24 bytes",
"= 1400 - 30", "Total length field = (0038)_16". There is hardly any
two-dimensional structure: few real fractions, no integrals, no
matrices. What there IS, everywhere, is difficult handwriting.

A formula model buys 2-D layout and pays for it by having been
trained mostly on typeset LaTeX. A handwriting model buys the pen
and pays for it by flattening every superscript. Which trade is
better here is an empirical question, so both run on the same
prepared images and the answer goes in 05_math/README.

Its output is plain text, not LaTeX. Downstream that difference
matters, so it is recorded per engine in the output rather than
assumed.
"""

import time

import numpy as np
import torch
from PIL import Image


MODEL_ID = "microsoft/trocr-base-handwritten"

MAX_NEW_TOKENS = 64


def _tokenizer():
    """
    Build the fast tokenizer from the model's own vocabulary files.

    AutoProcessor cannot do this. The repo ships a SLOW tokenizer -
    vocab.json plus merges.txt, no tokenizer.json - and transformers
    5.x no longer carries the converter for that case, so loading it
    the normal way fails with "Couldn't instantiate the backend
    tokenizer". Installing sentencepiece, which the error suggests,
    does not help: this is byte-level BPE, not a sentencepiece model.

    So the byte-level BPE is assembled here from those two files
    directly. It is the model's own vocabulary, not a substitute -
    50265 tokens, matching config.decoder.vocab_size exactly.
    """

    from huggingface_hub import hf_hub_download
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer(
        hf_hub_download(MODEL_ID, "vocab.json"),
        hf_hub_download(MODEL_ID, "merges.txt"),
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", cls_token="<s>", sep_token="</s>",
        mask_token="<mask>",
    )


class TrOCR:

    name = "trocr"

    def __init__(self, beams=1, max_new_tokens=MAX_NEW_TOKENS):

        from transformers import AutoImageProcessor, VisionEncoderDecoderModel

        self.images = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.tokenizer = _tokenizer()
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).eval()

        self.beams = beams
        self.max_new_tokens = max_new_tokens

    def describe(self):

        return {
            "engine": self.name,
            "model": MODEL_ID,
            "output": "text",
            "beams": self.beams,
        }

    def read(self, image):

        started = time.time()

        pixels = self.images(
            images=Image.fromarray(image).convert("RGB"), return_tensors="pt",
        ).pixel_values

        with torch.no_grad():
            generated = self.model.generate(
                pixels,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.beams,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self.tokenizer.decode(
            generated.sequences[0], skip_special_tokens=True,
        )

        scores = self.model.compute_transition_scores(
            generated.sequences, generated.scores, normalize_logits=True,
        )[0]

        scores = scores[torch.isfinite(scores)]

        confidence = (
            round(float(np.exp(scores.mean().item())), 4)
            if scores.numel() else 0.0
        )

        return {
            "latex": text.strip(),
            "confidence": confidence,
            "seconds": round(time.time() - started, 2),
        }
