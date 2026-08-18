"""
Sumen: an image-to-LaTeX formula recogniser.

    hoang-quoc-trung/sumen-base - Donut-Swin encoder, mBART decoder,
    349M parameters, ~1.4GB on disk.

WHY THIS ONE
------------
It is the only image-to-LaTeX model tested here that is both
transformers-native and trained on handwritten formulas as well as
printed ones. The obvious alternative, breezedeus/pix2text-mfr, is
a tenth of the size and much faster, but ships only as ONNX and
would drag in onnxruntime and optimum for a model this project will
likely fine-tune anyway.

Its encoder takes a 224x468 image - a wide, short frame, which is
the shape of a line of working, so a prepared expression arrives
without being squashed.

WHAT THE CONFIDENCE MEANS - AND WHAT IT DOES NOT
------------------------------------------------
The mean per-token probability of the sequence the decoder chose.
Unlike the router's simulated confidence this is a real quantity.

DO NOT USE IT TO TRIAGE. It was added here on the assumption that a
model reading handwriting it cannot parse would at least be unsure
about it, and that assumption is wrong on this corpus. Measured over
46 expressions, mean confidence 0.87, and the wrong readings score
as high as the right ones:

    = 256 - 1      read exactly right          0.96
    = 69           read as \\frac{\\varepsilon
                   \\delta 9}{\\varepsilon}      0.94

A formula model asked to read this handwriting does not fail loudly.
It fabricates fluent, well-formed LaTeX and is confident about it.
The number is still recorded, because it is worth knowing that it
does not separate - but anything built on top of this stage needs a
different signal, and the honest one today is a human looking at
preview/.

CPU ONLY, AND THAT IS FINE
--------------------------
There is no GPU on this machine and the corpus is not large: ~5-10s
per expression, and a booklet's worth of maths is a few hundred
expressions. Greedy decoding by default - beams multiply the cost
and, on handwriting this hard, mostly produce a more confident wrong
answer.
"""

import time

import numpy as np
import torch
from PIL import Image


MODEL_ID = "hoang-quoc-trung/sumen-base"

MAX_NEW_TOKENS = 96


class Sumen:

    name = "sumen"

    def __init__(self, beams=1, max_new_tokens=MAX_NEW_TOKENS):

        # Imported here rather than at module scope so that
        # `--engine none` never pays for loading transformers.
        from transformers import AutoProcessor, VisionEncoderDecoderModel

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).eval()

        self.beams = beams
        self.max_new_tokens = max_new_tokens

    def describe(self):

        return {
            "engine": self.name,
            "model": MODEL_ID,
            "output": "latex",
            "beams": self.beams,
        }

    def read(self, image):

        started = time.time()

        # The processor wants RGB; a prepared expression is greyscale.
        pixels = self.processor(
            Image.fromarray(image).convert("RGB"), return_tensors="pt",
        ).pixel_values

        with torch.no_grad():
            generated = self.model.generate(
                pixels,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.beams,
                output_scores=True,
                return_dict_in_generate=True,
            )

        latex = self.processor.tokenizer.decode(
            generated.sequences[0], skip_special_tokens=True,
        )

        return {
            "latex": latex.strip(),
            "confidence": self._confidence(generated),
            "seconds": round(time.time() - started, 2),
        }

    def _confidence(self, generated):
        """Mean probability of the tokens the decoder actually chose."""

        scores = self.model.compute_transition_scores(
            generated.sequences,
            generated.scores,
            normalize_logits=True,
        )[0]

        scores = scores[torch.isfinite(scores)]

        if scores.numel() == 0:
            return 0.0

        return round(float(np.exp(scores.mean().item())), 4)
