"""
Read a margin mark with TrOCR, so marker_text can judge what it says.

Kept in its own module because it is the only thing in 07_reconstruct
that needs torch. reconstruct.py imports marker_text unconditionally
and this lazily, so the geometry path still runs on a machine without
a 2.5GB install.

TrOCR is the recogniser 04_ocr already chose for this corpus, so this
adds no new dependency to the repo as a whole - see
modules/04_ocr/README.md for why it beat the alternatives on this
handwriting.

CROPS ARE NOT LINES
-------------------
TrOCR base is trained on full handwritten LINES, and a question number
is one to four glyphs. That mismatch is the risk in this approach, so
it is measured rather than assumed: `validate.py` reads all 282
candidates from the 10-student survey and renders what each one came
back as, next to the crop it came from.

Two things help it, both cheap:

  - pad the crop. A glyph flush to the border reads worse than one with
    white around it, and the model has seen line images that always
    have margins.
  - upscale small crops. A 40px-tall mark is far below the 384px the
    processor resizes to, so it arrives blurred; scaling up first with
    a smooth interpolation gives the resize something to work with.
"""

from functools import lru_cache

import cv2
import numpy as np

MODEL_NAME = "microsoft/trocr-base-handwritten"

# White border added around every crop, in pixels of the ORIGINAL crop.
# See the module note - the model expects margins.
PAD_PX = 12

# Any crop shorter than this is scaled up to it before the processor
# sees it. 384 is TrOCR's own input height; going straight there from
# ~40px over-smooths, so this sits between the two.
TARGET_HEIGHT = 160

PAD_VALUE = 255


@lru_cache(maxsize=1)
def _model():
    """Loaded once per process. Heavy - about 1.3GB resident.

    Forced OFFLINE. Measured here: with the hub reachable-but-slow,
    from_pretrained sat for over ten minutes on network round-trips
    before touching the cached weights, and looked exactly like a hang.
    Offline it is 46s, of which the actual weight load is 3s. The model
    has to be in the local cache either way - this only stops
    transformers asking the network whether it has changed.

    Set explicitly rather than left to the caller's environment: a
    silent ten-minute stall is a bad default for a step that otherwise
    takes seconds. To deliberately re-fetch, clear the HF cache.
    """

    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.eval()

    return processor, model


def strip_ring(crop, ink_threshold=180):
    """Remove an enclosing hand-drawn circle, returning its interior.

    Students circle question numbers, and TrOCR reads the circle as a
    character - measured on student_01, six circled marks came back as
    "0", "0", "6.", "0", "a b", where the ground truth was (1), (2a),
    (2b), (2c), (3b). The one uncircled mark in the same sample, "4b.",
    read correctly. The ring is the single biggest error source, and it
    is trivial to find: it is a component whose bounding box fills the
    crop and which encloses a hole big enough to hold the glyphs.

    Returns the crop unchanged when there is no such ring, so uncircled
    conventions - "2a)", "3b.", "Q4b)" - pass straight through.
    """

    if crop is None or crop.size == 0:
        return crop

    mask = (crop < ink_threshold).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return crop

    height, width = mask.shape
    area = height * width

    best = None

    for index, contour in enumerate(contours):

        # a hole is a contour WITH a parent, in RETR_CCOMP terms
        if hierarchy[0][index][3] < 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # the interior has to be big enough to actually hold the glyphs,
        # and not so big it is the whole crop (which would mean there
        # was no ring, just a border of noise)
        if w * h < 0.12 * area or w * h > 0.92 * area:
            continue

        if best is None or w * h > best[2] * best[3]:
            best = (x, y, w, h)

    if best is None:
        return crop

    x, y, w, h = best

    # a hair of inset, so the ring's own stroke does not survive at the
    # edge of the returned crop and read as a bracket
    inset = 2
    x0 = min(x + inset, x + w - 1)
    y0 = min(y + inset, y + h - 1)
    x1 = max(x + w - inset, x0 + 1)
    y1 = max(y + h - inset, y0 + 1)

    return crop[y0:y1, x0:x1]


def prepare(crop):
    """Pad and upscale a greyscale crop into something TrOCR can read."""

    if crop is None or crop.size == 0:
        return None

    crop = strip_ring(crop)

    if crop is None or crop.size == 0:
        return None

    crop = cv2.copyMakeBorder(
        crop, PAD_PX, PAD_PX, PAD_PX, PAD_PX,
        cv2.BORDER_CONSTANT, value=PAD_VALUE,
    )

    height, width = crop.shape[:2]

    if height < TARGET_HEIGHT:
        scale = TARGET_HEIGHT / height
        crop = cv2.resize(
            crop, (max(1, int(width * scale)), TARGET_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )

    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

    return crop


def read_batch(crops, batch_size=8, progress=None):
    """Read many crops at once. Returns a list of strings, one per crop.

    Batched because per-call overhead dominates at this crop size - the
    images are tiny and the model is not.

    `progress` is called as progress(done, total) after each batch. This
    run is minutes long on CPU and silent progress is indistinguishable
    from a hang, so callers are expected to pass one.
    """

    import torch

    processor, model = _model()

    prepared = [prepare(c) for c in crops]
    texts = [None] * len(prepared)

    usable = [(i, p) for i, p in enumerate(prepared) if p is not None]

    for start in range(0, len(usable), batch_size):

        window = usable[start:start + batch_size]

        pixels = processor(
            images=[p for _, p in window], return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            ids = model.generate(pixels, max_new_tokens=8)

        decoded = processor.batch_decode(ids, skip_special_tokens=True)

        for (index, _), text in zip(window, decoded):
            texts[index] = text

        if progress:
            progress(min(start + batch_size, len(usable)), len(usable))

    return ["" if t is None else t for t in texts]


def read(crop):
    """Read a single crop."""

    return read_batch([crop])[0]
