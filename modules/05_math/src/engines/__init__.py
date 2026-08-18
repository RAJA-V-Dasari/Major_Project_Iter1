"""
The recogniser is a swappable part, so it lives behind one interface.

An engine takes ONE prepared expression image and returns the LaTeX
it reads, plus a confidence. That is the whole contract:

    engine.read(image) -> {"latex": str, "confidence": float,
                           "seconds": float}

WHY THIS IS PLUGGABLE RATHER THAN JUST IMPORTED
-----------------------------------------------
No recogniser tested here is good enough on this corpus to build the
rest of the module around (see the measurements in 05_math/README).
Which one wins is still an open question, and the answer will change
- fine-tuning on this handwriting is the obvious next step, and a
fine-tuned checkpoint should drop in without touching math_ocr.py.

The comparison itself is also a deliverable: "formula model versus
handwriting model on the same prepared images" is a question this
project has to answer, and it can only be answered if both can be
run under identical conditions.
"""

from importlib import import_module


ENGINES = {
    # image -> LaTeX. A formula recogniser: knows fractions,
    # superscripts and 2-D layout.
    "sumen": ("sumen", "Sumen"),

    # image -> plain text. A handwriting recogniser: knows this kind
    # of pen, but has no notion of a superscript.
    "trocr": ("trocr", "TrOCR"),

    # no model at all. Runs the whole stage - routing, preparation,
    # geometry, output - without a download, so the pipeline can be
    # exercised offline and the preparation checked on its own.
    "none": ("null", "NullEngine"),
}


def load(name, **kwargs):

    if name not in ENGINES:
        raise SystemExit(
            f"unknown engine {name!r} - choose from {', '.join(ENGINES)}"
        )

    module_name, class_name = ENGINES[name]

    module = import_module(f"engines.{module_name}")

    return getattr(module, class_name)(**kwargs)
