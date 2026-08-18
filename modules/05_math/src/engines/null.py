"""
No recogniser at all.

Runs everything except the reading: routing is consumed, crops are
prepared, expressions are cut, geometry is mapped back to the page
and the output file is written with empty readings.

WHY IT EARNS A FILE
-------------------
Two reasons, both practical.

The stage weighs 1.4GB of model and 5-10s per expression. Checking
that the plumbing works - that the routed JSON parses, that every
crop resolves, that the page coordinates come out right - should not
require either. With this engine the whole corpus runs in seconds.

And it separates the two things that can go wrong. If output looks
wrong under `--engine none`, the fault is in this module. If it only
looks wrong with a real engine, the fault is the recogniser. On a
corpus where the recogniser is currently the weak link, keeping that
line clear matters.
"""


class NullEngine:

    name = "none"

    def describe(self):

        return {
            "engine": self.name,
            "model": None,
            "output": "none",
        }

    def read(self, image):

        return {"latex": "", "confidence": 0.0, "seconds": 0.0}
