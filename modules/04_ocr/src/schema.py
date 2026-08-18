"""
The OCR output contract.

This is the single definition of what 04_ocr emits, and therefore of
what anything downstream may rely on. It is deliberately separate from
both the simulator and the real engine so that the two cannot drift:
simulate.py fills these records with synthetic text, the real engine
fills the same records with recognised text, and consumers cannot tell
the difference from the shape of the data.

THE SIMULATED FLAG IS NOT DECORATION
------------------------------------
Every file this module writes carries `simulated: true` until a real
engine has produced it. Synthetic transcripts that look exactly like
real ones are genuinely dangerous - somebody will otherwise quote an
accuracy number, or a student's "answer", that a random number
generator invented. Consumers should refuse to report results as real
while that flag is set, and validate_run() below fails loudly if the
flag is missing rather than assuming the friendly default.

UNITS AND FRAMES
----------------
`bbox` is in the coordinate frame of the PREPARED page image
(modules/01_prepare/03_tone/output/<source>), which is 1598x2177 for
every page in this corpus. It is the region's own extent, NOT the
padded crop - the crop on disk is a little larger on each side, see
02_segment/src/crop_lines.py.

Text is Unicode, already stripped of leading/trailing whitespace.
`confidence` is 0.0-1.0 over the whole line, not per character.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


# Bump when a field changes meaning or disappears, so a consumer can
# tell a new run from an incompatible one.
SCHEMA_VERSION = 1

# What a line's recognition can come to.
#
#   ok       - text was read, use it
#   empty    - the crop held no readable text (a stray mark, a smudge).
#              NOT an error: the region was real, it just says nothing.
#   diagram  - the region is flagged `tall` by segmentation, i.e. a
#              figure/brace/long-division rather than a row of writing.
#              A line recogniser should not be asked to read it, and
#              its text is empty by construction.
#   failed   - the engine errored on this crop. Distinct from `empty`
#              because it means something needs fixing, not that the
#              page is blank there.
STATUSES = ("ok", "empty", "diagram", "failed")


@dataclass
class LineResult:
    """One recognised line. The atom of this module's output."""

    # Stable identity. `line_uid` is unique across the whole corpus and
    # is the key anything downstream should join on.
    line_uid: str            # "s01_c1_p05_b00_l02"
    page_id: str             # "s01_c1_p05"
    student: int
    cie: int
    page: int
    block_id: int
    line_id: int

    # Where it came from, so a result can always be traced back to
    # pixels a human can look at.
    bbox: list               # [x1, y1, x2, y2] in prepared-page frame
    crop: str                # path relative to 02_segment/crops/

    # What was read.
    text: str
    confidence: float
    status: str              # one of STATUSES

    # Position within the page, 0-based, in reading order. Assigned by
    # this module because it is the last stage that sees the geometry.
    reading_order: int

    # True when segmentation flagged the region as not-a-text-line.
    tall: bool = False

    # Engine-specific extras (per-word boxes, alternatives, timings).
    # Consumers must treat this as optional and never require a key.
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class PageResult:
    """Every line of one page, in reading order."""

    page_id: str
    student: int
    cie: int
    page: int
    source: str              # prepared-page path, relative
    size: list               # [width, height]
    lines: list              # list[LineResult]

    @property
    def text(self):
        """The page as plain text, one line per recognised line."""

        return "\n".join(
            line.text for line in self.lines
            if line.status == "ok" and line.text
        )

    def to_dict(self):
        return {
            "page_id": self.page_id,
            "student": self.student,
            "cie": self.cie,
            "page": self.page,
            "source": self.source,
            "size": self.size,
            "lines": [line.to_dict() for line in self.lines],
        }


def validate_run(payload):
    """
    Check a loaded ocr.json before trusting it.

    Raises ValueError with a specific reason rather than returning a
    bool, because every caller would otherwise have to invent its own
    message for the same handful of problems.
    """

    for key in ("schema_version", "simulated", "engine", "pages"):
        if key not in payload:
            raise ValueError(f"missing top-level key: {key!r}")

    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {payload['schema_version']} != "
            f"{SCHEMA_VERSION} this consumer understands"
        )

    seen = set()

    for page in payload["pages"]:

        orders = []

        for line in page["lines"]:

            if line["status"] not in STATUSES:
                raise ValueError(
                    f"{line['line_uid']}: unknown status "
                    f"{line['status']!r}"
                )

            if not 0.0 <= line["confidence"] <= 1.0:
                raise ValueError(
                    f"{line['line_uid']}: confidence "
                    f"{line['confidence']} outside 0..1"
                )

            if line["line_uid"] in seen:
                raise ValueError(f"duplicate line_uid: {line['line_uid']}")

            seen.add(line["line_uid"])
            orders.append(line["reading_order"])

        if sorted(orders) != list(range(len(orders))):
            raise ValueError(
                f"{page['page_id']}: reading_order is not 0..n-1 "
                f"without gaps"
            )

    return len(seen)
