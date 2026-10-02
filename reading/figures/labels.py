"""
Hand labels: which vertical stretches of which pages hold a drawing.

Ranges are in ORIGINAL page pixels (1598 x 2177), read off the ruled
contact sheets in this folder. A band is labelled "drawn" if its centre
falls inside one of the ranges below.

WHAT COUNTS AS DRAWN
--------------------
Anything with two-dimensional structure that would be worth cutting out
of the page and keeping as an image: node-and-edge diagrams, boxes with
arrows, packet-format strips, and hand-ruled tables. The pipeline treats
all of these the same way - it crops them - so the classifier should
too.

WHAT DOES NOT COUNT
-------------------
Running prose and bulleted lists. Ordinary mathematical working written
as a column of lines also does not count - see `s03_c1_p04` in the
prose set below.

`s02_c3_p08` moved the other way. Its CRC long division is laid out as
an indented staircase, which is visually closer to a diagram than to a
line of prose, and there is no plan to run text evaluation on this kind
of page as it stands - a cropped picture of the working is an
acceptable representation of it for now. So it is labelled drawn, not
as a claim that it structurally IS a diagram, but as a product decision
about how this kind of content should be handled.
"""

# page -> list of (y1, y2) stretches that are drawn
DRAWN = {
    "s01_c3_p03": [(550, 900), (1040, 1500)],
    "s02_c1_p04": [(790, 1160)],
    "s10_c1_p04": [(740, 1260)],
    "s13_c1_p02": [(1350, 1900)],
    "s04_c3_p03": [(540, 960), (1050, 1510), (1560, 2050)],
    "s12_c3_p02": [(1330, 1900)],
    "s14_c1_p07": [(1270, 1570)],
    "s09_c2_p07": [(240, 700), (880, 1420)],
    # the sparse tree the current detector misses - the case that
    # started this experiment
    "s19_c1_p08": [(250, 730)],
    # CRC long division staircase - treated as a figure by decision,
    # not because it fails a structural test. See the module note.
    "s02_c3_p08": [(150, 800)],
}

# pages inspected and confirmed to hold no drawing at all
PROSE_ONLY = [
    "s02_c1_p06",
    "s02_c2_p09",
    "s03_c1_p02",
    "s06_c1_p05",     # neat cursive
    "s03_c1_p04",     # mathematics written as lines
    "s25_c3_p07",     # messy hand, worked calculation
]

ALL_PAGES = sorted(set(DRAWN) | set(PROSE_ONLY))


def label_for(pid, y1, y2):
    """1 if the band's centre sits inside a drawn stretch, else 0."""
    mid = (y1 + y2) / 2
    return int(any(a <= mid <= b for a, b in DRAWN.get(pid, [])))
