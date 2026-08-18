"""
Re-route regions once their text is known.

    04_ocr/output/ocr.json          (what the recogniser read)
    03_router/output/routed_regions.json  (the pre-OCR routing)
        -> 03_router/output/after_ocr/student_NN/cie_M/routed_regions.json

WHY THIS STAGE EXISTS AT ALL
----------------------------
route.py sends everything readable to `ocr` and nothing to `math_ocr`,
which looks like an omission and is not. Separating equations from
prose by looking at the crop was measured on 600 real regions and
rejected: ink density, inter-symbol gap over line height,
component-size variation and aspect ratio are all unimodal, so any
cut would be an invented boundary, and the corpus has no labelled
maths to validate one against. See rules.py.

After recognition the question is easy, because "=" is a character you
can read rather than a property you have to infer from ink statistics.
So maths is identified here, and this is the ONLY place a region is
addressed to `math_ocr`.

THE ORDER OF THE PIPELINE IS THEREFORE NOT LINEAR
-------------------------------------------------
    02_segment -> 03_router -> 04_ocr -> 03_router (again) -> 05_math

A line goes to the text recogniser first even when it turns out to be
an equation. That costs one wasted recognition per equation, which is
cheap, and buys not having to guess. The alternative - guessing from
pixels - was measured and does not work.

WHAT 05_MATH RECEIVES
---------------------
Exactly the router schema it already reads, one file per booklet, so
its own simulate_router.py is unnecessary and has been removed. The
crop path travels through unchanged, so nothing needs re-cropping: the
maths engine re-reads the same image the text engine saw.

Run:
    python reroute.py                     # from simulated OCR text
    python reroute.py --ocr <path>        # from real recognise.py output
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import config
import rules


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES_DIR = STAGE_DIR.parent

ROUTED_PATH = STAGE_DIR / "output" / "routed_regions.json"
OCR_PATH = MODULES_DIR / "04_ocr" / "output" / "ocr.json"

OUT_DIR = STAGE_DIR / "output" / "after_ocr"


def load_text_by_uid(ocr_path):
    """line_uid -> recognised text, for lines that produced any."""

    if not ocr_path.exists():
        raise SystemExit(
            f"{ocr_path} not found - run 04_ocr/src/simulate.py "
            f"(fast) or recognise.py (real) first"
        )

    with open(ocr_path) as handle:
        payload = json.load(handle)

    simulated = payload.get("simulated", False)

    text = {}

    for page in payload["pages"]:
        for line in page["lines"]:
            # only lines that were actually read can be judged; empty,
            # diagram and failed carry no text to look at
            if line.get("status") == "ok" and line.get("text"):
                text[line["line_uid"]] = line["text"]

    return text, simulated


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", type=Path, default=OCR_PATH)
    parser.add_argument("--routed", type=Path, default=ROUTED_PATH)
    args = parser.parse_args()

    if not args.routed.exists():
        raise SystemExit(f"{args.routed} not found - run route.py first")

    with open(args.routed) as handle:
        routed = json.load(handle)

    text_by_uid, simulated = load_text_by_uid(args.ocr)

    moved = Counter()
    booklets = defaultdict(list)

    for page in routed["pages"]:

        page_regions = []

        for region in page["regions"]:

            route = region["metadata"]["route"]
            reason = region["metadata"]["reason"]

            recognised = text_by_uid.get(region["id"])

            if recognised is not None:

                route, reason = rules.reroute_by_content(
                    {"route": route, "reason": reason}, recognised
                )

            if route != region["metadata"]["route"]:
                moved[f"{region['metadata']['route']} -> {route}"] += 1

            page_regions.append({
                **region,
                "processor": config.PROCESSOR_NAMES[route],
                "metadata": {**region["metadata"],
                             "route": route,
                             "reason": reason,
                             "text": recognised},
            })

        # 05_math reads one routed file per booklet
        student = int(page["page"][1:3])
        cie = int(page["page"][5])

        booklets[(student, cie)].append({
            "page": page["page"],
            "regions": page_regions,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    math_regions = 0

    for (student, cie), pages in sorted(booklets.items()):

        target = OUT_DIR / f"student_{student:02d}" / f"cie_{cie}"
        target.mkdir(parents=True, exist_ok=True)

        with open(target / "routed_regions.json", "w") as handle:
            json.dump({"pages": pages,
                       "from_simulated_ocr": simulated}, handle, indent=1)

        written += 1

        math_regions += sum(
            1 for page in pages for region in page["regions"]
            if region["processor"] == "math_ocr"
        )

    counts = Counter(
        region["processor"]
        for pages in booklets.values()
        for page in pages for region in page["regions"]
    )

    print(f"Text source     : {args.ocr}")

    if simulated:
        print("                  SIMULATED text - the maths found below is "
              "as synthetic as the text it came from")

    print(f"Booklets written: {written}")
    print(f"Processors      : " + ", ".join(
        f"{k}={v}" for k, v in counts.most_common()))
    print(f"Re-routed       : " + (", ".join(
        f"{k}: {v}" for k, v in moved.most_common()) or "nothing moved"))
    print()
    print(f"Output          : {OUT_DIR}")
    print(f"05_math reads   : {MODULES_DIR / '05_math' / 'input' / 'routed'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
