"""
Route every segmented region to the processor that should read it.

    03_router/input/  (02_segment/crops: manifest.csv + line images)
        -> 03_router/output/  routed.json, routed.csv

Pipeline shape, kept from the module this replaces because it is the
right one: load -> validate -> order -> route -> save.

WHAT CHANGED, AND WHY IT HAD TO
-------------------------------
The previous router (modules/module2_router) keyed everything on a
`label` per region - paragraph / equation / diagram / table - and
dropped regions below a confidence threshold. Neither exists here.
02_segment emits geometry only and deliberately does not classify,
and its output is deterministic, so there is no probability to
threshold. Feeding that router would have meant inventing a label and
a confidence for all 33,577 regions, which is exactly the kind of
fabricated number this pipeline has avoided.

So routing is done on what is actually known - see rules.py, including
the measurement showing why maths cannot be told from prose by
geometry - and maths is identified after recognition instead.

READING ORDER IS RE-DERIVED, NOT TRUSTED
----------------------------------------
Regions arrive in the order segmentation emitted them, which is
already top-to-bottom. This re-sorts by (y1, x1) anyway and reports
any page where the two disagree: it is a cheap independent check on
the upstream stage, and a silent ordering bug downstream is expensive
to find.

Run:
    python route.py                # whole corpus
    python route.py --limit 50     # a sample
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import config
import rules


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent

MANIFEST_PATH = STAGE_DIR / "input" / "manifest.csv"
SEGMENTATION_PATH = (STAGE_DIR.parent / "02_segment" / "output"
                     / "segmentation.json")

OUT_DIR = STAGE_DIR / "output"

# Corpus median, used only if a page's own pitch is unavailable.
FALLBACK_PITCH = 58.4


def load_regions():
    """Croppable regions from the segmentation manifest."""

    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"{MANIFEST_PATH} not found - run "
            f"02_segment/src/crop_lines.py first"
        )

    with open(MANIFEST_PATH) as handle:
        rows = list(csv.DictReader(handle))

    regions = []

    for row in rows:

        # regions with no crop were too small to hold a glyph; there is
        # nothing for any processor to read, so they are not routed.
        # They stay counted, so the totals reconcile upstream.
        if not row["crop"]:
            continue

        regions.append({
            "line_uid": (f"{row['page_id']}_b{int(row['block_id']):02d}"
                         f"_l{int(row['line_id']):02d}"),
            "page_id": row["page_id"],
            "student": int(row["student"]),
            "cie": int(row["cie"]),
            "page": int(row["page"]),
            "block_id": int(row["block_id"]),
            "line_id": int(row["line_id"]),
            "x1": int(row["x1"]), "y1": int(row["y1"]),
            "x2": int(row["x2"]), "y2": int(row["y2"]),
            "tall": bool(int(row["tall"])),
            "crop": row["crop"],
        })

    return regions, len(rows) - len(regions)


def page_pitches():
    """Per-page rule pitch, so thresholds stay in physical units."""

    if not SEGMENTATION_PATH.exists():
        return {}

    with open(SEGMENTATION_PATH) as handle:
        return {p["page_id"]: p["rule_pitch"] for p in json.load(handle)}


def order_page(regions):
    """
    Reading order for one page, top-to-bottom then left-to-right.

    Returns (ordered, agrees) where `agrees` says whether this matches
    the order segmentation already emitted.
    """

    upstream = sorted(regions, key=lambda r: (r["block_id"], r["line_id"]))

    ordered = sorted(regions, key=lambda r: (r["y1"], r["x1"]))

    agrees = [r["line_uid"] for r in upstream] == [
        r["line_uid"] for r in ordered
    ]

    return ordered, agrees


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="first N pages only")
    args = parser.parse_args()

    regions, uncroppable = load_regions()

    pitches = page_pitches()

    by_page = defaultdict(list)

    for region in regions:
        by_page[region["page_id"]].append(region)

    page_ids = sorted(by_page)

    if args.limit:
        page_ids = page_ids[:args.limit]

    pages = []
    routes = Counter()
    tags = Counter()
    disagreements = []

    for page_id in page_ids:

        pitch = pitches.get(page_id, FALLBACK_PITCH)

        ordered, agrees = order_page(by_page[page_id])

        if not agrees:
            disagreements.append(page_id)

        routed = []

        for order, region in enumerate(ordered):

            route, reason, region_tags = rules.route_geometry(region, pitch)

            routes[route] += 1

            for tag in region_tags:
                tags[tag] += 1

            routed.append({
                **region,
                "reading_order": order,
                "route": route,
                "reason": reason,
                "tags": region_tags,
            })

        pages.append({
            "page_id": page_id,
            "student": routed[0]["student"],
            "cie": routed[0]["cie"],
            "page": routed[0]["page"],
            "rule_pitch": pitch,
            "regions": routed,
        })

    payload = {
        "routes": list(config.ROUTES),
        "default_route": config.DEFAULT_ROUTE,
        "math_rules_are_provisional": config.MATH_RULES_ARE_PROVISIONAL,
        "note": ("Maths is NOT identified here. Geometry cannot separate "
                 "it from prose on this corpus (see rules.py); it is "
                 "identified after recognition via "
                 "rules.reroute_by_content()."),
        "pages_total": len(pages),
        "regions_total": sum(len(p["regions"]) for p in pages),
        "regions_without_crop": uncroppable,
        "pages": pages,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "routed.json", "w") as handle:
        json.dump(payload, handle, indent=1)

    # 04_ocr reads this shape: pages -> regions, each carrying the
    # `processor` string it filters on. Emitted here so the recogniser
    # consumes a REAL routing decision rather than a simulated one.
    #
    # `page` is the page_id string, not the page number. The number
    # collides - student_01 page 2 and student_02 page 2 are both 2 -
    # and the consumer groups regions by this key, so an int would
    # merge 61 students' page 2 into one bucket and then label the
    # whole bucket with whichever student happened to sort first.
    # A unique key removes that failure by construction.
    compat = {"pages": []}

    for page in pages:
        compat["pages"].append({
            "page": page["page_id"],
            "regions": [
                {
                    "id": region["line_uid"],
                    "page": page["page_id"],
                    "bbox": {"x1": region["x1"], "y1": region["y1"],
                             "x2": region["x2"], "y2": region["y2"]},
                    "crop_path": region["crop"],
                    "processor": config.PROCESSOR_NAMES[region["route"]],
                    "reading_order": region["reading_order"],
                    "ignored": False,
                    "metadata": {"route": region["route"],
                                 "reason": region["reason"],
                                 "tags": region["tags"]},
                }
                for region in page["regions"]
            ],
        })

    with open(OUT_DIR / "routed_regions.json", "w") as handle:
        json.dump(compat, handle, indent=1)

    with open(OUT_DIR / "routed.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line_uid", "page_id", "student", "cie", "page",
                         "reading_order", "x1", "y1", "x2", "y2",
                         "tall", "route", "tags", "crop"])
        for page in pages:
            for region in page["regions"]:
                writer.writerow([
                    region["line_uid"], region["page_id"], region["student"],
                    region["cie"], region["page"], region["reading_order"],
                    region["x1"], region["y1"], region["x2"], region["y2"],
                    int(region["tall"]), region["route"],
                    "|".join(region["tags"]), region["crop"],
                ])

    print(f"Pages           : {len(pages)}")
    print(f"Regions routed  : {payload['regions_total']}")
    print(f"Not routed      : {uncroppable} (no crop - too small to read)")
    print()
    for route in config.ROUTES:
        if routes[route]:
            print(f"  {route:10s} : {routes[route]}")
    print()
    print(f"Tags            : " + ", ".join(
        f"{k}={v}" for k, v in tags.most_common()))
    print(f"Reading order   : re-derived; disagrees with segmentation on "
          f"{len(disagreements)} page(s)")

    if disagreements:
        print(f"  first few: {disagreements[:5]}")

    print()
    print(f"Output          : {OUT_DIR}")
    print()
    print("No region routed to math_ocr: maths is identified after "
          "recognition, not here.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
