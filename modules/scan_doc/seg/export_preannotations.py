"""
Convert detector output into COCO JSON that CVAT can import as
pre-annotations, so labeling becomes "fix the class" instead of
"draw every box from scratch".

Input is any regions JSON in this module's standard shape:

    {"pages": [{"page": 1, "image": "page_1.png", "regions":
        [{"id": 1, "label": "figure", "confidence": 0.9,
          "bbox": [x1, y1, x2, y2]}, ...]}]}

Both segment.py (DocLayout-YOLO) and detect_lines.py (docTR) emit this.

Usage:
    python export_preannotations.py                          # default trio
    python export_preannotations.py output/regions.json      # any single source
    python export_preannotations.py a.json b.json            # merged together
    python export_preannotations.py 'a.json:table,figure'    # only those labels

Passing several sources merges them per page. A source may be suffixed
with `:label1,label2` to take only those labels from it, which is how
each detector contributes just the thing it is good at:

    detect_lines.py   -> text geometry (excellent on handwriting)
    detect_nontext.py -> diagrams, arrows, seals, drawn boxes
    segment.py        -> printed tables only; DocLayout-YOLO detects
                         those reliably even though its labels are
                         useless for handwriting

Then in CVAT: create a task with the same page images, and use
Actions -> Upload annotations -> COCO 1.0 with the produced file.

Import maps every source label through SOURCE_LABEL_MAP so the boxes
arrive already carrying the closest target class; anything unmapped
lands in `unknown` for you to reassign.
"""

import json
import sys
from pathlib import Path

from PIL import Image


INPUT_DIR = Path("input/pages")
OUTPUT_PATH = Path("output/cvat_preannotations.json")

DEFAULT_SOURCES = [
    "output/lines.json",
    "output/nontext.json",
    # DocLayout-YOLO mislabels handwriting, but its printed-table
    # detection is solid, so take only that from it
    "output/regions.json:table",
]


# ==========================
# Target label set
# ==========================
#
# These are the classes the fine-tuned model will be trained on.
# Keep this list in sync with dataset/data.yaml and config in segment.py.

TARGET_LABELS = [
    "text",            # handwritten prose lines            -> ocr
    "question",        # "4b)", "a)", "ii)" part markers    -> ocr
    "equation",        # mathematical expressions           -> math_ocr
    "diagram",         # drawings, arrows, worked layouts   -> diagram_parser
    "table",           # ruled tables                       -> table_parser
    "code",            # code snippets                      -> (needs routing)
    "crossed_out",     # struck-through content             -> ignore
    "page_furniture",  # page numbers, watermarks, seals    -> (needs routing)
    "unknown",         # needs a human decision             -> manual_review
]

# NOTE: these names deliberately match the keys in Module 2's
# config.ROUTING_TABLE so routed output needs no translation layer.
# Two of them are NOT yet in that table and will fail Module 2's
# VALID_LABELS check until Raja adds them:
#
#     "code":           -> probably its own code_parser, or ocr
#     "page_furniture": -> "ignore"
#
# See README for the full integration note.


# Best-effort mapping from source detector labels to target labels.
# These are only starting guesses - the whole point of the CVAT pass is
# that a human corrects them.

SOURCE_LABEL_MAP = {
    # DocLayout-YOLO labels
    "plain text": "text",
    "title": "text",
    "figure": "unknown",       # on handwriting this is usually mislabeled text
    "figure_caption": "text",
    "table": "table",
    "table_caption": "text",
    "table_footnote": "text",
    "isolate_formula": "equation",
    "formula_caption": "text",
    "abandon": "page_furniture",

    # docTR / line-detector labels
    "line": "text",
    "word": "text",
    "block": "text",

    # geometric proposals from detect_nontext.py - these are exactly the
    # regions no detector could name, so they always need a human
    "nontext": "unknown",
}


def load_source(path):

    with open(path) as f:
        return json.load(f)


def parse_source(spec):
    """
    Split a source spec into its path and optional label filter.

        "output/lines.json"           -> (Path(...), None)
        "output/regions.json:table"   -> (Path(...), {"table"})
    """

    spec = str(spec)

    # rsplit so Windows-style drive letters or colons in a directory
    # name don't get mistaken for the filter separator
    if ":" in spec:

        path, _, labels = spec.rpartition(":")

        if path and not labels.endswith(".json"):
            return Path(path), {l.strip() for l in labels.split(",") if l.strip()}

    return Path(spec), None


def merge_sources(specs):
    """
    Combine several detector outputs into one document, keyed by page
    image so a page's text and non-text boxes end up together.

    Region ids are renumbered per page, since each source numbers its
    own regions from 1 and CVAT needs them distinct.
    """

    pages = {}
    order = []

    for spec in specs:

        path, keep = parse_source(spec)

        document = load_source(path)

        for page in document["pages"]:

            image = page["image"]

            if image not in pages:
                pages[image] = {
                    "page": page["page"],
                    "image": image,
                    "regions": [],
                }
                order.append(image)

            regions = page["regions"]

            if keep is not None:
                regions = [r for r in regions if r["label"] in keep]

            pages[image]["regions"].extend(regions)

    for image in order:

        for index, region in enumerate(pages[image]["regions"], start=1):
            region["id"] = index

    return {"pages": [pages[image] for image in order]}


def build_coco(document):

    categories = [
        {"id": i, "name": name, "supercategory": ""}
        for i, name in enumerate(TARGET_LABELS, start=1)
    ]

    category_id = {name: i for i, name in enumerate(TARGET_LABELS, start=1)}

    images = []
    annotations = []

    annotation_id = 1

    for image_id, page in enumerate(document["pages"], start=1):

        image_name = page["image"]
        image_path = INPUT_DIR / image_name

        with Image.open(image_path) as im:
            width, height = im.size

        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        for region in page["regions"]:

            x1, y1, x2, y2 = region["bbox"]

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            if w == 0 or h == 0:
                continue

            target = SOURCE_LABEL_MAP.get(region["label"], "unknown")

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id[target],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "segmentation": [],
                    "iscrowd": 0,
                    "attributes": {
                        "occluded": False,
                        # keep provenance so you can tell what the detector
                        # originally thought, while relabeling
                        "source_label": region["label"],
                        "source_confidence": region.get("confidence", 0.0),
                    },
                }
            )

            annotation_id += 1

    return {
        "licenses": [{"id": 0, "name": "", "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "Pre-annotations for answer-sheet segmentation",
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


def main():

    sources = sys.argv[1:] or DEFAULT_SOURCES

    missing = [s for s in sources if not parse_source(s)[0].exists()]

    if missing:
        raise FileNotFoundError(
            f"{', '.join(str(s) for s in missing)} not found. "
            "Run detect_lines.py and detect_nontext.py first."
        )

    document = merge_sources(sources)

    coco = build_coco(document)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Sources    : {', '.join(str(p) for p in sources)}")
    print(f"Images     : {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Saved      : {OUTPUT_PATH}")
    print()
    print("Import into CVAT via: Actions -> Upload annotations -> COCO 1.0")


if __name__ == "__main__":
    main()
