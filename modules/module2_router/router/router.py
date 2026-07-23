"""
Region Router

Main orchestration class for Module 2.
"""

import json
from pathlib import Path

from rich.console import Console

from models.document import Document
from models.page import Page
from models.region import Region, BoundingBox

from config import ROUTING_TABLE

console = Console()


class RegionRouter:

    def __init__(self):

        self.document = Document()

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------

    def load(self, json_path):

        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        pages = {}

        for item in data["regions"]:

            bbox = BoundingBox(
                x1=item["bbox"][0],
                y1=item["bbox"][1],
                x2=item["bbox"][2],
                y2=item["bbox"][3],
            )

            region = Region(
                id=item["id"],
                page=item["page"],
                label=item["label"],
                confidence=item["confidence"],
                bbox=bbox,
                crop_path=item["crop_path"]
            )

            if region.page not in pages:
                pages[region.page] = Page(region.page)

            pages[region.page].add_region(region)

        for page in sorted(pages.keys()):
            self.document.add_page(pages[page])

        console.print(
            f"[green]Loaded {self.document.total_regions()} regions[/green]"
        )

    # --------------------------------------------------
    # VALIDATE
    # --------------------------------------------------

    def validate(self):

        from router.validator import validate_document

        validate_document(self.document)

    # --------------------------------------------------
    # SORT
    # --------------------------------------------------

    def sort_regions(self):

        from router.ordering import sort_document

        sort_document(self.document)

    # --------------------------------------------------
    # ROUTE
    # --------------------------------------------------

    def route(self):

        from router.dispatcher import dispatch_document

        dispatch_document(
            self.document,
            ROUTING_TABLE
        )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    def save(self, output_path):

        output = {
            "pages": []
        }

        for page in self.document.pages:

            page_data = {
                "page": page.page_number,
                "regions": []
            }

            for region in page.regions:

                page_data["regions"].append(
                    region.to_dict()
                )

            output["pages"].append(page_data)

        with open(output_path, "w") as f:
            json.dump(
                output,
                f,
                indent=4
            )

        console.print(
            f"[cyan]Saved routed output → {output_path}[/cyan]"
        )