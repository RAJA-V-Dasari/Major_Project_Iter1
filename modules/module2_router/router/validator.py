"""
Validation of segmented regions.
"""

from rich.console import Console

from config import (
    DISCARD_THRESHOLD,
    VALID_LABELS
)

console = Console()


def validate_document(document):

    removed = 0

    for page in document.pages:

        valid_regions = []

        for region in page.regions:

            if region.confidence < DISCARD_THRESHOLD:
                removed += 1
                continue

            if region.label not in VALID_LABELS:
                region.label = "unknown"

            valid_regions.append(region)

        page.regions = valid_regions

    console.print(
        f"[yellow]Validation complete "
        f"(removed {removed} regions)[/yellow]"
    )