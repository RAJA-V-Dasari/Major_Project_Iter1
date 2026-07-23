"""
Module 2 - Region Router

Pipeline:

Segmentation JSON
        ↓
Validation
        ↓
Ordering
        ↓
Routing
        ↓
Output JSON
"""

from rich.console import Console

from config import (
    SEGMENTATION_JSON,
    OUTPUT_JSON
)

from router.router import RegionRouter


console = Console()


def main():

    console.rule("[bold green]Module 2 : Region Router")

    router = RegionRouter()

    router.load(SEGMENTATION_JSON)

    router.validate()

    router.sort_regions()

    router.route()

    router.save(OUTPUT_JSON)

    console.print(f"\n[green]✔ Routing complete.")
    console.print(f"[cyan]Saved to:[/cyan] {OUTPUT_JSON}")


if __name__ == "__main__":
    main()