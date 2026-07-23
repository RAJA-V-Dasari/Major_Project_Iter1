"""
Assign downstream processors.
"""

from rich.console import Console

console = Console()


def dispatch_document(document, routing_table):

    for page in document.pages:

        for region in page.regions:

            region.processor = routing_table.get(
                region.label,
                "manual_review"
            )

    console.print(
        "[green]Processor routing complete[/green]"
    )