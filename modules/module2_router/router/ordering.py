"""
Reading-order sorting.
"""


def sort_document(document):

    for page in document.pages:

        page.regions.sort(
            key=lambda r: (
                r.bbox.y1,
                r.bbox.x1
            )
        )

        for idx, region in enumerate(page.regions, start=1):
            region.reading_order = idx