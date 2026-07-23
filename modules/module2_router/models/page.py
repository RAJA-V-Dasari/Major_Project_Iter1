"""
Represents one page in the answer booklet.
"""

from dataclasses import dataclass, field
from typing import List

from .region import Region


@dataclass
class Page:

    page_number: int

    regions: List[Region] = field(default_factory=list)

    def add_region(self, region: Region):

        self.regions.append(region)

    def sort(self):

        self.regions.sort(
            key=lambda r: (
                r.bbox.y1,
                r.bbox.x1
            )
        )

    def __len__(self):

        return len(self.regions)