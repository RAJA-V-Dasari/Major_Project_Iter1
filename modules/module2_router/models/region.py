"""
Region Data Model

Represents one segmented region produced by Module 1.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class BoundingBox:
    """
    Bounding box coordinates.

    (x1, y1) --------
       |             |
       |             |
       -------- (x2, y2)
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2
        )


@dataclass
class Region:
    """
    Represents one segmented object.
    """

    id: int

    page: int

    label: str

    confidence: float

    bbox: BoundingBox

    crop_path: str

    processor: Optional[str] = None

    reading_order: Optional[int] = None

    ignored: bool = False

    metadata: dict = field(default_factory=dict)

    def to_dict(self):

        return {
            "id": self.id,
            "page": self.page,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": {
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "x2": self.bbox.x2,
                "y2": self.bbox.y2,
            },
            "crop_path": self.crop_path,
            "processor": self.processor,
            "reading_order": self.reading_order,
            "ignored": self.ignored,
            "metadata": self.metadata,
        }