"""
Represents the complete answer booklet.
"""

from dataclasses import dataclass, field
from typing import List

from .page import Page


@dataclass
class Document:

    pages: List[Page] = field(default_factory=list)

    def add_page(self, page: Page):

        self.pages.append(page)

    def total_regions(self):

        return sum(len(page.regions) for page in self.pages)

    def get_page(self, page_number):

        for page in self.pages:

            if page.page_number == page_number:

                return page

        return None

"""
Represents the complete answer booklet.
"""

from dataclasses import dataclass, field
from typing import List

from .page import Page


@dataclass
class Document:

    pages: List[Page] = field(default_factory=list)

    def add_page(self, page: Page):

        self.pages.append(page)

    def total_regions(self):

        return sum(len(page.regions) for page in self.pages)

    def get_page(self, page_number):

        for page in self.pages:

            if page.page_number == page_number:

                return page

        return None