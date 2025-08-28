from typing import Optional, List
from pydantic import BaseModel, Field

from akd.structures import PaperDataItem


class SubSection(BaseModel):
    """Represents a sub-section within a scientific paper, containing a title and its associated textual content."""
    title: str = Field(..., title="Title of the sub-section")
    content: str = Field(..., title="Content of the subsection as a string")


class Section(BaseModel):
    """Represents a section of a scientific paper, optionally containing subsections and associated content."""
    title: str = Field(..., title="Title of the section")
    content: Optional[str] = Field(..., title="Content of the section as a string")
    subsections: Optional[list[SubSection]] = Field(..., title="List of optional subsections pertaining to the section")


class ParsedPaper(PaperDataItem):
    """A fully parsed scientific paper."""
    figures: Optional[list] = Field(default=None, title="List of figures present in the paper")
    tables: Optional[list] = Field(default=None, title="List of tables present in the paper")
    section_titles: Optional[list] = Field(default=None, title="List of section titles")
    sections: Optional[List[Section]] = Field(default=None, title="Sections present in the paper")
