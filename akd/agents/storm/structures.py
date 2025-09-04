from typing import List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from akd.agents.search.aspect_search.structures import InterviewState, Perspectives

# =============================================================
# Outline structures
# =============================================================


class OutlineSubsection(BaseModel):
    """Represents a subsection of the outline"""

    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class OutlineSection(BaseModel):
    """Represents a section of the outline"""

    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: Optional[List[OutlineSubsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    """Structure out the article's outline"""

    page_title: str = Field(..., title="Title of the Wikipedia page")
    sections: List[OutlineSection] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


# =============================================================
# Article structures
# =============================================================


class ArticleSubSection(BaseModel):
    """The subsection of the generated article"""

    subsection_title: str = Field(..., title="Title of the subsection")
    content: str = Field(
        ...,
        title="Full content of the subsection. Include [#] citations to the cited sources where relevant.",
    )

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.content}".strip()


class ArticleSection(BaseModel):
    """The section of the generated article"""

    section_title: str = Field(..., title="Title of the section")
    content: str = Field(..., title="Full content of the section")
    subsections: Optional[List[ArticleSubSection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )
    citations: List[str] = Field(default_factory=list)

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            subsection.as_str for subsection in self.subsections or []
        )
        citations = "\n".join([f" [{i}] {cit}" for i, cit in enumerate(self.citations)])
        return (
            f"## {self.section_title}\n\n{self.content}\n\n{subsections}".strip()
            + f"\n\n{citations}".strip()
        )


# =============================================================
# Research structures
# =============================================================


class ResearchState(TypedDict):
    """Tracks state of research on the topic"""

    topic: str
    outline: Outline
    perspectives: Perspectives
    interview_results: List[InterviewState]
    references: dict
    sections: List[ArticleSection]
    article: str
