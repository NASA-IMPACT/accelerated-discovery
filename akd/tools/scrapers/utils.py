from docling_core.types import DoclingDocument
from docling_core.types.doc.document import SectionHeaderItem, TitleItem
from pydantic import BaseModel, Field


class _DoclingMetadataExtractorOutputSchema(BaseModel):
    """Output schema for the DoclingMetadataExtractor utility."""

    title: str = Field(
        default="Untitled",
        description="The extracted title of the document.",
    )
    published_date: str | None = None


class _DoclingMetadataExtractor:
    """A utility class for extracting metadata from DoclingDocument objects.

    Hidden from public API. Used internally by the web scraper tool.

    For title:
        Uses a prioritized search strategy:
        1. Main section headers (level=1) in first 10 text items
        2. Any TitleItem in the document
        3. Any main section header (level=1) anywhere in document
        4. Document name attribute
        5. "Untitled" as final fallback
    """

    def __init__(
        self,
        early_search_limit: int = 10,
        fallback_title: str = "Untitled",
        debug: bool = False,
    ) -> None:
        self.early_search_limit = early_search_limit
        self.fallback_title = fallback_title
        self.debug = bool(debug)

    async def arun(self, doc: DoclingDocument) -> _DoclingMetadataExtractorOutputSchema:
        """Extract title metadata from a DoclingDocument."""
        title = self.extract_title(doc)
        return _DoclingMetadataExtractorOutputSchema(title=title)

    def extract_title(self, doc: DoclingDocument) -> str:
        return (
            self._find_early_main_section_header(doc)
            or self._find_title_item(doc)
            or self._find_any_main_section_header(doc)
            or self._get_document_name(doc)
            or self.fallback_title
        )

    def _find_early_main_section_header(self, doc: DoclingDocument) -> str | None:
        """
        Looks for main section headers (level=1) in the first N text items.
        This captures titles that appear early in the document structure.
        """
        search_items = doc.texts[: self.early_search_limit]
        for text_item in search_items:
            if self._is_main_section_header(text_item):
                return text_item.text.strip()
        return None

    def _find_title_item(self, doc: DoclingDocument) -> str | None:
        """
        Searches for any TitleItem in the entire document.
        TitleItems are explicitly marked as titles in the document structure.
        """
        for text_item in doc.texts:
            if self._is_title_item(text_item):
                return text_item.text.strip()
        return None

    def _find_any_main_section_header(self, doc: DoclingDocument) -> str | None:
        """
        Searches for any level=1 SectionHeaderItem anywhere in the document.
        This is a broader search than the early header search.
        """
        for text_item in doc.texts:
            if self._is_main_section_header(text_item):
                return text_item.text.strip()
        return None

    def _get_document_name(self, doc: DoclingDocument) -> str | None:
        """
        Extracts title from the document's name attribute.
        Returns None if name is empty or doesn't exist.
        """
        name = getattr(doc, "name", None)
        if name and name.strip():
            return name.strip()
        return None

    def _is_main_section_header(self, text_item) -> bool:
        """
        Checks if a text item is a main section header (level=1) with valid text.
        """
        return (
            isinstance(text_item, SectionHeaderItem)
            and getattr(text_item, "level", None) == 1
            and getattr(text_item, "text", None)
            and text_item.text.strip()
        )

    def _is_title_item(self, text_item) -> bool:
        """
        Checks if a text item is a TitleItem with valid text.
        """
        return isinstance(text_item, TitleItem) and getattr(text_item, "text", None) and text_item.text.strip()
