from docling_core.types import DoclingDocument
from docling_core.types.doc.document import SectionHeaderItem, TitleItem

from akd._base import OutputSchema, UnrestrictedAbstractBase


class _DoclingTitleExtractorOutputSchema(OutputSchema):
    """
    Output schema for the DoclingTitleExtractor tool.
    Represents the extracted title as a string.
    """

    title: str


class _DoclingTitleExtractor(UnrestrictedAbstractBase):
    """
    A utility class for extracting titles from DoclingDocument objects.
    (Hidden from public API)

    Uses a prioritized search strategy:
    1. Main section headers (level=1) in first 10 text items
    2. Any TitleItem in the document
    3. Any main section header (level=1) anywhere in document
    4. Document name attribute
    5. "Untitled" as final fallback

    Note:
    - This class is not intended for direct use outside of the web scraper tool.
    - It is designed to be used internally by the web scraper tool to extract titles
    - For convenience, we just bypass config-based validation here.
    """

    input_schema: DoclingDocument
    output_schema: _DoclingTitleExtractorOutputSchema

    def __init__(
        self,
        early_search_limit: int = 10,
        fallback_title: str = "Untitled",
        debug: bool = False,
    ) -> None:
        """
        Initialize the title extractor.

        Args:
            early_search_limit: Number of text items to search for early section headers
            fallback_title: Title to use when no other title is found
        """
        self.early_search_limit = early_search_limit
        self.fallback_title = fallback_title
        self.debug = bool(debug)

    async def _arun(
        self,
        doc: DoclingDocument,
        **kwargs,
    ) -> _DoclingTitleExtractorOutputSchema:
        """
        Extracts the title from a DoclingDocument using prioritized search strategies.

        Args:
            doc: The DoclingDocument to extract title from

        Returns:
            The extracted title string, or fallback_title if no suitable title found
        """
        title = (
            self._find_early_main_section_header(doc)
            or self._find_title_item(doc)
            or self._find_any_main_section_header(doc)
            or self._get_document_name(doc)
            or self.fallback_title
        )
        return _DoclingTitleExtractorOutputSchema(title=title)

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
        return (
            isinstance(text_item, TitleItem)
            and getattr(text_item, "text", None)
            and text_item.text.strip()
        )
