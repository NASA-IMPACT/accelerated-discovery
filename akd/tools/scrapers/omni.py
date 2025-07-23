import re
from typing import Any, Literal, Tuple
from urllib.parse import unquote

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import (
    DocumentConverter,
    HTMLFormatOption,
    PdfFormatOption,
)
from docling_core.types import DoclingDocument
from loguru import logger
from pydantic import Field, FilePath, FileUrl, HttpUrl, field_validator, model_validator

from ._base import (
    PDFScraper,
    ScrapedMetadata,
    ScraperToolConfig,
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
    WebScraper,
)
from .utils import _DoclingMetadataExtractor


class OmniScraperInputSchema(ScraperToolInputSchema):
    """
    Input schema for OmniScraper tools, extending the base scraper input schema.
    """

    url: HttpUrl | FilePath | FileUrl = Field(
        ...,
        description="The URL of the document to scrape.",
    )


class OmniScrapedMetadata(ScrapedMetadata):
    url: HttpUrl | FilePath | FileUrl = Field(
        ...,
        description="The URL or file path of the scraped document.",
    )


class OmniScraper(WebScraper, PDFScraper):
    """
    Base class for OmniScraper tools, combining web scraping and PDF scraping capabilities.
    """

    input_schema = OmniScraperInputSchema
    output_schema = ScraperToolOutputSchema
    config_schema = ScraperToolConfig


class DoclingScraperConfig(ScraperToolConfig):
    pdf_mode: Literal["accurate", "fast"] = Field(default="fast")
    analyze_image: bool = Field(
        default=False,
        description="Use VML to analyze image? (might slow down the parse)",
    )
    do_table_structure: bool = Field(default=False)
    use_ocr: bool = Field(default=False, description="Use OCR for PDF parsing?")
    allowed_formats: list[InputFormat] = Field(
        default_factory=lambda: [
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.HTML,
            InputFormat.XLSX,
            InputFormat.ASCIIDOC,
            InputFormat.MD,
        ],
    )
    title_fallback: str = Field(
        default="Untitled",
    )
    title_early_search_limit: int = Field(
        default=10,
    )

    @field_validator("pdf_mode", "export_type", mode="before")
    @classmethod
    def convert_to_lowercase(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    def add_image_format_if_needed(self):
        if self.analyze_image and InputFormat.IMAGE not in self.allowed_formats:
            self.allowed_formats.append(InputFormat.IMAGE)
        return self


class DoclingScraper(OmniScraper):
    """
    Scrapes PDF, HTML, docs (and other supported formats) into Markdown using Docling.

    Note:

    - This tool requires the `docling` package to be installed.
    - It supports both PDF and HTML formats, with options for OCR and table structure parsing.
    - The `pdf_mode` can be set to "accurate" or "fast"
    - Make sure `huggingface-cli login` is run before using this tool to access the required models.
    """

    config_schema = DoclingScraperConfig

    def _post_init(
        self,
    ) -> None:
        super()._post_init()
        self._setup_converter()
        self._metadata_extractor = _DoclingMetadataExtractor(
            early_search_limit=self.config.title_early_search_limit,
            fallback_title=self.config.title_fallback,
            debug=self.config.debug,
        )

    def _setup_converter(
        self,
        custom_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the DocumentConverter, conditionally configuring for pdf_mode,
        Common Table Structure (CV) and image analysis options.
        """
        # Disable table-structure (CV) if images are not analyzed
        pipeline_options = PdfPipelineOptions(
            do_table_structure=self.do_table_structure,
            do_ocr=self.use_ocr,
        )
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE
            if self.pdf_mode.lower() == "accurate"
            else TableFormerMode.FAST
        )

        format_options: dict[InputFormat, Any] = {
            InputFormat.HTML: HTMLFormatOption(),
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
        self.format_options = format_options
        if custom_options:
            format_options.update(custom_options)

        if self.debug:
            logger.debug(
                f"Docling format options :: {format_options}",
            )

        self.doc_converter = DocumentConverter(
            allowed_formats=self.allowed_formats,
            format_options=format_options,
        )

    async def _get_docling_document(
        self,
        file_path: str,
        force_format: InputFormat | None = None,
        reconfigure: bool = False,
    ) -> DoclingDocument:
        """
        Convert a single URL into a DoclingDocument.
        Raises:
            RuntimeError      -- conversion failure
        """
        if reconfigure or not hasattr(self, "doc_converter"):
            self._setup_converter()

        try:
            result = self.doc_converter.convert(file_path)
            return result.document
        except Exception as e:
            raise RuntimeError(f"Conversion failed ({file_path}): {e}")

    async def _clean_markdown(self, md: str) -> str:
        """
        Tidy up markdown: collapse excessive blank lines & trim trailing spaces.
        """
        md = re.sub(r"\n\s*\n\s*\n", "\n\n", md)
        lines = [line.rstrip() for line in md.splitlines()]
        return "\n".join(lines).strip() + "\n"

    async def _process_document(
        self,
        path: str,
    ) -> Tuple[str, OmniScrapedMetadata]:
        """
        Full round-trip: fetch doc, export to markdown, clean, and build metadata (docling doesn't support metadata extraction).
        """
        doc = await self._get_docling_document(path)

        content = ""
        if self.export_type == "html":
            content = doc.export_to_html()
        else:
            content = doc.export_to_markdown()
            content = await self._clean_markdown(content)

        _docling_metadata = await self._metadata_extractor.arun(doc)

        metadata = OmniScrapedMetadata(
            url=path,
            query=path,
            title=_docling_metadata.title,
        )
        return content, metadata

    async def _arun(
        self,
        params: OmniScraperInputSchema,
        **kwargs,
    ) -> ScraperToolOutputSchema:
        """
        Entry point for external calls.
        Returns cleaned markdown + metadata, or raises a descriptive error.
        """
        path = str(params.url)
        if path.startswith("file://"):
            path = unquote(params.url.path)
        try:
            content, meta = await self._process_document(path)
            return ScraperToolOutputSchema(content=content, metadata=meta)

        except FileNotFoundError as e:
            # local file was missing
            raise RuntimeError(f"[File Not Found] {e}")

        except RuntimeError as e:
            # issues during conversion
            raise RuntimeError(f"[Conversion Error] {e}")

        except Exception as e:
            raise RuntimeError(f"[Internal Error] Failed to scrape {path}") from e
