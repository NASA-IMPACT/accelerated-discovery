import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import (
    DocumentConverter,
    HTMLFormatOption,
    PdfFormatOption,
)
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import SectionHeaderItem, TitleItem
from loguru import logger
from markdownify import markdownify
from pydantic import Field, HttpUrl, field_validator, model_validator
from readability import Document
from requests import HTTPError, RequestException

from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.tools import BaseTool, BaseToolConfig


class WebpageScraperToolInputSchema(InputSchema):
    """
    Input schema for the WebpageScraperTool.
    """

    url: HttpUrl = Field(
        ...,
        description="URL of the webpage to scrape.",
    )
    include_links: bool = Field(
        default=True,
        description="Whether to preserve hyperlinks in the markdown output.",
    )


class WebpageMetadata(SearchResultItem):
    keywords: Optional[List[str]] = Field(
        None,
        description="List of keywords from the webpage content.",
    )


class WebpageScraperToolOutputSchema(OutputSchema):
    """Schema for the output of the WebpageScraperTool."""

    content: str = Field(
        ...,
        description="The scraped content in markdown format.",
    )
    metadata: WebpageMetadata = Field(
        ...,
        description="Metadata about the scraped webpage.",
    )


class WebpageScraperToolConfig(BaseToolConfig):
    """Configuration for the WebpageScraperTool."""

    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        description="User agent string to use for requests.",
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for HTTP requests.",
    )
    max_content_length: int = Field(
        default=100_000_000,
        description="Maximum content length in bytes to process.",
    )

    debug: bool = Field(
        default=True,
        description="Boolean flag for debug mode",
    )


class WebScraperToolBase(BaseTool):
    input_schema = WebpageScraperToolInputSchema
    output_schema = WebpageScraperToolOutputSchema
    config_schema = WebpageScraperToolConfig

    async def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extracts the author information from the webpage using multiple methods.

        Args:
            soup (BeautifulSoup): The parsed HTML content.

        Returns:
            Optional[str]: The extracted author name(s) or None if not found.
        """
        # Check standard meta tags
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta and author_meta.get("content"):
            return author_meta.get("content")

        # Check schema.org markup
        article_schema = soup.find("script", type="application/ld+json")
        if article_schema:
            try:
                data = json.loads(article_schema.string)
                if isinstance(data, dict):
                    author = data.get("author", {}).get("name")
                    if author:
                        return author
            except json.JSONDecodeError:
                pass

        # Check common author classes/IDs
        author_elements = soup.find_all(class_=re.compile(r"author|byline", re.I))
        if author_elements:
            authors = []
            for element in author_elements:
                if element.string and len(element.string.strip()) > 0:
                    authors.append(element.string.strip())
            if authors:
                return ", ".join(authors)

        return None

    async def _extract_metadata(
        self,
        soup: BeautifulSoup,
        doc: Document,
        url: str,
    ) -> WebpageMetadata:
        """
        Extracts comprehensive metadata from the webpage using multiple methods and sources.

        Args:
            soup (BeautifulSoup): The parsed HTML content.
            doc (Document): The readability document.
            url (str): The URL of the webpage.

        Returns:
            WebpageMetadata: The extracted metadata.
        """
        # domain = urlparse(url).netloc

        metadata = {
            "query": url,
            "url": url,
            "title": doc.title() or self._extract_title(soup),
            "doi": None,
            "keywords": [],
            "author": None,
            "publication_date": None,
        }

        # Extract Dublin Core metadata
        dc_mappings = {
            "DC.subject": "category",
            "DC.date.modified": "published_date",
            "DC.identifier": "doi",  # Some sites use this for DOI
        }

        for dc_name, meta_key in dc_mappings.items():
            meta_tag = soup.find("meta", attrs={"name": dc_name})
            if meta_tag and meta_tag.get("content"):
                metadata[meta_key] = meta_tag.get("content")

        # Extract Open Graph metadata
        og_mappings = {
            "og:description": "description",
            "og:site_name": "site_name",
            "og:type": "type",
            "og:image": "image_url",
            "og:locale": "language",
            "og:section": "section",
            "article:section": "section",
            "article:tag": "tags",
            "article:published_time": "published_date",
            "article:modified_time": "modified_date",
        }

        for og_prop, meta_key in og_mappings.items():
            try:
                meta_tag = soup.find("meta", property=og_prop)
                if meta_tag and meta_tag.get("content"):
                    metadata[meta_key] = meta_tag.get("content")
            except KeyError:
                continue

        # Extract schema.org metadata (JSON-LD)
        schema_tags = soup.find_all("script", type="application/ld+json")
        for tag in schema_tags:
            try:
                schema_data = json.loads(tag.string)
                if isinstance(schema_data, dict):
                    if "datePublished" in schema_data:
                        metadata["published_date"] = (
                            metadata["published_date"] or schema_data["datePublished"]
                        )
                    if "keywords" in schema_data:
                        metadata["keywords"] = (
                            metadata["keywords"] or schema_data["keywords"]
                        )
            except (json.JSONDecodeError, AttributeError):
                pass

        # Extract Twitter Card metadata
        twitter_mappings = {
            "twitter:description": "description",
            "twitter:site": "site_name",
        }

        for twitter_name, meta_key in twitter_mappings.items():
            meta_tag = soup.find("meta", attrs={"name": twitter_name})
            if meta_tag and meta_tag.get("content"):
                metadata[meta_key] = metadata[meta_key] or meta_tag.get("content")

        # Extract citation metadata
        citation_mappings = {
            "citation_title": "title",
            "citation_doi": "doi",
            "citation_journal_title": "publisher",
            "citation_keywords": "keywords",
            "citation_publication_date": "published_date",
        }

        for citation_name, meta_key in citation_mappings.items():
            meta_tag = soup.find("meta", attrs={"name": citation_name})
            if meta_key not in metadata:
                continue
            if meta_tag and meta_tag.get("content"):
                metadata[meta_key] = metadata[meta_key] or meta_tag.get("content")

        # Build citation string for academic content
        if metadata["doi"] or (
            metadata["title"] and metadata["author"] and metadata["publication_date"]
        ):
            citation_parts = []
            if metadata["author"]:
                citation_parts.append(metadata["author"])
            if metadata["title"]:
                citation_parts.append(f'"{metadata["title"]}"')
            if "publisher" in metadata:
                citation_parts.append(metadata["publisher"])
            if metadata["publication_date"]:
                citation_parts.append(metadata["publication_date"][:4])  # Just the year
            if metadata["doi"]:
                citation_parts.append(f"DOI: {metadata['doi']}")

            if citation_parts:
                metadata["citation"] = ". ".join(citation_parts)

        # Clean up metadata values
        for key, value in metadata.items():
            if isinstance(value, str):
                metadata[key] = value.strip()

        if isinstance(metadata["keywords"], str):
            metadata["keywords"] = [metadata["keywords"]]
        return WebpageMetadata(**metadata)

    async def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extracts the title using multiple fallback methods.

        Args:
            soup (BeautifulSoup): The parsed HTML content.

        Returns:
            str: The extracted title or a default string if not found.
        """
        # Try OG title first
        og_title = soup.find("meta", property="og:title")
        if og_title:
            return og_title.get("content")

        # Try regular title tag
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Try main heading
        h1_tag = soup.find("h1")
        if h1_tag and h1_tag.string:
            return h1_tag.string.strip()

        return None


class SimpleWebScraper(WebScraperToolBase):
    """
    Tool for scraping webpage content and converting it to markdown format.
    """

    async def _fetch_webpage(self, url: str) -> str:
        """
        Fetches the webpage content with custom headers and proper error handling.

        Args:
            url (str): The URL to fetch.

        Returns:
            str: The HTML content of the webpage.

        Raises:
            HTTPError: If the HTTP request fails
            ValueError: If content length exceeds maximum
            RequestException: For other request-related errors
        """
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

        try:
            response = requests.get(url, headers=headers, timeout=self.config.timeout)
            response.raise_for_status()

            if len(response.content) > self.config.max_content_length:
                raise ValueError(
                    f"Content length exceeds maximum of {self.config.max_content_length} bytes",
                )

            return response.text

        except requests.exceptions.HTTPError as http_err:
            raise HTTPError(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as req_err:
            raise RequestException(f"Error fetching webpage: {req_err}")

    async def _clean_markdown(self, markdown: str) -> str:
        """
        Cleans up the markdown content by removing excessive whitespace and normalizing formatting.

        Args:
            markdown (str): Raw markdown content.

        Returns:
            str: Cleaned markdown content.
        """
        # Remove multiple blank lines
        markdown = re.sub(r"\n\s*\n\s*\n", "\n\n", markdown)
        # Remove trailing whitespace
        markdown = "\n".join(line.rstrip() for line in markdown.splitlines())
        # Ensure content ends with single newline
        markdown = markdown.strip() + "\n"
        return markdown

    async def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extracts the main content from the webpage using custom heuristics.

        Args:
            soup (BeautifulSoup): Parsed HTML content.

        Returns:
            str: Main content HTML.
        """
        # Remove unwanted elements
        for element in soup.find_all(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        # Try to find main content container
        content_candidates = [
            soup.find("main"),
            soup.find(id=re.compile(r"content|main", re.I)),
            soup.find(class_=re.compile(r"content|main", re.I)),
            soup.find("article"),
        ]

        main_content = next(
            (candidate for candidate in content_candidates if candidate),
            None,
        )

        if not main_content:
            main_content = soup.find("body")

        return str(main_content) if main_content else str(soup)

    async def _arun(
        self,
        params: WebpageScraperToolInputSchema,
    ) -> WebpageScraperToolOutputSchema:
        """
        Runs the WebpageScraperTool with the given parameters.

        Args:
            params (WebpageScraperToolInputSchema): The input parameters for the tool.

        Returns:
            WebpageScraperToolOutputSchema: The output containing the markdown content and metadata.
        """

        if params.url.path.endswith((".pdf", ".PDF")):
            raise RuntimeError(f"Can't parse url with PDF :: {params.url}")
        html_content = await self._fetch_webpage(str(params.url))

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract main content using custom extraction
        main_content = await self._extract_main_content(soup)

        # Convert to markdown
        markdown_options = {
            "strip": ["script", "style"],
            "heading_style": "ATX",
            "bullets": "-",
            "wrap": True,
        }

        if not params.include_links:
            markdown_options["strip"].append("a")

        markdown_content = markdownify(main_content, **markdown_options)

        # Clean up the markdown
        markdown_content = await self._clean_markdown(markdown_content)

        # Extract metadata
        metadata = await self._extract_metadata(
            soup,
            Document(html_content),
            str(params.url),
        )

        return WebpageScraperToolOutputSchema(
            content=markdown_content.strip(),
            metadata=metadata,
        )


class Crawl4AIWebScraper(WebScraperToolBase):
    async def fetch(self, url: str):
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(url=url)

    async def _arun(
        self,
        params: WebpageScraperToolInputSchema,
        **kwargs,
    ) -> WebpageScraperToolOutputSchema:
        """
        Async version of the run method.
        """
        if params.url.path.endswith((".pdf", ".PDF")):
            raise RuntimeError(f"Can't parse url with PDF :: {params.url}")

        crawl_result = await self.fetch(str(params.url))

        html_content = crawl_result.html

        soup = BeautifulSoup(html_content, "html.parser")
        metadata = await self._extract_metadata(
            soup,
            Document(html_content),
            str(params.url),
        )

        return WebpageScraperToolOutputSchema(
            content=crawl_result.markdown.strip(),
            metadata=metadata,
        )


class DoclingScraperConfig(BaseToolConfig):
    pdf_mode: Literal["accurate", "fast"] = Field(default="fast")
    export_type: Literal["markdown", "html"] = Field(default="markdown")
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


class DoclingScraper(WebScraperToolBase):
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

    def _setup_converter(
        self,
        custom_options: Optional[Dict[str, Any]] = None,
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

        format_options: Dict[InputFormat, Any] = {
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
        force_format: Optional[InputFormat] = None,
        reconfigure: bool = False,
    ) -> DoclingDocument:
        """
        Convert a single URL into a DoclingDocument.
        Raises:
            RuntimeError      -- conversion failure
        """
        if reconfigure or not hasattr(self, "doc_converter"):
            self.setup_converter()

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

    async def _extract_title(self, doc: DoclingDocument) -> str:
        """
        Extracts the title from a DoclingDocument.
        Looks for SectionHeaderItem (level=1) first, then TitleItem, then doc.name.
        If none found, returns "Untitled".
        """
        # First priority: Look for main section headers (level=1) early in document
        for text_item in doc.texts[:10]:
            if (
                isinstance(text_item, SectionHeaderItem)
                and getattr(text_item, "level", None) == 1
                and getattr(text_item, "text", None)
                and text_item.text.strip()
            ):
                return text_item.text.strip()

        # Second priority: Look for any TitleItem
        for text_item in doc.texts:
            if (
                isinstance(text_item, TitleItem)
                and getattr(text_item, "text", None)
                and text_item.text.strip()
            ):
                return text_item.text.strip()

        # Third priority: Look for any level=1 SectionHeaderItem anywhere
        for text_item in doc.texts:
            if (
                isinstance(text_item, SectionHeaderItem)
                and getattr(text_item, "level", None) == 1
                and getattr(text_item, "text", None)
                and text_item.text.strip()
            ):
                return text_item.text.strip()

        # Fall back to document name
        if getattr(doc, "name", None) and doc.name.strip():
            return doc.name.strip()

        # Final fallback
        return "Untitled"

    async def _process_document(
        self,
        path: str,
    ) -> Tuple[str, WebpageMetadata]:
        """
        Full round-trip: fetch doc, export to markdown, clean, and build metadata (docling doesn't support metadata extraction).
        """
        doc = await self._get_docling_document(path)
        markdown = doc.export_to_markdown()
        markdown = await self._clean_markdown(markdown)

        metadata = WebpageMetadata(
            url=path,
            query=path,
            title=await self._extract_title(doc),
        )
        return markdown, metadata

    async def _arun(
        self,
        params: WebpageScraperToolInputSchema,
        **kwargs,
    ) -> WebpageScraperToolOutputSchema:
        """
        Entry point for external calls.
        Returns cleaned markdown + metadata, or raises a descriptive error.
        """
        path = str(params.url)
        try:
            md, meta = await self._process_document(path)
            return WebpageScraperToolOutputSchema(content=md, metadata=meta)

        except FileNotFoundError as e:
            # local file was missing
            raise RuntimeError(f"[File Not Found] {e}")

        except RuntimeError as e:
            # issues during conversion
            raise RuntimeError(f"[Conversion Error] {e}")

        except Exception as e:
            raise RuntimeError(f"[Internal Error] Failed to scrape {path}") from e
