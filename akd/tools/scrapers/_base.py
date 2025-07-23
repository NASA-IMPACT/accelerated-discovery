import json
import os
import re
import tempfile
from typing import Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import AnyUrl, Field, computed_field
from readability import Document

from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig


class ScraperToolInputSchema(InputSchema):
    """
    Input schema for the ScraperTool.
    """

    url: AnyUrl = Field(
        ...,
        description="URL of the webpage to scrape.",
    )
    include_links: bool = Field(
        default=True,
        description="Whether to preserve hyperlinks in the markdown output.",
    )


class ScrapedMetadata(SearchResultItem):
    keywords: list[str] | None = Field(
        None,
        description="List of keywords from the webpage content.",
    )


class ScraperToolOutputSchema(OutputSchema):
    """Schema for the output of the ScraperTool."""

    content: str = Field(
        ...,
        description="The scraped content in markdown format.",
    )
    metadata: ScrapedMetadata = Field(
        ...,
        description="Metadata about the scraped webpage.",
    )


class ScraperToolConfig(BaseToolConfig):
    """Configuration for the ScraperTool."""

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


class ScraperToolBase(BaseTool[ScraperToolInputSchema, ScraperToolOutputSchema]):
    """
    Base class for web scraping tools.
    This class provides common functionality for scraping and extracting metadata.
    """

    input_schema = ScraperToolInputSchema
    output_schema = ScraperToolOutputSchema
    config_schema = ScraperToolConfig


class WebScraperToolConfig(ScraperToolConfig):
    """Configuration for the WebScraperTool."""

    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        description="User agent string to use for requests.",
    )
    accept_header: str = Field(
        default="text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        description="Accept header for HTTP requests.",
    )
    accept_language: str = Field(
        default="en-US,en;q=0.5",
        description="Accept-Language header for HTTP requests.",
    )

    @computed_field
    def headers(self) -> dict[str, str]:
        """
        Returns the headers to use for HTTP requests.
        This can be overridden in subclasses to customize headers.
        """
        return {
            "User-Agent": self.user_agent,
            "Accept": self.accept_header,
            "Accept-Language": self.accept_language,
            "Connection": "keep-alive",
        }


class WebScraper(ScraperToolBase):
    config_schema = WebScraperToolConfig

    async def _extract_author(self, soup: BeautifulSoup) -> str | None:
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
    ) -> ScrapedMetadata:
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
        return ScrapedMetadata(**metadata)

    async def _extract_title(self, soup: BeautifulSoup) -> str | None:
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


class PDFScraper(ScraperToolBase):
    @staticmethod
    def is_pdf_local(path: str) -> bool:
        """Checks if a local file path points to a PDF."""
        val_lower = path.lower()
        if not val_lower.startswith(("http://", "https://")):
            return val_lower.endswith(".pdf")
        return False

    @staticmethod
    async def is_pdf_remote(url: str) -> bool:
        """Check if a remote URL points to a PDF."""
        val_lower = url.lower()
        try:
            parsed = urlparse(val_lower)
            path = parsed.path
            # Check common PDF URL patterns
            if path.endswith(".pdf") or "/pdf" in path:
                return True

            # For uncertain cases, check content-type header
            async with httpx.AsyncClient() as client:
                response = await client.head(
                    url,
                    timeout=10.0,
                    follow_redirects=True,
                )
                content_type = response.headers.get("content-type", "").lower()
                return "application/pdf" in content_type
        except Exception:
            # Fallback to basic pattern matching if HTTP request fails
            try:
                parsed = urlparse(val_lower)
                return parsed.path.endswith(".pdf") or "/pdf" in parsed.path
            except Exception:
                return False
        return False

    @staticmethod
    async def is_pdf(val: str) -> bool:
        """
        Determines if a URL or file path points to a PDF.
        Args:
            val: URL or file path to check
        Returns:
            bool: True if the value points to a PDF
        """
        val_lower = val.lower()
        return PDFScraper.is_pdf_local(val_lower) or await PDFScraper.is_pdf_remote(
            val_lower,
        )

    async def _download_pdf_from_url(
        self,
        url: str,
    ) -> Tuple[str, tempfile.NamedTemporaryFile]:
        """
        Downloads a PDF from a URL and returns the path to the temporary file.

        Args:
            url: The URL of the PDF to download

        Returns:
            tuple[str, tempfile.NamedTemporaryFile]:
                containing the local path to the downloaded file
                and the temporary file object

        Raises:
            RuntimeError: If there is an error downloading the PDF
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        headers = self.headers
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()

                    # Check content length if provided
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > self.max_content_length:
                        raise RuntimeError(
                            f"PDF size ({content_length} bytes) exceeds maximum allowed size ({self.max_content_length} bytes)",
                        )

                    logger.debug(f"Downloading PDF at {temp_file.name}")
                    downloaded_bytes = 0
                    with open(temp_file.name, "wb") as f:
                        async for chunk in response.aiter_bytes(
                            chunk_size=self.chunk_size,
                        ):
                            downloaded_bytes += len(chunk)
                            # Safety check: ensure we don't exceed max_content_length
                            if downloaded_bytes > self.max_content_length:
                                raise RuntimeError(
                                    f"Downloaded size ({downloaded_bytes} bytes) exceeds maximum allowed size ({self.max_content_length} bytes)",
                                )
                            f.write(chunk)

            return temp_file.name, temp_file
        except Exception as e:
            os.unlink(temp_file.name)
            raise RuntimeError(f"Error downloading PDF from URL: {e}")
