import os
import re
import tempfile
from typing import Tuple
from urllib.parse import unquote

import fitz
import httpx
from loguru import logger
from markdownify import markdownify
from pydantic import AnyUrl, Field, field_validator

from ._base import (
    PDFScraper,
    ScrapedMetadata,
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
    WebScraperToolConfig,
)


class PDFScraperInputSchema(ScraperToolInputSchema):
    """
    Input schema for the PDFScraper tool.

    Accepts both web URLs and local file paths. File paths are automatically
    converted to file:// URLs for proper validation.

    Examples:
        - Web URL: "https://example.com/file.pdf"
        - Local file: "/Users/name/document.pdf"
        - Relative file: "./documents/file.pdf"
    """

    url: AnyUrl = Field(
        ...,
        description="The URL or local file path of the PDF to scrape.",
    )

    @field_validator("url", mode="before")
    @classmethod
    def convert_file_path_to_url(cls, v):
        """
        Convert file paths to file:// URLs for proper Pydantic validation.

        Args:
            v: Input value (URL string or file path)

        Returns:
            str: Properly formatted URL
        """
        if not isinstance(v, str):
            return v

        # Check if it's already a URL (has a scheme)
        if any(
            v.lower().startswith(scheme)
            for scheme in ["http://", "https://", "file://"]
        ):
            return v

        # Convert file path to file:// URL
        if os.path.isabs(v):
            # Absolute path
            # Handle Windows paths by converting backslashes to forward slashes
            normalized_path = v.replace("\\", "/")
            if normalized_path.startswith("/"):
                return f"file://{normalized_path}"
            else:
                # Windows absolute path like C:/path
                return f"file:///{normalized_path}"
        else:
            # Relative path
            return f"file://{v}"


class PDFScraperToolConfig(WebScraperToolConfig):
    chunk_size: int = Field(
        8192,
        description="The size of each chunk of PDF to process while downloading the PDF (in bytes).",
    )
    accept_header: str = Field(
        default="application/pdf,application/octet-stream,text/html,application/xhtml+xml,*/*;q=0.8",
        description="Accept header for HTTP requests with PDF as primary target.",
    )


class SimplePDFScraper(PDFScraper):
    input_schema = PDFScraperInputSchema
    config_schema = PDFScraperToolConfig

    async def _fetch_pdf_text(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file using PyMuPDF (fitz).
        """
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text("text") for page in doc])

            if not text.strip():
                raise ValueError(
                    "Extracted text is empty.The PDF may be scanned or image-based.",
                )

            return text
        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {e}")

    async def _extract_metadata(self, path: str) -> ScrapedMetadata:
        """
        Extracts metadata from a PDF file.
        """
        try:
            doc = fitz.open(path)
            metadata = doc.metadata
            keywords = (
                metadata.pop("keywords", "").split(",")
                if metadata.get("keywords")
                else None
            )
            return ScrapedMetadata(
                url=self._path_to_file_url(path),
                pdf_url=path
                if path.startswith("http")
                else self._path_to_file_url(path),
                query=str(path),
                title=metadata.pop("title", None),
                published_date=metadata.pop("creationDate", None),
                tags=keywords,
                extra=metadata,
            )
        except Exception as e:
            raise RuntimeError(f"Error extracting metadata from PDF: {e}")

    async def _clean_markdown(self, markdown: str) -> str:
        """
        Cleans up extracted markdown text.
        """
        markdown = re.sub(
            r"\n\s*\n\s*\n",
            "\n\n",
            markdown,
        )  # Remove excessive blank lines
        markdown = "\n".join(
            line.rstrip() for line in markdown.splitlines()
        )  # Trim trailing spaces
        markdown = markdown.strip() + "\n"
        return markdown

    @staticmethod
    def _is_url(val) -> bool:
        """Check if a value is a web URL (http or https)."""
        val_lower = str(val).lower()
        return val_lower.startswith(("http://", "https://"))

    def _path_to_file_url(self, path: str) -> str:
        """
        Convert local file path to file:// URL or return web URL as-is.

        Args:
            path: Local file path or web URL

        Returns:
            str: file:// URL for local paths, original URL for web URLs
        """
        if self._is_url(path):
            return path
        return f"file://{os.path.abspath(path)}"

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

    async def _process_pdf(self, path: str) -> tuple[str, ScrapedMetadata]:
        """
        Processes a PDF file, extracting text and metadata.

        Args:
            path: Path to the PDF file

        Returns:
            tuple containing the markdown content and metadata
        """
        # Extract text from the PDF
        extracted_text = await self._fetch_pdf_text(path)

        # Convert to markdown
        markdown_content = markdownify(extracted_text)

        # Clean markdown content
        markdown_content = await self._clean_markdown(markdown_content)

        # Extract metadata
        metadata = await self._extract_metadata(path)

        return markdown_content, metadata

    async def _arun(
        self,
        params: PDFScraperInputSchema,
        **kwargs,
    ) -> ScraperToolOutputSchema:
        """
        Runs the PDF scraper with a functional approach.

        Args:
            params: Input parameters containing the URL or path

        Returns:
            WebpageScraperToolOutputSchema with content and metadata
        """
        path = str(params.url)
        pdf_path = str(params.url)
        if not await self.is_pdf(path):
            raise RuntimeError(f"{path} is not a valid pdf.")
        temp_file = None
        try:
            # Handle URL vs local file path
            if self._is_url(path):
                pdf_path, temp_file = await self._download_pdf_from_url(path)
            else:
                pdf_path = unquote(params.url.path)
            markdown_content, metadata = await self._process_pdf(pdf_path)
            # Update metadata URLs to use proper format (file:// for local files, original for web URLs)
            file_url = self._path_to_file_url(path)
            metadata.url = file_url
            metadata.pdf_url = file_url
            return ScraperToolOutputSchema(
                content=markdown_content,
                metadata=metadata,
            )
        finally:
            # Clean up the temporary file if it exists
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
