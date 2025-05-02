import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz
import requests
from loguru import logger
from markdownify import markdownify
from pydantic import Field, FilePath, HttpUrl

from .web_scrapers import (
    WebpageMetadata,
    WebpageScraperToolInputSchema,
    WebpageScraperToolOutputSchema,
    WebScraperToolBase,
)


class PdfMetadata(WebpageMetadata):
    url: Union[HttpUrl, FilePath] = Field(
        ...,
        description="The URL or local file path of the search result",
    )
    pdf_url: Optional[Union[HttpUrl, FilePath]] = Field(
        None,
        description="The PDF URL or local file path of the search paper",
    )
    title: str = Field(
        ...,
        description="The title of the search result",
    )
    content: Optional[str] = Field(
        None,
        description="The content snippet of the search result",
    )
    query: str = Field(
        ...,
        description="The query used to obtain the search result",
    )
    category: Optional[str] = Field(
        None,
        description="Category of the search result",
    )
    doi: Optional[str] = Field(
        None,
        description="Digital Object Identifier (DOI) of the pdf",
    )
    published_date: Optional[str] = Field(
        None,
        description="Publication date for the pdf",
    )
    engine: Optional[str] = Field(
        None,
        description="Engine that fetched the search result",
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Tags for the search result",
    )
    extra: Optional[Dict[str, Any]] = Field(
        None,
        description="Extra information from the search result",
    )


class PDFScraperToolInputSchema(WebpageScraperToolInputSchema):
    """
    Input schema for the PDFScraper
    """

    url: Union[HttpUrl, FilePath] = Field(..., description="Path of the pdf")


class SimplePDFScraper(WebScraperToolBase):
    """
    Tool for extracting content from PDF files
    and converting it to markdown format.
    """

    async def _fetch_pdf_text(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file using PyMuPDF (fitz).
        """
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text("text") for page in doc])

            if not text.strip():
                raise ValueError(
                    "Extracted text is empty." "The PDF may be scanned or image-based.",
                )

            return text
        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {e}")

    async def _extract_metadata(self, path: str) -> PdfMetadata:
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
            return PdfMetadata(
                url=path if path.startswith("http") else FilePath(path),
                pdf_url=FilePath(path)
                if not path.startswith("http")
                else HttpUrl(path),
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
        res = True
        try:
            _ = HttpUrl(val)
        except:
            res = False
        return res

    @staticmethod
    def _is_pdf(val) -> bool:
        return val.lower().endswith(".pdf")

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
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/pdf",
        }
        try:
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()

            logger.debug(f"Downloading PDF at {temp_file.name}")
            with open(temp_file.name, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return temp_file.name, temp_file
        except Exception as e:
            os.unlink(temp_file.name)
            raise RuntimeError(f"Error downloading PDF from URL: {e}")

    async def _process_pdf(self, path: str) -> tuple[str, PdfMetadata]:
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

    async def arun(
        self,
        params: WebpageScraperToolInputSchema,
    ) -> WebpageScraperToolOutputSchema:
        """
        Runs the PDF scraper with a functional approach.

        Args:
            params: Input parameters containing the URL or path

        Returns:
            WebpageScraperToolOutputSchema with content and metadata
        """
        path = str(params.url)
        pdf_path = str(params.url)
        if not self._is_pdf(path):
            raise RuntimeError(f"{path} is not a valid pdf.")
        temp_file = None
        try:
            # Handle URL vs local file path
            if self._is_url(path):
                pdf_path, temp_file = await self._download_pdf_from_url(path)
            markdown_content, metadata = await self._process_pdf(pdf_path)
            metadata.url = metadata.pdf_url = path
            return WebpageScraperToolOutputSchema(
                content=markdown_content,
                metadata=metadata,
            )
        finally:
            # Clean up the temporary file if it exists
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
