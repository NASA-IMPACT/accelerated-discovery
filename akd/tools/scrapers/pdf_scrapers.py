import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz
import requests
from loguru import logger
from markdownify import markdownify
from pydantic import Field, FilePath, HttpUrl
from bs4 import BeautifulSoup

from .web_scrapers import (
    WebpageMetadata,
    WebpageScraperToolInputSchema,
    WebpageScraperToolOutputSchema,
    WebScraperToolBase,
)

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.types import DoclingDocument


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

    input_schema: PDFScraperToolInputSchema
    output_schema: WebpageScraperToolOutputSchema

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
        except Exception:
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

    async def _arun(
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


class DoclingPDFScraper(WebScraperToolBase):

    def __init__(self, 
                 mode: str = 'accurate',
                 export_type: str = 'markdown'):
        self.mode = mode
        self.export_type = export_type


    async def _get_docling_document(self, pdf_path: str, mode: str = 'accurate') -> DoclingDocument:
        """
        Processes a PDF file or URL, returns a Docling Document.

        Args:
            path: Path to the PDF file / url

        Returns:
            Docling document
        """
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        if mode == 'accurate':
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE 
        else:
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST 
        doc_converter = DocumentConverter(
            allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.IMAGE,
                ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = doc_converter.convert(pdf_path)
        return result.document
    

    def _parse_html(html_content: str) -> str:
        """
        Parses html content from a docling document into a string

        Args:
            path: HTML content as a string

        Returns:
            str
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        parsed_text = []
        sections = soup.find_all('h2')
        for section in sections:
            section_title = section.get_text(strip=True)
            parsed_text.append(f"\n{section_title}\n{'=' * len(section_title)}")
            section_content = []
            sibling = section.find_next_sibling()
            while sibling and sibling.name != 'h2':
                if sibling.name in ['p', 'ul', 'li']:
                    section_content.append(sibling.get_text(strip=True))
                elif sibling.name == 'table':
                    caption = sibling.find('caption')
                    if caption:
                        caption_text = caption.get_text(strip=True)
                        section_content.append(caption_text)
                elif sibling.name == 'figure':
                    figcaption = sibling.find('figcaption')
                    if figcaption:
                        figure_text = figcaption.get_text(strip=True)
                        section_content.append(figure_text)
                sibling = sibling.find_next_sibling()
            parsed_text.append('\n'.join(section_content))
        return '\n\n'.join(parsed_text)
    
    
    async def _process_pdf(self, path: str) -> tuple[str, PdfMetadata]:
        """
        Processes a PDF file, extracting text and metadata.

        Args:
            path: Path to the PDF file

        Returns:
            tuple containing the markdown content and metadata
        """
        # Extract text from PDF
        docling_document = await self._get_docling_document(path)

        # Convert to markdown
        if self.export_type == 'markdown':  
            markdown_content = docling_document.export_to_markdown()
        else:
            html_content = docling_document.export_to_html()
            extracted_text = self._parse_html(html_content)
            markdown_content = markdownify(extracted_text)

        # Clean markdown content
        markdown_content = await self._clean_markdown(markdown_content)

        # Docling does not support metadata extraction
        metadata = None

        return markdown_content, metadata
    

    async def _arun(
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