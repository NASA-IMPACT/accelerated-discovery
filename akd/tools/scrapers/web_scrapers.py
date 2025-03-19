import asyncio
import json
import re
from abc import abstractmethod
from typing import List, Optional
from urllib.parse import urlparse

import requests
from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from markdownify import markdownify
from pydantic import Field, HttpUrl
from readability import Document

from ..search import SearchResultItem


class WebpageScraperToolInputSchema(BaseIOSchema):
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


class WebpageScraperToolOutputSchema(BaseIOSchema):
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

    def __init__(self, config: Optional[WebpageScraperToolConfig] = None):
        """
        Initializes the WebpageScraperTool.

        Args:
            config (WebpageScraperToolConfig): Configuration for the tool.
        """
        config = config or WebpageScraperToolConfig()
        super().__init__(config)
        self.user_agent = config.user_agent
        self.timeout = config.timeout
        self.max_content_length = config.max_content_length
        self.debug = config.debug
        self.config = config

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
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

    def _extract_metadata(
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
        domain = urlparse(url).netloc

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
                            metadata["keywords"] or scheme_data["keywords"]
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
                citation_parts.append(f'DOI: {metadata["doi"]}')

            if citation_parts:
                metadata["citation"] = ". ".join(citation_parts)

        # Clean up metadata values
        for key, value in metadata.items():
            if isinstance(value, str):
                metadata[key] = value.strip()

        if isinstance(metadata["keywords"], str):
            metadata["keywords"] = [metadata["keywords"]]
        return WebpageMetadata(**metadata)

    def _extract_title(self, soup: BeautifulSoup) -> str:
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

    @abstractmethod
    def run(
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

        raise NotImplementedError()


class SimpleWebScraper(WebScraperToolBase):
    """
    Tool for scraping webpage content and converting it to markdown format.
    """

    def _fetch_webpage(self, url: str) -> str:
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

    def _clean_markdown(self, markdown: str) -> str:
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

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
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

    def run(
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
        html_content = self._fetch_webpage(str(params.url))

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract main content using custom extraction
        main_content = self._extract_main_content(soup)

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
        markdown_content = self._clean_markdown(markdown_content)

        # Extract metadata
        metadata = self._extract_metadata(soup, Document(html_content), str(params.url))

        return WebpageScraperToolOutputSchema(
            content=markdown_content,
            metadata=metadata,
        )


class Crawl4AIWebScraper(WebScraperToolBase):
    async def fetch(self, url: str):
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(url=url)

    async def run_async(
        self,
        params: WebpageScraperToolInputSchema,
    ) -> WebpageScraperToolOutputSchema:
        """
        Async version of the run method.
        """
        if params.url.path.endswith((".pdf", ".PDF")):
            raise RuntimeError(f"Can't parse url with PDF :: {params.url}")

        crawl_result = await self.fetch(str(params.url))

        html_content = crawl_result.html

        soup = BeautifulSoup(html_content, "html.parser")
        metadata = self._extract_metadata(soup, Document(html_content), str(params.url))

        return WebpageScraperToolOutputSchema(
            content=crawl_result.markdown,
            metadata=metadata,
        )

    def run(
        self,
        params: WebpageScraperToolInputSchema,
    ) -> WebpageScraperToolOutputSchema:
        """
        Synchronous wrapper for run_async.
        """
        return asyncio.run(self.run_async(params))
