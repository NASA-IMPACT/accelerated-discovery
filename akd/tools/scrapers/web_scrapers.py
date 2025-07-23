import re

import httpx
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from markdownify import markdownify
from readability import Document
from requests import HTTPError, RequestException

from ._base import ScraperToolInputSchema, ScraperToolOutputSchema, WebScraper


class SimpleWebScraper(WebScraper):
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
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()

                if len(response.content) > self.max_content_length:
                    raise ValueError(
                        f"Content length exceeds maximum of {self.max_content_length} bytes",
                    )

                return response.text

        except httpx.HTTPStatusError as http_err:
            raise HTTPError(f"HTTP error occurred: {http_err}")
        except httpx.TimeoutException as timeout_err:
            raise RequestException(f"Request timeout: {timeout_err}")
        except httpx.RequestError as req_err:
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
        params: ScraperToolInputSchema,
        **kwargs,
    ) -> ScraperToolOutputSchema:
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

        return ScraperToolOutputSchema(
            content=markdown_content.strip(),
            metadata=metadata,
        )


class Crawl4AIWebScraper(WebScraper):
    async def fetch(self, url: str):
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(url=url)

    async def _arun(
        self,
        params: ScraperToolInputSchema,
        **kwargs,
    ) -> ScraperToolOutputSchema:
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

        return ScraperToolOutputSchema(
            content=crawl_result.markdown.strip(),
            metadata=metadata,
        )
