import asyncio
import importlib.util
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from loguru import logger
from pydantic import Field

from ._base import (
    ScraperToolBase,
    ScraperToolConfig,
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
)
from .omni import DoclingScraper

# Detect if PyPaperBot module is available in the current environment
_PYPAPERBOT_AVAILABLE: bool = importlib.util.find_spec("PyPaperBot") is not None


class PyPaperBotScraperConfig(ScraperToolConfig):
    """Configuration options for PyPaperBotScraper."""

    max_runtime_seconds: int = Field(
        default=120,
        description="Maximum time allowed for PyPaperBot to run (seconds)",
    )
    scholar_pages: str = Field(
        default="1",
        description="Number of Google Scholar pages to inspect when using query fallback",
    )
    scholar_results: int = Field(
        default=7,
        ge=1,
        le=10,
        description="Number of results to download when scholar-pages=1",
    )
    enable_query_fallback: bool = Field(
        default=True,
        description="Use title-based query fallback if DOI cannot be extracted",
    )

    # Anna's Archive configuration (SciDB only)
    annas_archive_mirror: str = Field(
        default="https://annas-archive.se/",
        description="Anna's Archive (SciDB) mirror URL for PyPaperBot",
    )
    proxy: Optional[str] = Field(
        default=None,
        description="Comma-separated proxies string understood by PyPaperBot --proxy",
    )
    single_proxy: Optional[str] = Field(
        default=None,
        description="Single proxy URL for PyPaperBot --single-proxy",
    )
    selenium_chrome_version: Optional[int] = Field(
        default=None,
        description="First three digits of local Chrome version to enable Selenium path in PyPaperBot",
    )


class PyPaperBotScraper(ScraperToolBase):
    """
    Scraper that uses PyPaperBot package to fetch full-text PDFs via DOI or query
    from Anna's Archive (SciDB), then converts the file to Markdown using Docling.

    Reference: ferru97/PyPaperBot on GitHub.
    """

    input_schema = ScraperToolInputSchema
    output_schema = ScraperToolOutputSchema
    config_schema = PyPaperBotScraperConfig

    def __init__(
        self,
        config: PyPaperBotScraperConfig | None = None,
        scraper: ScraperToolBase | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(config=config or PyPaperBotScraperConfig(), debug=debug)
        self.scraper = scraper or DoclingScraper(debug=debug)
        logger.info(
            f"Initialized PyPaperBotScraper with config: {self.config} | scraper={self.scraper.__class__.__name__}",
        )

    def _find_downloaded_pdf(self, directory: Path) -> Optional[Path]:
        """Find first PDF file in directory."""
        try:
            for pdf_file in directory.glob("*.pdf"):
                if pdf_file.is_file() and pdf_file.stat().st_size > 0:
                    return pdf_file
        except Exception as e:
            if self.debug:
                logger.debug(f"PDF search failed in {directory}: {e}")
        return None

    async def _run_pypaperbot_with_doi(self, doi: str, out_dir: Path) -> Optional[Path]:
        """Run PyPaperBot with DOI."""
        if not _PYPAPERBOT_AVAILABLE:
            if self.debug:
                logger.debug("PyPaperBot package not available")
            return None

        try:
            cmd = [
                sys.executable,
                "-m",
                "PyPaperBot",
                "--doi",
                doi,
                "--dwn-dir",
                str(out_dir),
            ]

            # Add Anna's Archive mirror (required)
            cmd.extend(["--annas-archive-mirror", self.config.annas_archive_mirror])

            # Add optional parameters from config
            if self.config.proxy:
                cmd.extend(["--proxy", self.config.proxy])
            if self.config.single_proxy:
                cmd.extend(["--single-proxy", self.config.single_proxy])
            if self.config.selenium_chrome_version:
                cmd.extend(
                    [
                        "--selenium-chrome-version",
                        str(self.config.selenium_chrome_version),
                    ],
                )

            if self.debug:
                logger.debug(f"Running PyPaperBot command: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=out_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.max_runtime_seconds,
                )

                if self.debug:
                    logger.debug(f"PyPaperBot stdout: {stdout.decode()}")
                    if stderr:
                        logger.debug(f"PyPaperBot stderr: {stderr.decode()}")

                return self._find_downloaded_pdf(out_dir)
            except asyncio.TimeoutError:
                process.kill()
                if self.debug:
                    logger.debug(f"PyPaperBot timeout for DOI: {doi}")
                return None
        except Exception as e:
            if self.debug:
                logger.debug(f"PyPaperBot execution failed for DOI {doi}: {e}")
        return None

    async def _run_pypaperbot_with_query(
        self,
        query: str,
        out_dir: Path,
    ) -> Optional[Path]:
        """Run PyPaperBot with search query."""
        if not _PYPAPERBOT_AVAILABLE or not self.config.enable_query_fallback:
            return None

        try:
            cmd = [
                sys.executable,
                "-m",
                "PyPaperBot",
                "--query",
                query,
                "--scholar-pages",
                self.config.scholar_pages,
                "--scholar-results",
                str(self.config.scholar_results),
                "--dwn-dir",
                str(out_dir),
            ]

            # Add Anna's Archive mirror (required)
            cmd.extend(["--annas-archive-mirror", self.config.annas_archive_mirror])

            # Add optional parameters from config
            if self.config.proxy:
                cmd.extend(["--proxy", self.config.proxy])
            if self.config.selenium_chrome_version:
                cmd.extend(
                    [
                        "--selenium-chrome-version",
                        str(self.config.selenium_chrome_version),
                    ],
                )

            if self.debug:
                logger.debug(f"Running PyPaperBot command: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=out_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.max_runtime_seconds,
                )

                if self.debug:
                    logger.debug(f"PyPaperBot stdout: {stdout.decode()}")
                    if stderr:
                        logger.debug(f"PyPaperBot stderr: {stderr.decode()}")

                return self._find_downloaded_pdf(out_dir)
            except asyncio.TimeoutError:
                process.kill()
                if self.debug:
                    logger.debug(f"PyPaperBot timeout for query: {query}")
                return None
        except Exception as e:
            if self.debug:
                logger.debug(f"PyPaperBot execution failed for query {query}: {e}")
        return None

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """Extract DOI from URL using simple pattern matching."""
        import re

        # Common DOI patterns
        doi_patterns = [
            r"10\.\d{4,}/[^\s\"<>#]+",  # Standard DOI format
            r"/doi/(?:full/|pdf/|pdfdirect/)?(10\.[^/?#]+)",  # Wiley DOI extraction
        ]

        for pattern in doi_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                # Return the full DOI or the captured group
                return match.group(1) if match.groups() else match.group(0)

        return None

    def _extract_query_from_url(self, url: str) -> Optional[str]:
        """Extract a reasonable search query from URL by removing domain and using the rest."""
        res = url
        try:
            parsed = urlparse(url)
            # Use path + query as search term, removing leading slash
            query_term = (parsed.path + parsed.query).lstrip("/")
            # Clean up common file extensions
            res = re.sub(r"\.(pdf|html?|aspx?)$", "", query_term, flags=re.IGNORECASE)
        except Exception as e:
            if self.debug:
                logger.debug(f"Error extracting query from URL {url}: {e}")
        return res

    async def _arun(
        self,
        params: ScraperToolInputSchema,
        **_kwargs,
    ) -> ScraperToolOutputSchema:
        """Main execution method."""
        url_str = str(params.url)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # First, try to extract DOI from URL
            doi = self._extract_doi_from_url(url_str)
            pdf_path = None

            if doi:
                if self.debug:
                    logger.debug(f"Extracted DOI: {doi}")
                pdf_path = await self._run_pypaperbot_with_doi(doi, tmp_path)

            if self.debug:
                logger.debug(f"PDF path after DOI search: {pdf_path}")
            # Fallback to query-based search if no DOI found or DOI search failed
            if not pdf_path and self.config.enable_query_fallback:
                query = self._extract_query_from_url(url_str)
                if query:
                    if self.debug:
                        logger.debug(f"Falling back to query search: {query}")
                    pdf_path = await self._run_pypaperbot_with_query(query, tmp_path)

            # Convert PDF to markdown using Docling if PDF found
            if pdf_path and pdf_path.exists():
                try:
                    result = await self.scraper.arun(
                        self.scraper.input_schema(url=str(pdf_path)),
                    )

                    if result.content and result.content.strip():
                        # Update metadata with original URL
                        result.metadata.url = params.url
                        result.metadata.query = doi or url_str
                        return result
                except Exception as e:
                    if self.debug:
                        logger.debug(f"Docling conversion failed: {e}")

            # Return empty result if all strategies failed
            from ._base import ScrapedMetadata

            return ScraperToolOutputSchema(
                content="",
                metadata=ScrapedMetadata(
                    url=params.url,
                    title="",
                    query=doi or url_str,
                ),
            )
