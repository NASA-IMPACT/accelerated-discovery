import os
import tempfile
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import HttpUrl

from ._base import (
    ScrapedMetadata,
    ScraperToolBase,
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
)
from .omni import DoclingScraper, OmniScraperInputSchema
from .pypaperbot_scraper import PyPaperBotScraper
from .resolvers import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema


class CompositeScraper(ScraperToolBase):
    """
    Composite web scraper that runs multiple scrapers in waterfall manner/sequence.

    The idea is:
    - Start with the first scraper
    - If it produces a result, return it
    - If it fails, try the next scraper in the list

    Usage:
    ```python
    from akd.tools.scrapers._base import PDFScraper
    from akd.tools.scrapers import (
        ScraperToolInputSchema,
        ScraperToolOutputSchema,
    )
    from akd.tools.scrapers import (
        SimpleWebScraper,
        Crawl4AIWebScraper,
    )

    from akd.tools.scrapers import (
        SimplePDFScraper,
        PDFScraperInputSchema,
    )

    from akd.tools.scrapers import DoclingScraper

    from akd.tools.scrapers.composite import CompositeScraper

    _url = "https://arxiv.org/pdf/2402.01822v1"
    # _url = "https://nishparadox.com/writing/vipassana-retreat.html"

    scraper = CompositeScraper(
        DoclingScraper(),
        Crawl4AIWebScraper(),
        SimpleWebScraper(),
        SimplePDFScraper(),
    )

    output = await _scraper.arun(ScraperToolInputSchema(url=_url))
    ```
    """

    def __init__(
        self,
        *scrapers: ScraperToolBase,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.scrapers = scrapers

    async def _arun(
        self,
        params: ScraperToolInputSchema,
        **kwargs,
    ) -> ScraperToolOutputSchema:
        result = ScraperToolOutputSchema(
            content="",
            metadata=ScrapedMetadata(
                url=str(params.url),
                title="",
                query="",
            ),
        )
        for scraper in self.scrapers:
            try:
                if self.debug:
                    logger.debug(
                        f"Running scraper={scraper.__class__.__name__} for {params}",
                    )

                result = await scraper.arun(scraper.input_schema(**params.model_dump()))
                if result.content.strip():
                    result.metadata.extra = result.metadata.extra or {}
                    result.metadata.extra["scraper"] = scraper.__class__.__name__
                    result.metadata.extra["scraper_config"] = scraper.config
                    break
            except Exception as e:
                logger.error(
                    f"Error running {scraper.__class__.__name__}\n{str(e)}",
                )
                continue
        return result


class PrefetchingDoclingScraper(ScraperToolBase):
    """
    Wrapper scraper that prefetches the target URL using browser-like headers
    (including Referer), writes it to a temporary file, then invokes DoclingScraper
    on the local file. This helps avoid 403s from sites that block direct fetches
    without a Referer or specific headers.
    """

    input_schema = ScraperToolInputSchema
    output_schema = ScraperToolOutputSchema

    async def _resolve_url(self, url_str: str) -> str:
        """Best-effort resolver for common scholarly domains.
        - DOI: follow redirects to landing page
        - arXiv: map /abs/ → /pdf/<id>.pdf
        - Wiley/ScienceDirect: try to discover direct PDF link from HTML meta
        """
        try:
            parsed = urlparse(url_str)
            host = parsed.netloc.lower()

            # arXiv abs → pdf
            if "arxiv.org" in host and "/abs/" in parsed.path:
                paper_id = parsed.path.split("/abs/")[-1]
                return f"https://arxiv.org/pdf/{paper_id}.pdf"

            # DOI resolver
            if host in {"doi.org", "dx.doi.org"}:
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=10.0
                ) as client:
                    try:
                        r = await client.head(url_str)
                        r.raise_for_status()
                        return str(r.url)
                    except Exception:
                        # Fallback to GET without downloading body (httpx always downloads; we accept it)
                        r = await client.get(url_str)
                        r.raise_for_status()
                        return str(r.url)

            # Direct Wiley PDF URL construction without fetching HTML (avoid 403 on HTML pages)
            if "onlinelibrary.wiley.com" in host and "/doi/" in parsed.path:
                doi_path = parsed.path
                pdfdirect = doi_path.replace("/doi/", "/doi/pdfdirect/")
                return urljoin(url_str, pdfdirect)

            # ScienceDirect: try direct PDF endpoints using PII
            if "sciencedirect.com" in host and "/article/pii/" in parsed.path:
                # Prefer pdfft first (often works without cookies)
                pdfft = urljoin(
                    url_str,
                    parsed.path.rstrip("/") + "/pdfft?isDTMRedir=true&download=true",
                )
                return pdfft

            # Try to extract direct PDF from HTML for certain publishers (when HTML is accessible)
            if any(d in host for d in ["sciencedirect.com", "onlinelibrary.wiley.com"]):
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": f"{parsed.scheme}://{parsed.netloc}",
                    "Connection": "keep-alive",
                }
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=15.0, headers=headers
                ) as client:
                    resp = await client.get(url_str)
                    if (
                        resp.status_code == 200
                        and "text/html" in resp.headers.get("content-type", "").lower()
                    ):
                        soup = BeautifulSoup(resp.text, "html.parser")
                        meta_pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})
                        if meta_pdf and meta_pdf.get("content"):
                            return urljoin(url_str, meta_pdf["content"])
                        # Wiley heuristic: /doi/pdf/ or /doi/pdfdirect/
                        if "onlinelibrary.wiley.com" in host and "/doi/" in parsed.path:
                            doi_path = parsed.path
                            pdf2 = doi_path.replace("/doi/", "/doi/pdfdirect/")
                            return urljoin(url_str, pdf2)  # prefer pdfdirect
        except Exception as e:
            if self.debug:
                logger.debug(f"URL resolve fallback failed for {url_str}: {e}")
        return url_str

    async def _arun(
        self,
        params: ScraperToolInputSchema,
        **kwargs,
    ) -> ScraperToolOutputSchema:
        url_str = await self._resolve_url(str(params.url))
        tmp_path: str | None = None
        # Prepare browser-like headers with Referer derived from URL
        parsed = urlparse(url_str)
        referer = f"{parsed.scheme}://{parsed.netloc}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Referer": referer,
        }
        try:
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=30.0,
                    headers=headers,
                ) as client:
                    resp = await client.get(url_str)
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "").lower()
                    suffix = (
                        ".pdf"
                        if "pdf" in ctype or url_str.lower().endswith(".pdf")
                        else ".html"
                    )
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(resp.content)
                    tmp.flush()
                    tmp.close()
                    tmp_path = tmp.name

                # Invoke Docling on the downloaded file
                docling = DoclingScraper(debug=self.debug)
                out = await docling.arun(OmniScraperInputSchema(url=tmp_path))
                if out.content and out.content.strip():
                    return out

                # If Docling result is empty or too short, try PyPaperBot as a fallback
                try:
                    pypb = PyPaperBotScraper(debug=self.debug)
                    out_pb = await pypb.arun(
                        ScraperToolInputSchema(url=str(params.url))
                    )
                    return out_pb
                except Exception as e:
                    if self.debug:
                        logger.debug(f"PyPaperBot fallback failed: {e}")
                    return out
            except Exception as e:
                if self.debug:
                    logger.debug(f"Prefetch failed for {url_str}: {e}")
                # On prefetch failure (e.g., 403), try PyPaperBot directly
                pypb = PyPaperBotScraper(debug=self.debug)
                return await pypb.arun(ScraperToolInputSchema(url=str(params.url)))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass


class ResearchArticleResolver(BaseArticleResolver):
    def __init__(self, *resolvers: BaseArticleResolver, debug: bool = False) -> None:
        super().__init__(debug=debug)
        self.resolvers = resolvers

    async def validate_url(self, url: HttpUrl | str) -> bool:
        return super().validate_url(url)

    async def resolve(self, url: str | HttpUrl) -> ResolverOutputSchema:
        original_url = str(url)
        rname = self.__class__.__name__
        output = ResolverOutputSchema(
            url=original_url,
            resolver=rname,
        )
        for resolver in self.resolvers:
            rname = resolver.__class__.__name__
            try:
                logger.debug(f"Using resolver={rname} for url={original_url}")
                output = await resolver.arun(ResolverInputSchema(url=original_url))
                if not output.url:
                    raise ValueError("Resolver failure")
                if output.url:
                    break
            except Exception:
                logger.error(f"Error using resolver={rname}")
        return output

    async def _arun(
        self,
        params: ResolverInputSchema,
        **kwargs,
    ) -> ResolverOutputSchema:
        output = await self.resolve(params.url)
        if not output.url:
            output.url = params.url
            output.resolver = self.__class__.__name__
        return output
