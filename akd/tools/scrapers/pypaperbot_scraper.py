import asyncio
import importlib.util
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import Field

from ._base import ScraperToolBase, ScraperToolInputSchema, ScraperToolOutputSchema
from .omni import DoclingScraper, OmniScraperInputSchema

# Detect if PyPaperBot module is available in the current environment
_PYPAPERBOT_AVAILABLE: bool = importlib.util.find_spec("PyPaperBot") is not None


class PyPaperBotScraperConfig(ScraperToolBase.config_schema):
    """
    Configuration options for PyPaperBotScraper.
    """

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
    scihub_mirror: Optional[str] = Field(
        default_factory=lambda: os.environ.get("AKD_SCIHUB_MIRROR"),
        description="Preferred Sci-Hub mirror for PyPaperBot",
    )
    annas_archive_mirror: Optional[str] = Field(
        default_factory=lambda: os.environ.get("AKD_ANNAS_ARCHIVE_MIRROR"),
        description="Preferred Anna's Archive (SciDB) mirror for PyPaperBot",
    )
    proxy: Optional[str] = Field(
        default_factory=lambda: os.environ.get("AKD_PYPAPERBOT_PROXY")
        or os.environ.get("AKD_PROXY"),
        description="Comma-separated proxies string understood by PyPaperBot --proxy",
    )
    single_proxy: Optional[str] = Field(
        default_factory=lambda: os.environ.get("AKD_PYPAPERBOT_SINGLE_PROXY"),
        description="Single proxy URL for PyPaperBot --single-proxy",
    )
    selenium_chrome_version: Optional[int] = Field(
        default_factory=lambda: (
            int(os.environ.get("AKD_SELENIUM_CHROME_VERSION", "0")) or None
        ),
        description="First three digits of local Chrome version to enable Selenium path in PyPaperBot",
    )


class PyPaperBotScraper(ScraperToolBase):
    """
    Scraper that uses PyPaperBot to fetch full-text PDFs via DOI or query,
    then converts the file to Markdown using Docling.

    Reference: ferru97/PyPaperBot on GitHub.
    """

    input_schema = ScraperToolInputSchema
    output_schema = ScraperToolOutputSchema
    config_schema = PyPaperBotScraperConfig

    DOI_REGEX = re.compile(r"10\.\d{4,9}/[^\s\"<>#]+", re.IGNORECASE)

    async def _fetch_html(self, url: str) -> Optional[str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Referer": f"{httpx.URL(url).scheme}://{httpx.URL(url).host or ''}",
        }
        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=15.0, headers=headers
            ) as client:
                r = await client.get(url)
                if (
                    r.status_code == 200
                    and "html" in r.headers.get("content-type", "").lower()
                ):
                    return r.text
        except Exception as e:
            if self.debug:
                logger.debug(f"HTML fetch failed for {url}: {e}")
        return None

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        try:
            # Direct DOI resolver
            if any(host in url for host in ["doi.org", "dx.doi.org"]):
                m = self.DOI_REGEX.search(url)
                if m:
                    return m.group(0)

            # Wiley
            if "onlinelibrary.wiley.com" in url and "/doi/" in url:
                m = re.search(r"/doi/(?:full/|pdf/|pdfdirect/)?(10\.[^/?#]+)", url)
                if m:
                    return m.group(1)

            # Fallback generic DOI pattern in URL
            m = self.DOI_REGEX.search(url)
            if m:
                return m.group(0)
        except Exception:
            return None
        return None

    def _extract_doi_from_html(self, html: str) -> Optional[str]:
        try:
            soup = BeautifulSoup(html, "html.parser")
            tag = soup.find("meta", attrs={"name": "citation_doi"})
            if tag and tag.get("content"):
                return tag["content"].strip()
            text = soup.get_text(" ", strip=True)
            m = self.DOI_REGEX.search(text)
            if m:
                return m.group(0)
        except Exception:
            return None
        return None

    def _extract_title_from_html(self, html: str) -> Optional[str]:
        try:
            soup = BeautifulSoup(html, "html.parser")
            og = soup.find("meta", property="og:title")
            if og and og.get("content"):
                return og["content"].strip()
            title_tag = soup.find("title")
            if title_tag and title_tag.string:
                return title_tag.string.strip()
            h1 = soup.find("h1")
            if h1 and h1.get_text().strip():
                return h1.get_text().strip()
        except Exception:
            return None
        return None

    async def _download_pdf_to(self, pdf_url: str, out_dir: Path) -> Optional[Path]:
        try:
            url = httpx.URL(pdf_url)
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
                "Referer": f"{url.scheme}://{url.host or ''}",
                "Connection": "keep-alive",
            }
            async with httpx.AsyncClient(
                headers=headers, follow_redirects=True, timeout=30.0
            ) as client:
                r = await client.get(str(url))
                r.raise_for_status()
                ctype = r.headers.get("content-type", "").lower()
                if "pdf" not in ctype and not str(url).lower().endswith(".pdf"):
                    return None
                out_dir.mkdir(parents=True, exist_ok=True)
                tmp = tempfile.NamedTemporaryFile(
                    dir=str(out_dir), delete=False, suffix=".pdf"
                )
                tmp.write(r.content)
                tmp.flush()
                tmp.close()
                return Path(tmp.name)
        except Exception as e:
            if self.debug:
                logger.debug(f"Direct PDF download failed from {pdf_url}: {e}")
            return None

    async def _fetch_pdf_via_semanticscholar(
        self, doi: str, out_dir: Path
    ) -> Optional[Path]:
        try:
            api = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(api)
                if r.status_code != 200:
                    return None
                data = r.json()
                oa = data.get("openAccessPdf") or {}
                pdf_url = oa.get("url")
                if isinstance(pdf_url, str) and pdf_url:
                    return await self._download_pdf_to(pdf_url, out_dir)
        except Exception as e:
            if self.debug:
                logger.debug(f"Semantic Scholar OA lookup failed for {doi}: {e}")
        return None

    async def _fetch_pdf_via_unpaywall(self, doi: str, out_dir: Path) -> Optional[Path]:
        email = os.environ.get("UNPAYWALL_EMAIL")
        if not email:
            return None
        try:
            api = f"https://api.unpaywall.org/v2/{doi}?email={email}"
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(api)
                if r.status_code != 200:
                    return None
                data = r.json()
                loc = data.get("best_oa_location") or {}
                pdf_url = loc.get("url_for_pdf") or loc.get("url")
                if not pdf_url and isinstance(data.get("oa_locations"), list):
                    for location in data["oa_locations"]:
                        if location.get("url_for_pdf"):
                            pdf_url = location.get("url_for_pdf")
                            break
                if isinstance(pdf_url, str) and pdf_url:
                    return await self._download_pdf_to(pdf_url, out_dir)
        except Exception as e:
            if self.debug:
                logger.debug(f"Unpaywall OA lookup failed for {doi}: {e}")
        return None

    async def _fetch_pdf_via_crossref(self, doi: str, out_dir: Path) -> Optional[Path]:
        try:
            api = f"https://api.crossref.org/works/{doi}"
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(api)
                if r.status_code != 200:
                    return None
                data = r.json().get("message", {})
                links = data.get("link", [])
                for lk in links:
                    url = lk.get("URL") or lk.get("url")
                    ctype = lk.get("content-type") or ""
                    if isinstance(url, str) and "pdf" in ctype.lower():
                        path = await self._download_pdf_to(url, out_dir)
                        if path is not None:
                            return path
        except Exception as e:
            if self.debug:
                logger.debug(f"Crossref link lookup failed for {doi}: {e}")
        return None

    async def _run_pypaperbot_with_doi(self, doi: str, out_dir: Path) -> None:
        if not _PYPAPERBOT_AVAILABLE:
            raise RuntimeError(
                "PyPaperBot is not installed. Install it to enable Sci-Hub/Scholar PDF lookup."
            )
        cmd = [
            sys.executable,
            "-m",
            "PyPaperBot",
            f"--doi={doi}",
            f"--dwn-dir={str(out_dir)}",
            "--use-doi-as-filename",
        ]
        if self.config.scihub_mirror:
            cmd.append(f"--scihub-mirror={self.config.scihub_mirror}")
        if self.config.annas_archive_mirror:
            cmd.append(f"--annas-archive-mirror={self.config.annas_archive_mirror}")
        if self.config.proxy:
            cmd.append(f"--proxy={self.config.proxy}")
        if self.config.single_proxy:
            cmd.append(f"--single-proxy={self.config.single_proxy}")
        if isinstance(self.config.selenium_chrome_version, int):
            cmd.append(
                f"--selenium-chrome-version={self.config.selenium_chrome_version}"
            )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=float(self.config.max_runtime_seconds)
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise RuntimeError("PyPaperBot timed out while fetching DOI")
        if proc.returncode != 0:
            raise RuntimeError(
                f"PyPaperBot failed (DOI). rc={proc.returncode}, stderr={stderr.decode(errors='ignore')[:500]}"
            )

    async def _run_pypaperbot_with_query(self, query: str, out_dir: Path) -> None:
        if not _PYPAPERBOT_AVAILABLE:
            raise RuntimeError(
                "PyPaperBot is not installed. Install it to enable Sci-Hub/Scholar PDF lookup."
            )
        cmd = [
            sys.executable,
            "-m",
            "PyPaperBot",
            f"--query={query}",
            f"--scholar-pages={self.config.scholar_pages}",
            f"--scholar-results={int(self.config.scholar_results)}",
            f"--dwn-dir={str(out_dir)}",
        ]
        if self.config.scihub_mirror:
            cmd.append(f"--scihub-mirror={self.config.scihub_mirror}")
        if self.config.annas_archive_mirror:
            cmd.append(f"--annas-archive-mirror={self.config.annas_archive_mirror}")
        if self.config.proxy:
            cmd.append(f"--proxy={self.config.proxy}")
        if self.config.single_proxy:
            cmd.append(f"--single-proxy={self.config.single_proxy}")
        if isinstance(self.config.selenium_chrome_version, int):
            cmd.append(
                f"--selenium-chrome-version={self.config.selenium_chrome_version}"
            )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=float(self.config.max_runtime_seconds)
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise RuntimeError("PyPaperBot timed out while fetching query")
        if proc.returncode != 0:
            raise RuntimeError(
                f"PyPaperBot failed (query). rc={proc.returncode}, stderr={stderr.decode(errors='ignore')[:500]}"
            )

    def _find_downloaded_pdf(self, out_dir: Path) -> Optional[Path]:
        for root, _dirs, files in os.walk(out_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    return Path(root) / f
        return None

    async def _arun(
        self,
        params: ScraperToolInputSchema,
        **kwargs,
    ) -> ScraperToolOutputSchema:
        url_str = str(params.url)
        tmp_dir = Path(tempfile.mkdtemp(prefix="pypaperbot_"))
        pdf_path: Optional[Path] = None
        try:
            doi = self._extract_doi_from_url(url_str)
            html: Optional[str] = None
            if doi is None:
                html = await self._fetch_html(url_str)
                if html:
                    doi = self._extract_doi_from_html(html)

            if doi:
                try:
                    # 1) Try Crossref declared PDF links
                    pdf_path = await self._fetch_pdf_via_crossref(doi, tmp_dir)
                    if pdf_path is None:
                        # 2) Try Semantic Scholar OA PDF
                        pdf_path = await self._fetch_pdf_via_semanticscholar(
                            doi, tmp_dir
                        )
                    if pdf_path is None:
                        # 3) Try Unpaywall OA PDF
                        pdf_path = await self._fetch_pdf_via_unpaywall(doi, tmp_dir)
                    if pdf_path is None and _PYPAPERBOT_AVAILABLE:
                        # 4) Try PyPaperBot DOI path (SciHub/Scholar)
                        await self._run_pypaperbot_with_doi(doi, tmp_dir)
                        pdf_path = self._find_downloaded_pdf(tmp_dir)
                except Exception as e:
                    if self.debug:
                        logger.debug(f"PyPaperBot DOI path failed: {e}")

            if (
                pdf_path is None
                and self.config.enable_query_fallback
                and _PYPAPERBOT_AVAILABLE
            ):
                if html is None:
                    html = await self._fetch_html(url_str)
                title = self._extract_title_from_html(html or "") if html else None
                if title:
                    try:
                        await self._run_pypaperbot_with_query(title, tmp_dir)
                        pdf_path = self._find_downloaded_pdf(tmp_dir)
                    except Exception as e:
                        if self.debug:
                            logger.debug(f"PyPaperBot query path failed: {e}")

            if pdf_path is None:
                if not _PYPAPERBOT_AVAILABLE:
                    raise RuntimeError(
                        "PyPaperBot is not installed. Install it with pip to enable additional PDF retrieval paths."
                    )
                raise RuntimeError(
                    "PyPaperBot could not retrieve a PDF from provided URL"
                )

            docling = DoclingScraper(debug=self.debug)
            output = await docling.arun(OmniScraperInputSchema(url=str(pdf_path)))
            return output
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
                pass
