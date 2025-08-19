from abc import abstractmethod
from typing import Optional, Union
from urllib.parse import urljoin

import re

import requests
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import Field, HttpUrl

from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool, BaseToolConfig


class ResolverInputSchema(InputSchema):
    """Input schema for resolver"""

    url: HttpUrl = Field(..., description="Input url to resolve the article from")


class ResolverOutputSchema(OutputSchema):
    """Output schema for resolver"""

    url: HttpUrl = Field(..., description="Resolved output url")
    resolver: Optional[str] = Field(
        None,
        description="Resolver used to resolve the url",
    )


class ArticleResolverConfig(BaseToolConfig):
    """Configuration for the resolver"""

    model_config = {"arbitrary_types_allowed": True}

    session: Optional[requests.Session] = Field(
        None,
        description="Optional session to use for requests",
    )
    user_agent: Optional[str] = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36",
        description="Optional user agent to use for requests",
    )
    debug: bool = Field(
        False,
        description="Optional debug flag to enable debug logging",
    )


class BaseArticleResolver(BaseTool):
    """Base class for all paper resolvers."""

    input_schema = ResolverInputSchema
    output_schema = ResolverOutputSchema
    config_schema = ArticleResolverConfig

    @property
    def headers(self) -> dict:
        return {
            "User-Agent": self.user_agent,
        }

    @abstractmethod
    async def resolve(self, url: str | HttpUrl) -> str:
        """Resolve a URL to its DOI or full-text link."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def validate_url(self, url: HttpUrl | str) -> bool:
        """Check if this resolver can handle the given URL."""
        raise NotImplementedError("Subclasses must implement this method")

    async def _arun(
        self,
        params: ResolverInputSchema,
        **kwargs,
    ) -> ResolverOutputSchema:
        self.validate_url(params.url)
        return ResolverOutputSchema(
            url=await self.resolve(params.url),
            resolver=self.__class__.__name__,
        )


class IdentityResolver(BaseArticleResolver):
    def validate_url(self, url: Union[str, HttpUrl]) -> bool:
        return True

    async def resolve(self, url: Union[str, HttpUrl]) -> str:
        return str(url)


class ArxivResolver(BaseArticleResolver):
    def validate_url(self, url: HttpUrl | str):
        if "arxiv.org" not in str(url):
            raise RuntimeError("Not a valid arxiv url")
        return True

    async def resolve(self, url: HttpUrl | str) -> str:
        url = str(url)
        self.validate_url(url)
        paper_id = url.split("/")[-1]
        return f"https://arxiv.org/pdf/{paper_id}.pdf"


class ADSResolver(BaseArticleResolver):
    """Resolver for NASA ADS (Astrophysics Data System) papers."""

    def validate_url(self, url):
        """Check if this URL is from NASA ADS."""
        if "adsabs.harvard.edu" not in str(url):
            raise RuntimeError("Not a valid ADS URL")
        return True

    async def resolve(self, url: str | HttpUrl) -> str | None:
        """
        Resolve a NASA ADS URL to its DOI or direct PDF link.

        Args:
            url (str): The NASA ADS URL to resolve

        Returns:
            str: The DOI URL or direct PDF URL if found, None otherwise
        """
        url = str(url)
        self.validate_url(url)
        try:
            # Fetch the ADS page
            response = self.session.get(url, headers=self.headers)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to fetch ADS page: HTTP {response.status_code}",
                )

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Look for the DOI in meta tags
            doi_tag = soup.find("meta", {"name": "citation_doi"})
            if doi_tag:
                doi = doi_tag["content"]
                return f"https://doi.org/{doi}"

            # Look for the bibcode in meta tags
            bibcode_tag = soup.find("meta", {"name": "citation_bibcode"})
            if bibcode_tag:
                bibcode = bibcode_tag["content"]
                pdf_url = f"https://articles.adsabs.harvard.edu/pdf/{bibcode}"

                # Verify the PDF URL works
                pdf_response = self.session.head(pdf_url)
                if pdf_response.status_code == 200:
                    return pdf_url

            # If no DOI or direct PDF found, look for other PDF links
            pdf_links = [
                a.get("href")
                for a in soup.find_all("a")
                if a.get("href") and (a.get("href").endswith(".pdf") or "PDF" in a.text)
            ]

            if pdf_links:
                # Return the first PDF link, making it an absolute URL if needed
                return urljoin(url, pdf_links[0])

            return None

        except Exception as e:
            logger.error(f"Error resolving ADS URL: {e}")
            return None


class UnpaywallResolver(BaseArticleResolver):
    """Resolver for finding open access versions via Unpaywall API."""

    def validate_url(self, url: HttpUrl | str) -> bool:
        """Check if this URL contains a DOI that can be resolved via Unpaywall."""
        url_str = str(url)
        # Look for DOI patterns in the URL
        doi_patterns = [
            r'10\.\d{4,}/[^\s"<>#]+',  # Standard DOI format
            r'/doi/(?:full/|pdf/|pdfdirect/)?(10\.[^/?#]+)',  # DOI in path
        ]
        
        for pattern in doi_patterns:
            if re.search(pattern, url_str, re.IGNORECASE):
                return True
        return False

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """Extract DOI from URL using pattern matching."""
        doi_patterns = [
            r'10\.\d{4,}/[^\s"<>#]+',  # Standard DOI format
            r'/doi/(?:full/|pdf/|pdfdirect/)?(10\.[^/?#]+)',  # DOI in path
        ]
        
        for pattern in doi_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                # Return the full DOI or the captured group
                return match.group(1) if match.groups() else match.group(0)
        return None

    async def resolve(self, url: HttpUrl | str) -> str:
        """
        Resolve a DOI URL to its open access version via Unpaywall API.
        
        Args:
            url (str): The URL containing a DOI to resolve
            
        Returns:
            str: The open access PDF URL if found, otherwise the original URL
        """
        url_str = str(url)
        doi = self._extract_doi_from_url(url_str)
        
        if not doi:
            if self.config.debug:
                logger.debug(f"No DOI found in URL: {url_str}")
            return url_str
            
        try:
            # Query Unpaywall API
            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=research@example.com"
            response = self.session.get(unpaywall_url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                if self.config.debug:
                    logger.debug(f"Unpaywall API returned {response.status_code} for DOI: {doi}")
                return url_str
                
            data = response.json()
            
            # Check if paper is open access and has a PDF URL
            if data.get('is_oa', False):
                best_oa_location = data.get('best_oa_location')
                if best_oa_location and best_oa_location.get('url_for_pdf'):
                    pdf_url = best_oa_location['url_for_pdf']
                    if self.config.debug:
                        logger.debug(f"Found open access PDF via Unpaywall: {pdf_url}")
                    return pdf_url
                    
                # Fallback to host URL if no direct PDF
                if best_oa_location and best_oa_location.get('host_type') in ['publisher', 'repository']:
                    oa_url = best_oa_location.get('url')
                    if oa_url:
                        if self.config.debug:
                            logger.debug(f"Found open access version via Unpaywall: {oa_url}")
                        return oa_url
            
            if self.config.debug:
                logger.debug(f"No open access version found for DOI: {doi}")
            return url_str
            
        except Exception as e:
            if self.config.debug:
                logger.debug(f"Error querying Unpaywall for DOI {doi}: {e}")
            return url_str
