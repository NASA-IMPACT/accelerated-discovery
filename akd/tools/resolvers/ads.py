from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import httpx
from loguru import logger
from pydantic import HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema


class ADSResolver(BaseArticleResolver):
    """Resolver for NASA ADS (Astrophysics Data System) papers."""

    def validate_url(self, url):
        """Check if this URL is from NASA ADS."""
        if "adsabs.harvard.edu" not in str(url):
            raise RuntimeError("Not a valid ADS URL")
        return True

    async def resolve(
        self,
        params: ResolverInputSchema,
    ) -> Optional[ResolverOutputSchema]:
        """
        Resolve a NASA ADS URL to its DOI or direct PDF link.

        Args:
            params: ResolverInputSchema containing the NASA ADS URL to resolve

        Returns:
            Optional[ResolverOutputSchema]: The resolved result with DOI URL or direct PDF URL if found, None otherwise
        """

        # check if other resolver has already resolved the URL
        url = str(params.url)
        try:
            self.validate_url(url)
        except RuntimeError:
            return None
        try:
            async with self.session or httpx.AsyncClient(timeout=self.validation_timeout, headers=self.headers) as client:
                # Fetch the ADS page
                response = await client.get(url)
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
                    result = ResolverOutputSchema(**params.model_dump())
                    result.doi = doi
                    result.resolvers.append(self.__class__.__name__)
                    result.extra["is_url_resolved"] = True
                    result.url = HttpUrl(f"https://doi.org/{doi}")
                    return result

                # Look for the bibcode in meta tags
                bibcode_tag = soup.find("meta", {"name": "citation_bibcode"})
                if bibcode_tag:
                    bibcode = bibcode_tag["content"]
                    pdf_url = f"https://articles.adsabs.harvard.edu/pdf/{bibcode}"

                    # Verify the PDF URL works
                    pdf_response = await client.head(pdf_url)
                    if pdf_response.status_code == 200:
                        result = ResolverOutputSchema(**params.model_dump())
                        
                        result.resolvers.append(self.__class__.__name__)
                        if not result.extra:
                            result.extra = {}
                        result.extra["is_url_resolved"] = True
                        result.url = HttpUrl(pdf_url)
                        return result

                # If no DOI or direct PDF found, look for other PDF links
                pdf_links = [
                    a.get("href")
                    for a in soup.find_all("a")
                    if a.get("href") and (a.get("href").endswith(".pdf") or "PDF" in a.text)
                ]

                if pdf_links:
                    # Return the first PDF link, making it an absolute URL if needed
                    result = ResolverOutputSchema(**params.model_dump())
                    if not result.extra:
                        result.extra = {}
                    result.extra["is_url_resolved"] = True
                    result.url = HttpUrl(urljoin(url, pdf_links[0]))
                    result.resolvers.append(self.__class__.__name__)
                    return result

                # If no PDF found, return None
                if self.debug:
                    logger.debug(f"No resolvable PDF or DOI found for ADS URL: {url}")
                return None

        except Exception as e:
            logger.error(f"Error resolving ADS URL: {e}")
            return None
