import re
from typing import Optional, Union

import httpx
from loguru import logger
from pydantic import HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema


class UnpaywallResolver(BaseArticleResolver):
    """Resolver for finding open access versions via Unpaywall API."""

    def validate_url(self, url: Union[str, HttpUrl]) -> bool:
        """Check if this URL contains a DOI that can be resolved via Unpaywall."""
        url_str = str(url)
        # Look for DOI patterns in the URL
        doi_patterns = [
            r'10\.\d{4,}/[^\s"<>#]+',  # Standard DOI format
            r"/doi/(?:full/|pdf/|pdfdirect/)?(10\.[^/?#]+)",  # DOI in path
        ]

        for pattern in doi_patterns:
            if re.search(pattern, url_str, re.IGNORECASE):
                return True
        return False

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """Extract DOI from URL using pattern matching."""
        doi_patterns = [
            r'10\.\d{4,}/[^\s"<>#]+',  # Standard DOI format
            r"/doi/(?:full/|pdf/|pdfdirect/)?(10\.[^/?#]+)",  # DOI in path
        ]

        for pattern in doi_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                # Return the full DOI or the captured group
                return match.group(1) if match.groups() else match.group(0)
        return None

    async def resolve(
        self,
        params: ResolverInputSchema,
    ) -> Optional[ResolverOutputSchema]:
        """
        Resolve a DOI URL to its open access version via Unpaywall API.

        Args:
            params: ResolverInputSchema containing the URL with DOI to resolve

        Returns:
            ResolverOutputSchema with open access PDF URL if found, None if resolution fails
        """
        url_str = str(params.url)
        doi = self._extract_doi_from_url(url_str)

        if not doi:
            if self.debug:
                logger.debug(f"No DOI found in URL: {url_str}")
            return None

        try:
            # Query Unpaywall API
            unpaywall_url = (
                f"https://api.unpaywall.org/v2/{doi}?email=research@example.com"
            )

            async with httpx.AsyncClient(timeout=self.validation_timeout) as client:
                response = await client.get(unpaywall_url, headers=self.headers)

                if response.status_code != 200:
                    if self.debug:
                        logger.debug(
                            f"Unpaywall API returned {response.status_code} for DOI: {doi}",
                        )
                    return None

                data = response.json()

                # Check if paper is open access and has a PDF URL
                if data.get("is_oa", False):
                    best_oa_location = data.get("best_oa_location")
                    if best_oa_location and best_oa_location.get("url_for_pdf"):
                        pdf_url = best_oa_location["url_for_pdf"]
                        if self.debug:
                            logger.debug(
                                f"Found open access PDF via Unpaywall: {pdf_url}",
                            )

                        # Return ResolverOutputSchema with resolved URL and preserved metadata
                        result = ResolverOutputSchema(**params.model_dump())
                        result.resolved_url = HttpUrl(pdf_url)
                        result.resolvers.append(self.__class__.__name__)
                        return result

                    # Fallback to host URL if no direct PDF
                    if best_oa_location and best_oa_location.get("host_type") in [
                        "publisher",
                        "repository",
                    ]:
                        oa_url = best_oa_location.get("url")
                        if oa_url:
                            if self.debug:
                                logger.debug(
                                    f"Found open access version via Unpaywall: {oa_url}",
                                )

                            # Return ResolverOutputSchema with resolved URL and preserved metadata
                            result = ResolverOutputSchema(**params.model_dump())
                            result.resolved_url = HttpUrl(oa_url)
                            result.resolvers.append(self.__class__.__name__)
                            return result

                if self.debug:
                    logger.debug(f"No open access version found for DOI: {doi}")
                return None

        except Exception as e:
            if self.debug:
                logger.debug(f"Error querying Unpaywall for DOI {doi}: {e}")
            return None
