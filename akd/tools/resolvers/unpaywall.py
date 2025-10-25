import re
from typing import Optional, Union

import httpx
from loguru import logger
from pydantic import Field, HttpUrl

from ._base import (
    ArticleResolverConfig,
    BaseArticleResolver,
    ResolverInputSchema,
    ResolverOutputSchema,
)


class UnpaywallResolverConfig(ArticleResolverConfig):
    """Configuration for UnpaywallResolver."""

    email: str = Field(
        default="user@institution.edu",
        description="Email address required for Unpaywall API access",
    )


class UnpaywallResolver(BaseArticleResolver):
    """Resolver for finding open access versions via Unpaywall API."""

    config_schema = UnpaywallResolverConfig

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

    def _populate_metadata(
        self,
        result: ResolverOutputSchema,
        unpaywall_data: dict,
    ) -> None:
        """
        Populate metadata fields in the result from Unpaywall API response.

        Args:
            result: ResolverOutputSchema to populate with metadata
            unpaywall_data: Raw JSON response from Unpaywall API

        Unpaywall API provides rich metadata including:
        - title: Article title
        - published_date: Publication date (YYYY-MM-DD format)
        - journal_name: Journal/venue name
        - publisher: Publisher name
        - z_authors: List of author objects with given/family names
        - year: Publication year
        - genre: Publication type (journal-article, book-chapter, etc.)
        - is_oa: Open access status
        - oa_status: Type of OA (gold, green, hybrid, bronze, closed)
        """
        # Populate title if not already present
        if not result.title and unpaywall_data.get("title"):
            result.title = unpaywall_data["title"]

        # Populate published_date if not already present
        if not result.published_date and unpaywall_data.get("published_date"):
            result.published_date = unpaywall_data["published_date"]

        # Populate authors if not already present
        # Unpaywall uses 'z_authors' field with 'raw_author_name' for author information
        if not result.authors:
            authors = []
            z_authors = unpaywall_data.get("z_authors", [])

            for author in z_authors:
                # Use raw_author_name if available, otherwise combine given and family names
                raw_name = author.get("raw_author_name")
                if raw_name:
                    authors.append(raw_name)
                else:
                    # Fallback to combining given and family names
                    given = author.get("given", "")
                    family = author.get("family", "")
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                    elif given:
                        authors.append(given)

            if authors:
                result.authors = authors

        # Store all Unpaywall metadata in extra field
        # This allows access to any field from the Unpaywall API response
        # Exclude fields we've already populated at top level (title, published_date, doi)
        excluded_fields = {"title", "published_date", "doi", "z_authors"}

        for key, value in unpaywall_data.items():
            if key not in excluded_fields and value is not None:
                result.extra[key] = value

        if self.debug:
            logger.debug(
                f"Populated metadata from Unpaywall: title={result.title}, "
                f"published_date={result.published_date}, authors={len(result.authors or [])}",
            )

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
        # Try to get DOI from params first, then extract from URL
        doi = params.doi or self._extract_doi_from_url(url_str)
        if self.debug:
            logger.debug(f"Extracted DOI: {doi} from URL: {url_str}")

        if not doi:
            if self.debug:
                logger.debug(f"No DOI found in URL or params: {url_str}")
            return None

        try:
            # Query Unpaywall API
            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}"

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
                        result.doi = doi
                        result.extra["is_url_resolved"] = True
                        result.url = HttpUrl(pdf_url)
                        result.pdf_url = HttpUrl(pdf_url)  # Set pdf_url field
                        result.resolvers.append(self.__class__.__name__)

                        # Populate metadata fields from Unpaywall API response
                        self._populate_metadata(result, data)

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
                            result.doi = doi
                            result.extra["is_url_resolved"] = True
                            result.url = HttpUrl(oa_url)
                            # Try to set pdf_url if available in best_oa_location
                            if best_oa_location.get("url_for_pdf"):
                                result.pdf_url = HttpUrl(best_oa_location["url_for_pdf"])
                            result.resolvers.append(self.__class__.__name__)

                            # Populate metadata fields from Unpaywall API response
                            self._populate_metadata(result, data)

                            return result

                if self.debug:
                    logger.debug(f"No open access version found for DOI: {doi}")
                return None

        except Exception as e:
            if self.debug:
                logger.debug(f"Error querying Unpaywall for DOI {doi}: {e}")
            return None
