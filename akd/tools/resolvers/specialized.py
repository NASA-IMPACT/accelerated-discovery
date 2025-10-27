import re

from loguru import logger
from pydantic import AnyUrl, HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema


class PDFUrlResolver(BaseArticleResolver):
    """
    Resolver that prioritizes pdf_url field from search results.
    Returns the pdf_url directly if available, otherwise falls back to url.
    """

    def validate_url(self, url: str | HttpUrl) -> bool:
        """PDFUrlResolver accepts any URL"""
        return True

    async def resolve(self, params: ResolverInputSchema) -> ResolverOutputSchema | None:
        """
        Priority: pdf_url (if available) -> None if not available
        Returns the preferred PDF URL with metadata.
        """
        if params.pdf_url:
            result = ResolverOutputSchema(**params.model_dump())

            result.extra["is_url_resolved"] = True
            result.extra["url_source"] = self.__class__.__name__
            result.extra["url_type"] = "pdf"
            result.url = params.pdf_url
            result.resolvers.append(self.__class__.__name__)
            return result
        return None


class DOIResolver(BaseArticleResolver):
    """
    Resolver that extracts and resolves DOI from search results.

    This resolver:
    1. Extracts DOI from URL if params.doi is missing
    2. Normalizes DOI (removes prefixes like 'doi:', 'https://doi.org/')
    3. Validates DOI format
    4. Constructs canonical DOI URL (https://doi.org/{doi})
    """

    def validate_url(self, url: str | HttpUrl) -> bool:
        """DOIResolver accepts any URL"""
        return True

    @staticmethod
    def extract_doi_from_url(url: str | HttpUrl | AnyUrl | None) -> str | None:
        """
        Extract DOI from a URL if it contains one.

        Handles:
        - DOI resolver URLs: https://doi.org/10.1234/example
        - Publisher URLs with DOI: https://nature.com/articles/10.1038/...
        - Query parameters and fragments are stripped

        Examples:
            >>> DOIResolver.extract_doi_from_url("https://doi.org/10.1234/example")
            '10.1234/example'
            >>> DOIResolver.extract_doi_from_url("https://www.nature.com/articles/10.1038/s41586-021-03819-2")
            '10.1038/s41586-021-03819-2'
            >>> DOIResolver.extract_doi_from_url("https://doi.org/10.1000/xyz?foo=bar")
            '10.1000/xyz'
        """
        if not url:
            return None

        url_str = str(url).lower()

        # Check if URL contains doi.org or dx.doi.org
        if "doi.org/" in url_str:
            # Extract DOI after doi.org/ (stop at query params or fragment)
            match = re.search(r"doi\.org/(10\.[^\s?#]+)", url_str)
            if match:
                return match.group(1)

        # Check for DOI pattern in URL path (e.g., nature.com, science.org)
        # DOI pattern: 10.xxxx/yyyy... (stop at query params, fragment, or whitespace)
        match = re.search(r"(10\.\d{4,}/[^\s?#&]+)", url_str)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def normalize_doi(doi: str | None) -> str | None:
        """
        Normalize DOI by removing common prefixes.

        Examples:
            >>> DOIResolver.normalize_doi("doi:10.1234/example")
            '10.1234/example'
            >>> DOIResolver.normalize_doi("https://doi.org/10.1234/example")
            '10.1234/example'
        """
        if not doi:
            return None

        doi = doi.lower().strip()
        doi = re.sub(r"^doi:\s*", "", doi)
        doi = re.sub(r"^https?://doi\.org/", "", doi)
        doi = re.sub(r"^https?://dx\.doi\.org/", "", doi)
        return doi if doi else None

    @staticmethod
    def validate_doi_format(doi: str) -> bool:
        """
        Validate DOI format using the standard pattern.
        DOI format: 10.{registrant}/{suffix} where suffix contains no whitespace
        """
        doi_pattern = r"^10\.\d+/\S+$"
        return bool(re.match(doi_pattern, doi.strip()))

    async def resolve(
        self,
        params: ResolverInputSchema,
    ) -> ResolverOutputSchema | None:
        """
        Extract and resolve DOI to canonical URL.

        Priority:
        1. Use params.doi if provided
        2. Extract DOI from params.url if doi is missing
        3. Normalize and validate DOI
        4. Construct canonical DOI URL (https://doi.org/{doi})

        Returns None if no valid DOI is found.
        """
        doi = None

        # Step 1: Get DOI from params.doi or extract from URL
        if params.doi:
            doi = self.normalize_doi(params.doi)
        elif params.url:
            extracted = self.extract_doi_from_url(params.url)
            if extracted:
                doi = self.normalize_doi(extracted)
                if self.debug:
                    logger.debug(f"Extracted DOI from URL: {doi}")

        if not doi:
            if self.debug:
                logger.debug("No DOI found in params.doi or params.url")
            return None

        # Step 2: Validate DOI format
        if not self.validate_doi_format(doi):
            if self.debug:
                logger.debug(f"Invalid DOI format: {doi}")
            return None

        # Step 3: Construct canonical DOI URL
        result = ResolverOutputSchema(**params.model_dump())
        result.doi = doi  # Ensure DOI is populated
        result.extra["is_url_resolved"] = True
        result.extra["url_source"] = self.__class__.__name__
        result.extra["url_type"] = "doi_redirect"
        result.url = HttpUrl(f"https://doi.org/{doi}")
        result.resolvers.append(self.__class__.__name__)

        if self.debug:
            logger.debug(f"Resolved DOI: {doi} -> {result.url}")

        return result
