import re
from typing import Optional, Union

from loguru import logger
from pydantic import HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema


class PDFUrlResolver(BaseArticleResolver):
    """
    Resolver that prioritizes pdf_url field from search results.
    Returns the pdf_url directly if available, otherwise falls back to url.
    """

    def validate_url(self, url: Union[str, HttpUrl]) -> bool:
        """PDFUrlResolver accepts any URL"""
        return True

    async def resolve(self, params: ResolverInputSchema) -> ResolverOutputSchema | None:
        """
        Priority: pdf_url (if available) -> None if not available
        Returns the preferred PDF URL with metadata.
        """
        if params.pdf_url:
            result = ResolverOutputSchema(**params.model_dump())
            if not result.extra:
                result.extra = {}
            
            if not hasattr(result.extra, "original_url"):
                result.extra["original_url"] = []

            result.extra["original_url"].append(params.url)
            result.url = params.pdf_url
            result.resolvers.append(self.__class__.__name__)
            return result
        return None


class DOIResolver(BaseArticleResolver):
    """
    Resolver that prioritizes DOI field from search results.
    Constructs URL from DOI if available, validates DOI format and accessibility.
    """

    def validate_url(self, url: Union[str, HttpUrl]) -> bool:
        """DOIResolver accepts any URL"""
        return True

    def _validate_doi_format(self, doi: str) -> bool:
        """
        Validate DOI format using the standard pattern.
        DOI format: 10.{registrant}/{suffix} where suffix contains no whitespace
        """
        doi_pattern = r"^10\.\d+/\S+$"
        return bool(re.match(doi_pattern, doi.strip()))

    async def resolve(
        self,
        params: ResolverInputSchema,
    ) -> Optional[ResolverOutputSchema]:
        """
        Priority: doi (construct and validate URL) -> None if invalid
        Validates DOI format and constructs DOI URL if available.
        Returns None if DOI is invalid or not provided.
        """
        if params.doi is not None:
            # Validate DOI format first
            if not self._validate_doi_format(params.doi):
                if self.debug:
                    logger.debug(f"Invalid DOI format: {params.doi}")
                return None

            # Construct and return DOI URL with metadata
            result = ResolverOutputSchema(**params.model_dump())
            if not result.extra:
                result.extra = {}
            if not hasattr(result.extra, "original_url"):
                result.extra["original_url"] = []
            result.extra["original_url"].append(params.url)
            result.url = HttpUrl(f"https://doi.org/{params.doi}")
            result.resolvers.append(self.__class__.__name__)
            return result
        else:
            # No DOI provided - return None instead of falling back
            if self.debug:
                logger.debug("No DOI provided for DOI resolver")
            return None
