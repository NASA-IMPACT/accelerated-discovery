from abc import abstractmethod
from typing import List, Optional

import httpx
import requests
from loguru import logger
from pydantic import Field, HttpUrl

from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.tools import BaseTool, BaseToolConfig


class ResolverInputSchema(SearchResultItem, InputSchema):
    """
    Enhanced input schema for resolver with search result data.
    TThis is directly inherited from SearchResultItem
    to ensure compatibility with existing search result structures.
    It includes fields like url, title, query, doi, pdf_url, and authors.

    SearchResultItem is used to ensure that the resolver can handle
    the same structure as search results, allowing for seamless integration.

    The implementations can resolve:
    - URLs (full text url)
    - DOIs (doi if not present)
    """

    # Override to make title optional for resolver input
    title: str | None = Field(
        None,
        description="Title of the article if available from search result",
    )
    # Override to make query optional for resolver input
    query: str | None = Field(
        None,
        description="Query used to obtain the search result",
    )
        # allow extra fields in input schema
    class Config:
        extra = "allow"


class ResolverOutputSchema(SearchResultItem, OutputSchema):
    """Output schema for resolver"""

    # Override to make title optional for resolver output
    title: str | None = Field(
        None,
        description="Title of the resolved article if available",
    )
    # Override to make query optional for resolver output
    query: str | None = Field(
        None,
        description="Original query used to obtain the search result",
    )

    resolvers: List[str] = Field(
        default_factory=list,
        description="Resolvers used to resolve the search result",
    )

    resolved_url: HttpUrl | None = Field(
        None,
        description="Resolved URL of the article, if available",
    )


class ArticleResolverConfig(BaseToolConfig):
    """Configuration for the resolver"""

    model_config = {"arbitrary_types_allowed": True}

    session: Optional[httpx.AsyncClient] = Field(
        None,
        description="Optional session to use for requests",
    )
    user_agent: Optional[str] = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36",
        description="Optional user agent to use for requests",
    )
    validate_resolved_url: bool = Field(
        True,
        description="Whether to validate that the resolved URL is accessible",
    )
    validation_timeout: float = Field(
        10.0,
        description="Timeout in seconds for URL validation requests",
    )


class BaseArticleResolver(BaseTool[ResolverInputSchema, ResolverOutputSchema]):
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
    async def resolve(self, params: ResolverInputSchema) -> ResolverOutputSchema | None:
        """
        Resolve URL using all available information from input parameters.
        Returns ResolverOutputSchema with resolved data, or None if resolution fails.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def validate_url(self, url: HttpUrl) -> bool:
        """Check if this resolver can handle the given URL."""
        raise NotImplementedError("Subclasses must implement this method")

    async def _post_validate_url(self, url: HttpUrl) -> bool:
        """
        Validate that the resolved URL is accessible and returns a successful status code.

        Args:
            url: The resolved URL to validate

        Returns:
            bool: True if URL is accessible (2xx status), False otherwise

        Raises:
            ValueError: If URL returns non-2xx status or is inaccessible
        """
        if not self.validate_resolved_url:
            return True

        url_str = str(url)
        try:
            async with httpx.AsyncClient(timeout=self.validation_timeout) as client:
                response = await client.head(
                    url_str,
                    headers={"User-Agent": self.user_agent},
                    follow_redirects=True,
                )

                if not (200 <= response.status_code < 300):
                    raise ValueError(
                        f"URL validation failed: {url_str} returned status {response.status_code}",
                    )

                if self.debug:
                    logger.debug(
                        f"URL validation successful: {url_str} (status: {response.status_code})",
                    )

                return True

        except httpx.TimeoutException:
            raise ValueError(f"URL validation timeout: {url_str}")
        except httpx.RequestError as e:
            raise ValueError(f"URL validation failed: {url_str} - {str(e)}")
        except Exception as e:
            raise ValueError(f"URL validation error: {url_str} - {str(e)}")

    async def _arun(
        self,
        params: ResolverInputSchema,
        **kwargs,
    ) -> ResolverOutputSchema:
        self.validate_url(params.url)
        resolved_result = await self.resolve(params)

        # If resolution failed, raise an error
        if resolved_result is None:
            raise ValueError(f"Failed to resolve with {self.__class__.__name__}")

        # Post-validate the resolved URL if available
        if resolved_result.resolved_url:
            await self._post_validate_url(resolved_result.resolved_url)

        if self.__class__.__name__ not in resolved_result.resolvers:
            resolved_result.resolvers.append(self.__class__.__name__)

        return resolved_result
