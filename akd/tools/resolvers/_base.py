from abc import abstractmethod
from typing import Optional

import httpx
import requests
from loguru import logger
from pydantic import Field, HttpUrl

from akd._base import InputSchema, OutputSchema
from akd.tools import BaseTool, BaseToolConfig


class ResolverInputSchema(InputSchema):
    """Enhanced input schema for resolver with multiple URL sources"""

    url: HttpUrl = Field(..., description="Primary URL to resolve")
    pdf_url: Optional[HttpUrl] = Field(
        None,
        description="Direct PDF URL if available from search result",
    )
    doi: Optional[str] = Field(
        None,
        description="DOI identifier if available from search result",
    )


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
    async def resolve(self, params: ResolverInputSchema) -> HttpUrl | None:
        """
        Resolve URL using all available information from input parameters.
        Returns None if resolution fails or is not possible.
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
        resolved_url = await self.resolve(params)

        # If resolution failed, raise an error
        if resolved_url is None:
            raise ValueError(f"Failed to resolve URL with {self.__class__.__name__}")

        # Post-validate the resolved URL
        await self._post_validate_url(resolved_url)

        return ResolverOutputSchema(
            url=resolved_url,
            resolver=self.__class__.__name__,
        )
