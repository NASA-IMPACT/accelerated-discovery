from loguru import logger
from pydantic import HttpUrl

from .resolvers import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema
from .web_scrapers import (
    WebpageMetadata,
    WebpageScraperToolInputSchema,
    WebpageScraperToolOutputSchema,
    WebScraperToolBase,
)


class CompositeScraper(WebScraperToolBase):
    """Composite web scraper that runs multiple scrapers in sequence."""

    def __init__(
        self,
        *scrapers: WebScraperToolBase,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.scrapers = scrapers

    async def _arun(
        self,
        params: WebpageScraperToolInputSchema,
    ) -> WebpageScraperToolOutputSchema:
        result = WebpageScraperToolOutputSchema(
            content="",
            metadata=WebpageMetadata(
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

                result = await scraper.arun(params)
                if result.content.strip():
                    break
            except Exception as e:
                logger.error(
                    f"Error running {scraper.__class__.__name__}\n{str(e)}",
                )
                continue
        return result


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
