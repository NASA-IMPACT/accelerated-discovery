from typing import Union

from loguru import logger
from pydantic import HttpUrl

from .resolvers import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema
from .web_scrapers import (
    WebpageMetadata,
    WebpageScraperToolInputSchema,
    WebpageScraperToolOutputSchema,
    WebScraperToolBase,
)


class CompositeWebScraper(WebScraperToolBase):
    def __init__(
        self,
        *scrapers: WebScraperToolBase,
        debug: bool = False,
    ) -> None:
        self.debug = bool(debug)
        self.scrapers = scrapers

    async def arun(
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
                        f"Running scraper={scraper.__class__.__name__} "
                        f"for {params}",
                    )

                result = await scraper.arun(params)
                if result:
                    break
            except Exception as e:
                logger.error(
                    f"Error running {scraper.__class__.__name__}\n{str(e)}",
                )
                continue
        return result


class ResearchArticleResolver(BaseArticleResolver):
    def __init__(self, *resolvers: BaseArticleResolver):
        self.resolvers = resolvers

    def resolve(self, url: Union[str, HttpUrl]) -> ResolverOutputSchema:
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
                output = resolver.run(ResolverInputSchema(url=original_url))
                if not output.url:
                    raise ValueError("Resolver failure")
                if output.url:
                    break
            except:
                logger.error(f"Error using resolver={rname}")
        return output

    def run(self, params: ResolverInputSchema) -> ResolverOutputSchema:
        output = self.resolve(params.url)
        if not output.url:
            output.url = params.url
            output.resolver = self.__class__.__name__
        return output
