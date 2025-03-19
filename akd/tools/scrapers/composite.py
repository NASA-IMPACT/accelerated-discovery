from typing import Union

from loguru import logger
from pydantic import HttpUrl

from .resolvers import BaseArticleResolver
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

    def run(
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

                result = scraper.run(params)
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

    def resolve(self, url: Union[str, HttpUrl]) -> str:
        url = str(url)
        original_url = str(url)
        for resolver in self.resolvers:
            rname = resolver.__class__.__name__
            try:
                logger.debug(f"Using resolver={rname} for url={url}")
                url = resolver.run(url)
                break
            except:
                logger.error(f"Error using resolver={rname}")
        return url
