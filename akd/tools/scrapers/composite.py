from loguru import logger

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
