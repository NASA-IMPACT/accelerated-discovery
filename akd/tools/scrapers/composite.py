from loguru import logger

from ._base import (
    ScrapedMetadata,
    ScraperToolBase,
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
)

# Note: ResearchArticleResolver has been moved to akd.tools.resolvers.composite


class CompositeScraper(ScraperToolBase):
    """
    Composite web scraper that runs multiple scrapers in waterfall manner/sequence.

    The idea is:
    - Start with the first scraper
    - If it produces a result, return it
    - If it fails, try the next scraper in the list

    Usage:
    ```python
    from akd.tools.scrapers._base import PDFScraper
    from akd.tools.scrapers import (
        ScraperToolInputSchema,
        ScraperToolOutputSchema,
    )
    from akd.tools.scrapers import (
        SimpleWebScraper,
        Crawl4AIWebScraper,
    )

    from akd.tools.scrapers import (
        SimplePDFScraper,
        PDFScraperInputSchema,
    )

    from akd.tools.scrapers import DoclingScraper

    from akd.tools.scrapers.composite import CompositeScraper

    _url = "https://arxiv.org/pdf/2402.01822v1"
    # _url = "https://nishparadox.com/writing/vipassana-retreat.html"

    scraper = CompositeScraper(
        DoclingScraper(),
        Crawl4AIWebScraper(),
        SimpleWebScraper(),
        SimplePDFScraper(),
    )

    output = await _scraper.arun(ScraperToolInputSchema(url=_url))
    ```
    """

    def __init__(
        self,
        *scrapers: ScraperToolBase,
        debug: bool = False,
    ) -> None:
        super().__init__(debug=debug)
        self.scrapers = scrapers

    async def _arun(
        self,
        params: ScraperToolInputSchema,
        **kwargs,
    ) -> ScraperToolOutputSchema:
        result = ScraperToolOutputSchema(
            content="",
            metadata=ScrapedMetadata(
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

                result = await scraper.arun(scraper.input_schema(**params.model_dump()))
                if result.content.strip():
                    result.metadata.extra = result.metadata.extra or {}
                    result.metadata.extra["scraper"] = scraper.__class__.__name__
                    result.metadata.extra["scraper_config"] = scraper.config
                    break
            except Exception as e:
                logger.error(
                    f"Error running {scraper.__class__.__name__}\n{str(e)}",
                )
                continue
        return result


# ResearchArticleResolver has been moved to akd.tools.resolvers.composite
