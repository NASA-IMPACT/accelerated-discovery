from ._base import (
    ScrapedMetadata,
    ScraperToolBase,
    ScraperToolConfig,
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
)
from .composite import (
    CompositeScraper,
    ResearchArticleResolver,
)
from .pypaperbot_scraper import PyPaperBotScraperConfig
from .waterfall import WaterfallScraper, WaterfallScraperConfig
from .omni import DoclingScraper, DoclingScraperConfig, OmniScraperInputSchema
from .pdf_scrapers import PDFScraperInputSchema, SimplePDFScraper
from .pypaperbot_scraper import PyPaperBotScraper
from .web_scrapers import Crawl4AIWebScraper, SimpleWebScraper

__all__ = [
    "SimplePDFScraper",
    "SimpleWebScraper",
    "Crawl4AIWebScraper",
    "PDFScraperInputSchema",
    "ScraperToolInputSchema",
    "ScraperToolOutputSchema",
    "DoclingScraper",
    "DoclingScraperConfig",
    "OmniScraperInputSchema",
    "PyPaperBotScraper",
    "PyPaperBotScraperConfig",
    "CompositeScraper",
    "WaterfallScraper",
    "WaterfallScraperConfig",
    "ResearchArticleResolver",
    "ScrapedMetadata",
    "ScraperToolBase",
    "ScraperToolConfig",
]
