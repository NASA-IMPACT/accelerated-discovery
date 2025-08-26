from ._base import (
    ScrapedMetadata,
    ScraperToolBase,
    ScraperToolConfig,
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
)
from .omni import DoclingScraper, DoclingScraperConfig, OmniScraperInputSchema
from .pdf_scrapers import PDFScraperInputSchema, SimplePDFScraper
from .pypaperbot_scraper import PyPaperBotScraper, PyPaperBotScraperConfig
from .web_scrapers import Crawl4AIWebScraper, SimpleWebScraper

__all__ = [
    "SimplePDFScraper",
    "SimpleWebScraper",
    "Crawl4AIWebScraper",
    "PyPaperBotScraper",
    "PyPaperBotScraperConfig",
    "PDFScraperInputSchema",
    "ScraperToolInputSchema",
    "ScraperToolOutputSchema",
    "DoclingScraper",
    "DoclingScraperConfig",
    "OmniScraperInputSchema",
    "ScrapedMetadata",
    "ScraperToolBase",
    "ScraperToolConfig",
]
