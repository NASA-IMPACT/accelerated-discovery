from ._base import ScraperToolInputSchema, ScraperToolOutputSchema
from .omni import DoclingScraper, DoclingScraperConfig, OmniScraperInputSchema
from .pdf_scrapers import PDFScraperInputSchema, SimplePDFScraper
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
]
