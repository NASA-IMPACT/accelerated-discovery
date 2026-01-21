import importlib
from typing import TYPE_CHECKING

from ._base import (
    ScrapedMetadata,
    ScraperToolBase,
    ScraperToolConfig,
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
)
from .composite import CompositeScraper
from .pdf_scrapers import PDFScraperInputSchema, SimplePDFScraper
from .pypaperbot import PyPaperBotScraper, PyPaperBotScraperConfig
from .web_scrapers import Crawl4AIScraperConfig, Crawl4AIWebScraper, SimpleWebScraper

# Lazy imports for heavy dependencies (docling)
if TYPE_CHECKING:
    from .omni import DoclingScraper, DoclingScraperConfig, OmniScraperInputSchema

_LAZY_IMPORTS = {
    "DoclingScraper": ".omni",
    "DoclingScraperConfig": ".omni",
    "OmniScraperInputSchema": ".omni",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimplePDFScraper",
    "SimpleWebScraper",
    "CompositeScraper",
    "Crawl4AIWebScraper",
    "Crawl4AIScraperConfig",
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
