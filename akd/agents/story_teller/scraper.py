from akd.tools.scrapers.composite import CompositeScraper
from akd.tools.scrapers.composite import CompositeScraper
from akd.tools.scrapers.omni import DoclingScraper
from akd.tools.scrapers.pdf_scrapers import SimplePDFScraper
from akd.tools.scrapers.web_scrapers import Crawl4AIWebScraper, SimpleWebScraper


def get_default_scraper(debug=False):
  return CompositeScraper(
      Crawl4AIWebScraper(debug=debug),
      SimpleWebScraper(debug=debug),
      SimplePDFScraper(debug=debug),
      DoclingScraper(debug=debug),
)
