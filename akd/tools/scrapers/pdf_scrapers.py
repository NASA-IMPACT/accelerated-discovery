from typing import Union

from pydantic import Field, FilePath, HttpUrl

from .web_scrapers import WebpageScraperToolInputSchema


class PDFScraperToolInputSchema(WebpageScraperToolInputSchema):
    """
    Input schema for the PDFScraper
    """

    url: Union[HttpUrl, FilePath] = Field(..., description="Path of the pdf")
