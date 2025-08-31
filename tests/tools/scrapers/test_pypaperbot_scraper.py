"""Integration tests for PyPaperBotScraper."""

import importlib.util
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import AnyUrl

from akd.tools.scrapers._base import ScrapedMetadata, ScraperToolInputSchema
from akd.tools.scrapers.pypaperbot import PyPaperBotScraper, PyPaperBotScraperConfig

# Check if PyPaperBot is available (same check as in the scraper)
_PYPAPERBOT_AVAILABLE: bool = importlib.util.find_spec("PyPaperBot") is not None


class TestPyPaperBotScraper:
    """Test suite for PyPaperBotScraper."""

    def test_doi_extraction_from_arxiv_url(self):
        """Test DOI extraction from arXiv URLs."""
        scraper = PyPaperBotScraper()

        # Test the specific arXiv URL
        arxiv_url = "https://doi.org/10.48550/arXiv.2411.08181"
        doi = scraper._extract_doi_from_url(arxiv_url)

        assert doi == "10.48550/arXiv.2411.08181"

        # Test other DOI formats
        test_cases = [
            ("https://dx.doi.org/10.1038/nature12373", "10.1038/nature12373"),
            ("https://doi.org/10.1126/science.1234567", "10.1126/science.1234567"),
            (
                "https://www.nature.com/articles/10.1038/s41586-023-12345-6",
                "10.1038/s41586-023-12345-6",
            ),
        ]

        for url, expected_doi in test_cases:
            extracted_doi = scraper._extract_doi_from_url(url)
            assert extracted_doi == expected_doi

    def test_query_extraction_from_url(self):
        """Test query extraction from URLs."""
        scraper = PyPaperBotScraper()

        # Test the arXiv URL
        arxiv_url = "https://doi.org/10.48550/arXiv.2411.08181"
        query = scraper._extract_query_from_url(arxiv_url)

        assert "10.48550/arXiv.2411.08181" in query

    def test_scraper_config(self):
        """Test PyPaperBotScraper configuration."""
        config = PyPaperBotScraperConfig(
            max_runtime_seconds=60,
            scholar_pages="2",
            scholar_results=5,
            enable_query_fallback=False,
        )

        scraper = PyPaperBotScraper(config=config)

        assert scraper.config.max_runtime_seconds == 60
        assert scraper.config.scholar_pages == "2"
        assert scraper.config.scholar_results == 5
        assert scraper.config.enable_query_fallback is False

    @pytest.mark.skipif(
        not _PYPAPERBOT_AVAILABLE,
        reason="PyPaperBot not available in environment",
    )
    @pytest.mark.asyncio
    async def test_pypaperbot_scraper_with_arxiv_url_integration(self):
        """Integration test with actual PyPaperBot for the specific arXiv URL.

        This test requires PyPaperBot to be installed and will be skipped in CI/CD
        environments where it's not available.
        """
        # Configure with shorter timeout for testing
        config = PyPaperBotScraperConfig(
            max_runtime_seconds=60,  # Shorter timeout for testing
            enable_query_fallback=True,
        )

        scraper = PyPaperBotScraper(config=config, debug=True)

        # Test input
        input_data = ScraperToolInputSchema(
            url=AnyUrl("https://doi.org/10.48550/arXiv.2411.08181"),
        )

        # Run the scraper
        result = await scraper.arun(input_data)

        # Verify result structure
        assert result is not None
        assert hasattr(result, "content")
        assert hasattr(result, "metadata")

        # If content was successfully scraped, verify it contains the expected title
        if result.content and result.content.strip():
            assert (
                "Challenges in Guardrailing Large Language Models for Science"
                in result.content
            )
            assert result.metadata.url == input_data.url

        # Note: If PyPaperBot fails to download or the content is empty,
        # the test will still pass as long as the scraper handles it gracefully

    @pytest.mark.skipif(
        not _PYPAPERBOT_AVAILABLE,
        reason="PyPaperBot not available in environment",
    )
    @pytest.mark.asyncio
    async def test_pypaperbot_scraper_fallback_behavior(self):
        """Test fallback behavior when DOI extraction might fail."""
        config = PyPaperBotScraperConfig(
            max_runtime_seconds=30,
            enable_query_fallback=True,
        )

        scraper = PyPaperBotScraper(config=config, debug=True)

        # Test with a URL that might not have a clear DOI
        input_data = ScraperToolInputSchema(
            url=AnyUrl("https://example.com/some-paper-about-llm-guardrails"),
        )

        result = await scraper.arun(input_data)

        # Should handle gracefully even if no content is found
        assert result is not None
        assert hasattr(result, "content")
        assert hasattr(result, "metadata")
        assert result.metadata.url == input_data.url

    @pytest.mark.asyncio
    async def test_pypaperbot_scraper_with_mocked_docling(self):
        """Test PyPaperBotScraper with mocked Docling scraper to isolate PyPaperBot behavior."""
        # Create a mock for the internal scraper (Docling)
        mock_docling_result = type(
            "MockResult",
            (),
            {
                "content": "Challenges in Guardrailing Large Language Models for Science\n\nThis is test content from the mocked scraper.",
                "metadata": type(
                    "MockMetadata",
                    (),
                    {
                        "title": "Challenges in Guardrailing Large Language Models for Science",
                        "url": "test_url",
                    },
                )(),
            },
        )()

        mock_scraper = AsyncMock()
        mock_scraper.arun.return_value = mock_docling_result
        mock_scraper.input_schema = ScraperToolInputSchema

        # Test PyPaperBotScraper with mocked Docling
        scraper = PyPaperBotScraper(scraper=mock_scraper, debug=True)

        input_data = ScraperToolInputSchema(
            url=AnyUrl("https://doi.org/10.48550/arXiv.2411.08181"),
        )

        # Mock the PDF finding and PyPaperBot execution
        with (
            patch.object(scraper, "_find_downloaded_pdf") as mock_find_pdf,
            patch.object(scraper, "_run_pypaperbot_with_doi") as mock_run_pypaperbot,
        ):
            # Mock successful PDF download
            from pathlib import Path

            mock_pdf_path = Path("/tmp/test.pdf")
            mock_find_pdf.return_value = mock_pdf_path
            mock_run_pypaperbot.return_value = mock_pdf_path

            result = await scraper.arun(input_data)

            # Verify the result contains expected content
            assert result.content == mock_docling_result.content
            assert (
                "Challenges in Guardrailing Large Language Models for Science"
                in result.content
            )
            assert result.metadata.url == input_data.url

    def test_empty_result_handling(self):
        """Test that scraper handles empty results gracefully."""
        scraper = PyPaperBotScraper(debug=True)

        # This should not raise an exception
        empty_result = scraper.output_schema(
            content="",
            metadata=ScrapedMetadata(
                url=AnyUrl("https://doi.org/10.48550/arXiv.2411.08181"),
                title="",
                query="10.48550/arXiv.2411.08181",
            ),
        )

        assert empty_result.content == ""
        assert empty_result.metadata.url == AnyUrl(
            "https://doi.org/10.48550/arXiv.2411.08181",
        )
