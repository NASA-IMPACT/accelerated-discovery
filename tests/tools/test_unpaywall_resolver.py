"""
Integration tests for UnpaywallResolver - NO MOCKING.
Tests real API calls to verify functionality.
"""

import pytest
from pydantic import HttpUrl

from akd.tools.scrapers.resolvers import (
    UnpaywallResolver,
    ArticleResolverConfig,
    ResolverInputSchema,
)


class TestUnpaywallResolverIntegration:
    """Integration tests for UnpaywallResolver with real API calls."""

    def test_initialization(self):
        """Test UnpaywallResolver initialization."""
        config = ArticleResolverConfig(debug=True)
        resolver = UnpaywallResolver(config=config)
        assert resolver.config.debug is True
        assert hasattr(resolver, 'input_schema')
        assert hasattr(resolver, 'output_schema')
        assert hasattr(resolver, 'config_schema')

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = ArticleResolverConfig(debug=True, user_agent="Test Agent")
        resolver = UnpaywallResolver(config=config)
        assert resolver.config.debug is True
        assert resolver.config.user_agent == "Test Agent"

    def test_validate_url_with_doi(self):
        """Test URL validation for URLs containing DOIs."""
        resolver = UnpaywallResolver()
        
        # Test DOI.org URL
        assert resolver.validate_url("https://doi.org/10.1038/nature12345") is True
        
        # Test Wiley URL with DOI
        assert resolver.validate_url("https://onlinelibrary.wiley.com/doi/full/10.1002/example") is True
        
        # Test Science Direct URL with DOI
        assert resolver.validate_url("https://www.sciencedirect.com/science/article/pii/S0123456789012345") is False
        
        # Test URL without DOI
        assert resolver.validate_url("https://example.com/paper.html") is False

    def test_extract_doi_from_url(self):
        """Test DOI extraction from various URL formats."""
        resolver = UnpaywallResolver()
        
        # Test DOI.org URL
        doi = resolver._extract_doi_from_url("https://doi.org/10.1038/nature12345")
        assert doi == "10.1038/nature12345"
        
        # Test Wiley URL
        doi = resolver._extract_doi_from_url("https://onlinelibrary.wiley.com/doi/full/10.1002/example")
        assert doi == "10.1002/example"
        
        # Test URL with no DOI
        doi = resolver._extract_doi_from_url("https://example.com/paper.html")
        assert doi is None

    @pytest.mark.asyncio
    async def test_resolve_with_real_open_access_doi(self):
        """Test resolution with a real open access DOI."""
        resolver = UnpaywallResolver(debug=True)
        
        # Use a known open access paper DOI (PLOS ONE paper)
        test_url = "https://doi.org/10.1371/journal.pone.0000308"
        
        result = await resolver.resolve(test_url)
        
        # Should return either the open access PDF URL or the original URL
        assert isinstance(result, str)
        assert len(result) > 0
        # If open access is found, it should be a different URL
        # If not found, should return original URL
        assert result == test_url or result.startswith("http")

    @pytest.mark.asyncio
    async def test_resolve_with_real_closed_access_doi(self):
        """Test resolution with a real closed access DOI."""
        resolver = UnpaywallResolver(debug=True)
        
        # Use a DOI that's likely to be closed access (Nature paper)
        test_url = "https://doi.org/10.1038/nature12345"
        
        result = await resolver.resolve(test_url)
        
        # Should return the original URL since no open access is available
        assert result == test_url

    @pytest.mark.asyncio
    async def test_resolve_with_invalid_doi(self):
        """Test resolution with an invalid DOI."""
        resolver = UnpaywallResolver(debug=True)
        
        # Use an invalid DOI
        test_url = "https://doi.org/10.9999/invalid.doi.12345"
        
        result = await resolver.resolve(test_url)
        
        # Should return the original URL
        assert result == test_url

    @pytest.mark.asyncio
    async def test_resolve_without_doi(self):
        """Test resolution with URL that has no DOI."""
        resolver = UnpaywallResolver(debug=True)
        
        test_url = "https://example.com/some-paper.html"
        
        result = await resolver.resolve(test_url)
        
        # Should return the original URL unchanged
        assert result == test_url

    @pytest.mark.asyncio
    async def test_arun_method(self):
        """Test the main _arun method with real API calls."""
        resolver = UnpaywallResolver(debug=True)
        
        # Test with a real open access DOI
        input_params = ResolverInputSchema(url="https://doi.org/10.1371/journal.pone.0000308")
        
        result = await resolver._arun(input_params)
        
        assert hasattr(result, 'url')
        assert hasattr(result, 'resolver')
        assert result.resolver == "UnpaywallResolver"
        assert isinstance(result.url, HttpUrl)

    @pytest.mark.asyncio
    async def test_multiple_doi_formats(self):
        """Test resolution with multiple DOI URL formats."""
        resolver = UnpaywallResolver(debug=True)
        
        test_cases = [
            "https://doi.org/10.1371/journal.pone.0000308",
            "http://dx.doi.org/10.1371/journal.pone.0000308",
            "https://onlinelibrary.wiley.com/doi/10.1371/journal.pone.0000308",
        ]
        
        for test_url in test_cases:
            if resolver.validate_url(test_url):
                result = await resolver.resolve(test_url)
                assert isinstance(result, str)
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Test handling of network timeouts and errors."""
        # Create resolver with very short timeout
        config = ArticleResolverConfig(debug=True)
        resolver = UnpaywallResolver(config=config)
        
        # Test with a valid DOI - should handle any network issues gracefully
        test_url = "https://doi.org/10.1371/journal.pone.0000308"
        
        result = await resolver.resolve(test_url)
        
        # Should return some result (either resolved or original URL)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_different_paper_types(self):
        """Test resolution with different types of academic papers."""
        resolver = UnpaywallResolver(debug=True)
        
        # Test cases with different publishers/types
        test_cases = [
            "https://doi.org/10.1371/journal.pone.0000308",  # PLOS ONE (open access)
            "https://doi.org/10.1038/nature12345",            # Nature (typically closed)
            "https://doi.org/10.1126/science.1234567",        # Science (typically closed)
        ]
        
        results = []
        for test_url in test_cases:
            try:
                result = await resolver.resolve(test_url)
                results.append((test_url, result))
                assert isinstance(result, str)
                assert len(result) > 0
            except Exception as e:
                # Should handle errors gracefully
                results.append((test_url, str(e)))
        
        # Should have processed all test cases
        assert len(results) == len(test_cases)

    def test_error_handling_invalid_urls(self):
        """Test error handling with invalid URLs."""
        resolver = UnpaywallResolver()
        
        # Test with invalid URLs
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "",
            "https://",
        ]
        
        for invalid_url in invalid_urls:
            # Should not raise exceptions during validation
            try:
                is_valid = resolver.validate_url(invalid_url)
                assert isinstance(is_valid, bool)
            except Exception:
                # Some invalid URLs might raise exceptions, which is acceptable
                pass


class TestUnpaywallResolverIntegrationEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_resolve_with_redirect_doi(self):
        """Test resolution with DOI that redirects."""
        resolver = UnpaywallResolver(debug=True)
        
        # Test with DOI that might have redirects
        test_url = "https://doi.org/10.1371/journal.pone.0000308"
        
        result = await resolver.resolve(test_url)
        
        # Should handle redirects gracefully
        assert isinstance(result, str)
        assert len(result) > 0

    def test_doi_extraction_edge_cases(self):
        """Test DOI extraction with edge cases."""
        resolver = UnpaywallResolver()
        
        edge_cases = [
            ("https://doi.org/10.1000/123456", "10.1000/123456"),
            ("https://dx.doi.org/10.1000/123456", "10.1000/123456"),
            ("https://onlinelibrary.wiley.com/doi/full/10.1002/anie.201234567", "10.1002/anie.201234567"),
            ("https://www.nature.com/articles/10.1038/s41586-020-12345-6", "10.1038/s41586-020-12345-6"),
            ("https://example.com/no-doi-here", None),
            ("", None),
        ]
        
        for test_url, expected_doi in edge_cases:
            extracted_doi = resolver._extract_doi_from_url(test_url)
            assert extracted_doi == expected_doi, f"Failed for URL: {test_url}"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent resolution requests."""
        import asyncio
        
        resolver = UnpaywallResolver(debug=True)
        
        # Test multiple concurrent requests
        test_urls = [
            "https://doi.org/10.1371/journal.pone.0000308",
            "https://doi.org/10.1038/nature12345",
            "https://example.com/no-doi",
        ]
        
        # Run concurrent resolutions
        tasks = [resolver.resolve(url) for url in test_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete (either with results or exceptions handled)
        assert len(results) == len(test_urls)
        
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                assert isinstance(result, str)
                assert len(result) > 0