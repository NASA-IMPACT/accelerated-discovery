"""
Comprehensive tests for URL resolvers in the AKD project.

Tests cover individual resolvers (IdentityResolver, DOIResolver, PDFUrlResolver)
and the composite ResearchArticleResolver with various scenarios including
mocked HTTP requests and error handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from akd.tools.resolvers import (
    ArticleResolverConfig,
    ArxivResolver,
    DOIResolver,
    IdentityResolver,
    PDFUrlResolver,
    ResearchArticleResolver,
    ResolverInputSchema,
    ResolverOutputSchema,
)


class TestResolverInputSchema:
    """Test ResolverInputSchema validation and behavior."""

    def test_valid_input_schema(self):
        """Test creating valid input schema."""
        schema = ResolverInputSchema(
            url="https://example.com",
            title="Test Paper",
            query="test query",
            pdf_url="https://example.com/paper.pdf",
            doi="10.1000/182",
            authors=["John Doe", "Jane Smith"],
        )
        assert str(schema.url) == "https://example.com/"  # HttpUrl normalizes
        assert schema.title == "Test Paper"
        assert schema.query == "test query"
        assert str(schema.pdf_url) == "https://example.com/paper.pdf"
        assert schema.doi == "10.1000/182"
        assert schema.authors == ["John Doe", "Jane Smith"]

    def test_minimal_input_schema(self):
        """Test input schema with only required url field."""
        schema = ResolverInputSchema(url="https://example.com")
        assert str(schema.url) == "https://example.com/"  # HttpUrl normalizes
        assert schema.title is None
        assert schema.query is None
        assert schema.pdf_url is None
        assert schema.doi is None
        assert schema.authors is None

    def test_input_schema_with_authors(self):
        """Test input schema with authors field for CrossRefDOIResolver support."""
        schema = ResolverInputSchema(
            url="https://example.com",
            title="Deep Learning for Scientific Discovery",
            authors=["John Doe", "Jane Smith", "Bob Johnson"],
        )
        assert str(schema.url) == "https://example.com/"
        assert schema.title == "Deep Learning for Scientific Discovery"
        assert schema.authors == ["John Doe", "Jane Smith", "Bob Johnson"]

    def test_invalid_url_format(self):
        """Test that invalid URLs are rejected by pydantic."""
        with pytest.raises(ValueError):
            ResolverInputSchema(url="not-a-valid-url")


class TestIdentityResolver:
    """Test IdentityResolver functionality."""

    @pytest.fixture
    def resolver(self):
        """Create IdentityResolver instance."""
        return IdentityResolver(debug=True)

    @pytest.fixture
    def basic_input(self):
        """Basic input schema for testing."""
        return ResolverInputSchema(url="https://example.com")

    def test_validate_url_accepts_any_url(self, resolver):
        """Test that IdentityResolver accepts any URL."""
        assert resolver.validate_url("https://example.com") is True
        assert resolver.validate_url("http://test.org") is True
        assert resolver.validate_url("https://arxiv.org/abs/2411.08181") is True

    @pytest.mark.asyncio
    async def test_resolve_returns_same_url(self, resolver, basic_input):
        """Test that resolve returns the same URL."""
        result = await resolver.resolve(basic_input)
        assert isinstance(result, ResolverOutputSchema)
        assert str(result.url) == str(basic_input.url)
        assert "IdentityResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_arun_with_mocked_validation(self, resolver, basic_input):
        """Test full arun execution with mocked HTTP validation."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            result = await resolver.arun(basic_input)

            assert isinstance(result, ResolverOutputSchema)
            assert str(result.url) == str(basic_input.url)
            assert "IdentityResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_different_url_formats(self, resolver):
        """Test IdentityResolver with various URL formats."""
        test_cases = [
            ("https://example.com", "https://example.com/"),  # HttpUrl normalizes
            ("http://test.org", "http://test.org/"),
            ("https://arxiv.org/abs/2411.08181", "https://arxiv.org/abs/2411.08181"),
            ("https://doi.org/10.1000/182", "https://doi.org/10.1000/182"),
            (
                "https://example.com/path/to/paper.pdf",
                "https://example.com/path/to/paper.pdf",
            ),
        ]

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            for input_url, expected_url in test_cases:
                input_schema = ResolverInputSchema(url=input_url)
                result = await resolver.arun(input_schema)
                assert str(result.url) == expected_url
                assert "IdentityResolver" in result.resolvers


class TestDOIResolver:
    """Test DOIResolver functionality."""

    @pytest.fixture
    def resolver(self):
        """Create DOIResolver instance."""
        return DOIResolver(debug=True)

    def test_validate_url_accepts_any_url(self, resolver):
        """Test that DOIResolver accepts any URL."""
        assert resolver.validate_url("https://example.com") is True
        assert resolver.validate_url("http://test.org") is True

    def test_validate_doi_format_valid(self, resolver):
        """Test DOI format validation with valid DOIs."""
        valid_dois = [
            "10.1000/182",
            "10.1038/nature12373",
            "10.48550/arXiv.2411.08181",
            "10.1016/j.cell.2023.01.001",
            "10.1109/ICCV.2019.00123",
        ]

        for doi in valid_dois:
            assert resolver._validate_doi_format(doi) is True

    def test_validate_doi_format_invalid(self, resolver):
        """Test DOI format validation with invalid DOIs."""
        invalid_dois = [
            "not-a-doi",
            "10./invalid",
            "10.1000/",
            "10.1000/ space",
            "10.abc/test",
            "",
            "doi:10.1000/182",  # Missing prefix handling
        ]

        for doi in invalid_dois:
            assert resolver._validate_doi_format(doi) is False

    @pytest.mark.asyncio
    async def test_resolve_with_valid_doi(self, resolver):
        """Test resolve method with valid DOI."""
        input_schema = ResolverInputSchema(
            url="https://example.com",
            doi="10.1000/182",
        )

        result = await resolver.resolve(input_schema)
        assert isinstance(result, ResolverOutputSchema)
        assert str(result.url) == "https://doi.org/10.1000/182"
        assert result.doi == "10.1000/182"
        assert "DOIResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_resolve_with_invalid_doi(self, resolver):
        """Test resolve method with invalid DOI format."""
        input_schema = ResolverInputSchema(
            url="https://example.com",
            doi="invalid-doi",
        )

        result = await resolver.resolve(input_schema)
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_without_doi(self, resolver):
        """Test resolve method when no DOI is provided."""
        input_schema = ResolverInputSchema(url="https://example.com")

        result = await resolver.resolve(input_schema)
        assert result is None

    @pytest.mark.asyncio
    async def test_arun_with_valid_doi(self, resolver):
        """Test full arun execution with valid DOI."""
        input_schema = ResolverInputSchema(
            url="https://example.com",
            doi="10.1000/182",
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            result = await resolver.arun(input_schema)

            assert isinstance(result, ResolverOutputSchema)
            assert str(result.url) == "https://doi.org/10.1000/182"
            assert "DOIResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_arun_fails_with_invalid_doi(self, resolver):
        """Test that arun fails when DOI is invalid."""
        input_schema = ResolverInputSchema(
            url="https://example.com",
            doi="invalid-doi",
        )

        with pytest.raises(ValueError, match="Failed to resolve with DOIResolver"):
            await resolver.arun(input_schema)

    @pytest.mark.asyncio
    async def test_arun_fails_without_doi(self, resolver):
        """Test that arun fails when no DOI is provided."""
        input_schema = ResolverInputSchema(url="https://example.com")

        with pytest.raises(ValueError, match="Failed to resolve with DOIResolver"):
            await resolver.arun(input_schema)


class TestArxivResolver:
    """Test ArxivResolver functionality for arXiv URL transformation."""

    @pytest.fixture
    def resolver(self):
        """Create ArxivResolver instance."""
        return ArxivResolver(debug=True)

    def test_validate_url_accepts_arxiv_urls(self, resolver):
        """Test that ArxivResolver accepts arXiv URLs."""
        assert resolver.validate_url("https://arxiv.org/abs/2411.08181") is True
        assert resolver.validate_url("http://arxiv.org/abs/1234.5678") is True

    def test_validate_url_rejects_non_arxiv_urls(self, resolver):
        """Test that ArxivResolver rejects non-arXiv URLs."""
        with pytest.raises(RuntimeError, match="Not a valid arxiv url"):
            resolver.validate_url("https://example.com")

    @pytest.mark.asyncio
    async def test_resolve_arxiv_abs_to_pdf(self, resolver):
        """Test ArXiv abs URL resolution to PDF URL."""
        input_schema = ResolverInputSchema(url="https://arxiv.org/abs/2411.08181")
        result = await resolver.resolve(input_schema)
        assert isinstance(result, ResolverOutputSchema)
        print(result)
        assert str(result.url) == "https://arxiv.org/pdf/2411.08181.pdf"
        assert "ArxivResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_resolve_different_arxiv_papers(self, resolver):
        """Test resolution of different arXiv paper IDs."""
        test_cases = [
            (
                "https://arxiv.org/abs/2411.08181",
                "https://arxiv.org/pdf/2411.08181.pdf",
            ),
            ("https://arxiv.org/abs/1234.5678", "https://arxiv.org/pdf/1234.5678.pdf"),
            ("http://arxiv.org/abs/2301.00001", "https://arxiv.org/pdf/2301.00001.pdf"),
        ]

        for input_url, expected_pdf in test_cases:
            input_schema = ResolverInputSchema(url=input_url)
            result = await resolver.resolve(input_schema)
            assert isinstance(result, ResolverOutputSchema)
            assert str(result.url) == expected_pdf
            assert "ArxivResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_resolve_non_arxiv_returns_none(self, resolver):
        """Test that non-arXiv URLs return None."""
        input_schema = ResolverInputSchema(url="https://example.com")
        result = await resolver.resolve(input_schema)
        assert result is None

    @pytest.mark.asyncio
    async def test_arun_with_arxiv_url(self, resolver):
        """Test full arun execution with arXiv URL."""
        input_schema = ResolverInputSchema(url="https://arxiv.org/abs/2411.08181")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            result = await resolver.arun(input_schema)

            assert isinstance(result, ResolverOutputSchema)
            assert str(result.url) == "https://arxiv.org/pdf/2411.08181.pdf"
            assert "ArxivResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_arun_fails_with_non_arxiv(self, resolver):
        """Test that arun fails with non-arXiv URL."""
        input_schema = ResolverInputSchema(url="https://example.com")

        with pytest.raises(RuntimeError, match="Not a valid arxiv url"):
            await resolver.arun(input_schema)


class TestPDFUrlResolver:
    """Test PDFUrlResolver functionality."""

    @pytest.fixture
    def resolver(self):
        """Create PDFUrlResolver instance."""
        return PDFUrlResolver(debug=True)

    @pytest.fixture
    def input_with_pdf(self):
        """Input schema with PDF URL (realistic arXiv example)."""
        return ResolverInputSchema(
            url="https://arxiv.org/abs/2411.08181",
            pdf_url="https://arxiv.org/pdf/2411.08181.pdf",
        )

    @pytest.fixture
    def input_without_pdf(self):
        """Input schema without PDF URL."""
        return ResolverInputSchema(url="https://arxiv.org/abs/2411.08181")

    def test_validate_url_accepts_any_url(self, resolver):
        """Test that PDFUrlResolver accepts any URL."""
        assert resolver.validate_url("https://example.com") is True
        assert resolver.validate_url("http://test.org") is True

    @pytest.mark.asyncio
    async def test_resolve_prioritizes_pdf_url(self, resolver, input_with_pdf):
        """Test that resolve prioritizes pdf_url over primary url."""
        result = await resolver.resolve(input_with_pdf)
        assert isinstance(result, ResolverOutputSchema)
        assert str(result.url) == str(input_with_pdf.pdf_url)
        assert str(result.url) == "https://arxiv.org/pdf/2411.08181.pdf"
        assert "PDFUrlResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_resolve_returns_none_without_pdf(self, resolver, input_without_pdf):
        """Test that resolve returns None when no PDF URL is provided."""
        result = await resolver.resolve(input_without_pdf)
        assert result is None

    @pytest.mark.asyncio
    async def test_arun_with_pdf_url(self, resolver, input_with_pdf):
        """Test full arun execution with PDF URL."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            result = await resolver.arun(input_with_pdf)

            assert isinstance(result, ResolverOutputSchema)
            assert str(result.url) == "https://arxiv.org/pdf/2411.08181.pdf"
            assert "PDFUrlResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_arun_fails_without_pdf(self, resolver, input_without_pdf):
        """Test that arun fails when no PDF URL is provided."""
        with pytest.raises(
            ValueError,
            match="Failed to resolve with PDFUrlResolver",
        ):
            await resolver.arun(input_without_pdf)


class TestResearchArticleResolver:
    """Test ResearchArticleResolver composite functionality."""

    @pytest.fixture
    def arxiv_resolver(self):
        """Create ArxivResolver instance."""
        return ArxivResolver(debug=True)

    @pytest.fixture
    def pdf_resolver(self):
        """Create PDFUrlResolver instance."""
        return PDFUrlResolver(debug=True)

    @pytest.fixture
    def doi_resolver(self):
        """Create DOIResolver instance."""
        return DOIResolver(debug=True)

    @pytest.fixture
    def identity_resolver(self):
        """Create IdentityResolver instance."""
        return IdentityResolver(debug=True)

    @pytest.fixture
    def composite_resolver(
        self,
        arxiv_resolver,
        pdf_resolver,
        doi_resolver,
        identity_resolver,
    ):
        """Create composite resolver with all sub-resolvers (realistic order)."""
        return ResearchArticleResolver(
            arxiv_resolver,
            pdf_resolver,
            doi_resolver,
            identity_resolver,
            debug=True,
        )

    @pytest.fixture
    def original_composite_resolver(
        self,
        pdf_resolver,
        doi_resolver,
        identity_resolver,
    ):
        """Create composite resolver matching original user request."""
        return ResearchArticleResolver(
            pdf_resolver,
            doi_resolver,
            identity_resolver,
            debug=True,
        )

    def test_validate_url_accepts_any_url(self, composite_resolver):
        """Test that composite resolver accepts any URL."""
        assert composite_resolver.validate_url("https://example.com") is True
        assert composite_resolver.validate_url("http://test.org") is True

    @pytest.mark.asyncio
    async def test_arxiv_resolver_wins_for_arxiv_urls(self, composite_resolver):
        """Test that ArxivResolver wins for arXiv URLs."""
        input_schema = ResolverInputSchema(
            url="https://arxiv.org/abs/2411.08181",
            pdf_url="https://arxiv.org/pdf/2411.08181.pdf",
            doi="10.48550/arXiv.2411.08181",
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            result = await composite_resolver.arun(input_schema)

            # ArxivResolver should win since it transforms the URL
            assert str(result.url) == "https://arxiv.org/pdf/2411.08181.pdf"
            assert "ArxivResolver" in result.resolvers


    @pytest.mark.asyncio
    async def test_pdf_resolver_wins_when_arxiv_unavailable(
        self,
        original_composite_resolver,
    ):
        """Test that PDF resolver wins when ArxivResolver is not available."""
        input_schema = ResolverInputSchema(
            url="https://arxiv.org/abs/2411.08181",
            pdf_url="https://arxiv.org/pdf/2411.08181.pdf",
            doi="10.48550/arXiv.2411.08181",
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            result = await original_composite_resolver.arun(input_schema)

            # PDF resolver should win since ArxivResolver is not included
            assert str(result.url) == "https://arxiv.org/pdf/2411.08181.pdf"
            assert "PDFUrlResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_resolver_fallback_to_doi(self, composite_resolver):
        """Test fallback to DOI resolver when PDF is not available."""
        input_schema = ResolverInputSchema(
            url="https://example.com",
            doi="10.1000/182",
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            result = await composite_resolver.arun(input_schema)

            # DOI resolver should win since it transforms the URL
            assert str(result.url) == "https://doi.org/10.1000/182"
            assert "DOIResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_resolver_fallback_to_identity(self, composite_resolver):
        """Test fallback to identity resolver when others fail."""
        input_schema = ResolverInputSchema(url="https://example.com")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            result = await composite_resolver.arun(input_schema)

            # Identity resolver should be used as fallback
            assert (
                str(result.url) == "https://example.com/"
            )  # HttpUrl normalizes
            assert "IdentityResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_error_handling_continues_chain(
        self,
        pdf_resolver,
        doi_resolver,
        identity_resolver,
    ):
        """Test that errors in one resolver don't stop the chain."""
        # Create a failing PDF resolver
        failing_pdf_resolver = PDFUrlResolver(debug=True)
        with patch.object(
            failing_pdf_resolver,
            "arun",
            side_effect=Exception("PDF resolver failed"),
        ):
            composite_resolver = ResearchArticleResolver(
                failing_pdf_resolver,
                doi_resolver,
                identity_resolver,
                debug=True,
            )

            input_schema = ResolverInputSchema(
                url="https://example.com",
                doi="10.1000/182",
            )

            with patch("httpx.AsyncClient") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                    return_value=mock_response,
                )

                result = await composite_resolver.arun(input_schema)

                # DOI resolver should work despite PDF resolver failure
                assert str(result.url) == "https://doi.org/10.1000/182"
                assert "DOIResolver" in result.resolvers

    @pytest.mark.asyncio
    async def test_url_transformation_detection(self, composite_resolver):
        """Test detection of URL transformation vs. same URL return."""
        # Test with DOI that transforms URL
        input_schema = ResolverInputSchema(
            url="https://example.com",
            doi="10.1000/182",
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            result = await composite_resolver.arun(input_schema)

            # Should detect that URL was transformed and use DOI resolver
            assert str(result.url) != str(input_schema.url)
            assert str(result.url) == "https://doi.org/10.1000/182"
            assert "DOIResolver" in result.resolvers


class TestHTTPValidation:
    """Test HTTP validation functionality across resolvers."""

    @pytest.fixture
    def resolver_with_validation(self):
        """Create resolver with validation enabled."""
        config = ArticleResolverConfig(
            validate_resolved_url=True,
            validation_timeout=5.0,
        )
        return IdentityResolver(config=config, debug=True)

    @pytest.fixture
    def resolver_without_validation(self):
        """Create resolver with validation disabled."""
        config = ArticleResolverConfig(validate_resolved_url=False)
        return IdentityResolver(config=config, debug=True)

    @pytest.mark.asyncio
    async def test_successful_validation(self, resolver_with_validation):
        """Test successful URL validation."""
        input_schema = ResolverInputSchema(url="https://example.com")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            result = await resolver_with_validation.arun(input_schema)
            assert (
                str(result.url) == "https://example.com/"
            )  # HttpUrl normalizes

    @pytest.mark.asyncio
    async def test_validation_failure_4xx(self, resolver_with_validation):
        """Test validation failure with 4xx status."""
        input_schema = ResolverInputSchema(url="https://example.com")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                return_value=mock_response,
            )

            with pytest.raises(ValueError, match="URL validation failed"):
                await resolver_with_validation.arun(input_schema)

    @pytest.mark.asyncio
    async def test_validation_timeout(self, resolver_with_validation):
        """Test validation timeout handling."""
        input_schema = ResolverInputSchema(url="https://example.com")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout"),
            )

            with pytest.raises(ValueError, match="URL validation timeout"):
                await resolver_with_validation.arun(input_schema)

    @pytest.mark.asyncio
    async def test_validation_request_error(self, resolver_with_validation):
        """Test validation request error handling."""
        input_schema = ResolverInputSchema(url="https://example.com")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head = AsyncMock(
                side_effect=httpx.RequestError("Connection failed"),
            )

            with pytest.raises(ValueError, match="URL validation failed"):
                await resolver_with_validation.arun(input_schema)

    @pytest.mark.asyncio
    async def test_validation_disabled(self, resolver_without_validation):
        """Test that validation can be disabled."""
        input_schema = ResolverInputSchema(url="https://example.com")

        # No mocking needed - validation should be skipped
        result = await resolver_without_validation.arun(input_schema)
        assert str(result.url) == "https://example.com/"  # HttpUrl normalizes

