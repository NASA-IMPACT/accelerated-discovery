"""
Comprehensive tests for scraper tools - rebuilt from scratch for current implementation.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import HttpUrl

from akd.tools.scrapers import (
    CompositeScraper,
    DoclingScraper,
    PyPaperBotScraper,
    PyPaperBotScraperConfig,
    ResearchArticleResolver,
    ScrapedMetadata,
    ScraperToolInputSchema,
    ScraperToolOutputSchema,
    WaterfallScraper,
    WaterfallScraperConfig,
)


class TestPyPaperBotScraperConfig:
    """Test PyPaperBotScraperConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PyPaperBotScraperConfig()
        assert config.max_runtime_seconds == 120
        assert config.scholar_pages == "1"
        assert config.scholar_results == 7
        assert config.enable_query_fallback is True
        assert config.annas_archive_mirror == "https://annas-archive.se/"
        assert config.proxy is None
        assert config.single_proxy is None
        assert config.selenium_chrome_version is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PyPaperBotScraperConfig(
            max_runtime_seconds=240,
            scholar_results=10,
            enable_query_fallback=False,
            annas_archive_mirror="https://example-mirror.com",
            proxy="proxy1,proxy2",
            selenium_chrome_version=120
        )
        assert config.max_runtime_seconds == 240
        assert config.scholar_results == 10
        assert config.enable_query_fallback is False
        assert config.annas_archive_mirror == "https://example-mirror.com"
        assert config.proxy == "proxy1,proxy2"
        assert config.selenium_chrome_version == 120

    def test_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            PyPaperBotScraperConfig(scholar_results=0)  # Below minimum
        
        with pytest.raises(ValueError):
            PyPaperBotScraperConfig(scholar_results=15)  # Above maximum


class TestPyPaperBotScraper:
    """Test PyPaperBotScraper functionality."""

    def test_initialization(self):
        """Test PyPaperBotScraper initialization."""
        scraper = PyPaperBotScraper(debug=True)
        assert scraper.config.debug is True
        assert scraper.config.max_runtime_seconds == 120
        assert hasattr(scraper, 'input_schema')
        assert hasattr(scraper, 'output_schema')
        assert hasattr(scraper, 'config_schema')

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = PyPaperBotScraperConfig(max_runtime_seconds=60)
        scraper = PyPaperBotScraper(config=config)
        assert scraper.config.max_runtime_seconds == 60

    def test_doi_extraction_from_url(self):
        """Test DOI extraction from various URL formats."""
        scraper = PyPaperBotScraper()
        
        # Test DOI.org URL
        url = "https://doi.org/10.1038/nature12345"
        doi = scraper._extract_doi_from_url(url)
        assert doi == "10.1038/nature12345"
        
        # Test Wiley URL
        url = "https://onlinelibrary.wiley.com/doi/full/10.1002/example"
        doi = scraper._extract_doi_from_url(url)
        assert doi == "10.1002/example"
        
        # Test Wiley PDF URL
        url = "https://onlinelibrary.wiley.com/doi/pdf/10.1002/example123"
        doi = scraper._extract_doi_from_url(url)
        assert doi == "10.1002/example123"
        
        # Test embedded DOI in URL
        url = "https://example.com/article/10.1234/test.article"
        doi = scraper._extract_doi_from_url(url)
        assert doi == "10.1234/test.article"
        
        # Test URL with no DOI
        url = "https://example.com/paper.html"
        doi = scraper._extract_doi_from_url(url)
        assert doi is None

    def test_query_extraction_arxiv(self):
        """Test query extraction from arXiv URLs."""
        scraper = PyPaperBotScraper()
        
        # Test arXiv abs URL
        url = "https://arxiv.org/abs/2401.12345"
        query = scraper._extract_query_from_url(url)
        assert query == "arXiv:2401.12345"
        
        # Test arXiv PDF URL
        url = "https://arxiv.org/pdf/2401.12345.pdf"
        query = scraper._extract_query_from_url(url)
        assert query == "arXiv:2401.12345"
    
    def test_query_extraction_pubmed(self):
        """Test query extraction from PubMed URLs."""
        scraper = PyPaperBotScraper()
        
        url = "https://pubmed.ncbi.nlm.nih.gov/12345678"
        query = scraper._extract_query_from_url(url)
        assert query == "PMID 12345678"
    
    def test_query_extraction_unsupported(self):
        """Test query extraction from unsupported URLs."""
        scraper = PyPaperBotScraper()
        
        url = "https://example.com/paper.html"
        query = scraper._extract_query_from_url(url)
        assert query is None

    def test_find_downloaded_pdf(self):
        """Test finding downloaded PDF files."""
        scraper = PyPaperBotScraper()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create a test PDF file
            pdf_file = tmp_path / "test.pdf"
            pdf_file.write_bytes(b"fake pdf content")
            
            found_pdf = scraper._find_downloaded_pdf(tmp_path)
            assert found_pdf == pdf_file

    def test_find_downloaded_pdf_empty_directory(self):
        """Test finding PDF in empty directory."""
        scraper = PyPaperBotScraper()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            found_pdf = scraper._find_downloaded_pdf(tmp_path)
            assert found_pdf is None

    @pytest.mark.asyncio
    async def test_run_pypaperbot_unavailable(self):
        """Test behavior when PyPaperBot is not available."""
        with patch('akd.tools.scrapers.pypaperbot_scraper._PYPAPERBOT_AVAILABLE', False):
            scraper = PyPaperBotScraper()
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = await scraper._run_pypaperbot_with_doi("10.1000/test", Path(tmp_dir))
                assert result is None
                
                # Test query method also returns None when unavailable
                result = await scraper._run_pypaperbot_with_query("test query", Path(tmp_dir))
                assert result is None

    @pytest.mark.asyncio
    async def test_arun_with_doi(self):
        """Test main execution with DOI extraction."""
        # Mock DOI extraction to return a DOI
        scraper = PyPaperBotScraper()
        
        with patch.object(scraper, '_extract_doi_from_url', return_value="10.1000/test") as mock_extract:
            with patch.object(scraper, '_run_pypaperbot_with_doi', return_value=None) as mock_run:
                input_params = ScraperToolInputSchema(url="https://doi.org/10.1000/test")
                
                result = await scraper._arun(input_params)
                
                assert isinstance(result, ScraperToolOutputSchema)
                assert result.content == ""
                assert result.metadata.query == "10.1000/test"
                mock_extract.assert_called_once()
                mock_run.assert_called_once()

    @pytest.mark.asyncio  
    async def test_arun_fallback_to_query(self):
        """Test fallback to query when no DOI found."""
        scraper = PyPaperBotScraper()
        
        with patch.object(scraper, '_extract_doi_from_url', return_value=None):
            with patch.object(scraper, '_extract_query_from_url', return_value="test query"):
                with patch.object(scraper, '_run_pypaperbot_with_query', return_value=None) as mock_query:
                    input_params = ScraperToolInputSchema(url="https://arxiv.org/abs/2401.12345")
                    
                    result = await scraper._arun(input_params)
                    
                    assert isinstance(result, ScraperToolOutputSchema)
                    # Check that query method was called
                    mock_query.assert_called_once()
                    args, _ = mock_query.call_args
                    assert args[0] == "test query"
                    assert isinstance(args[1], Path)
    
    @pytest.mark.asyncio
    async def test_arun_no_fallback_when_disabled(self):
        """Test that query fallback is skipped when disabled."""
        config = PyPaperBotScraperConfig(enable_query_fallback=False)
        scraper = PyPaperBotScraper(config=config)
        
        with patch.object(scraper, '_extract_doi_from_url', return_value=None):
            with patch.object(scraper, '_run_pypaperbot_with_query') as mock_query:
                input_params = ScraperToolInputSchema(url="https://arxiv.org/abs/2401.12345")
                
                result = await scraper._arun(input_params)
                
                assert isinstance(result, ScraperToolOutputSchema)
                # Query method should not be called when fallback is disabled
                mock_query.assert_not_called()
    


class TestCompositeScraper:
    """Test CompositeScraper functionality."""

    def test_initialization(self):
        """Test CompositeScraper initialization."""
        mock_scraper = MagicMock()
        scraper = CompositeScraper(mock_scraper, debug=True)
        
        assert len(scraper.scrapers) == 1
        assert scraper.scrapers[0] == mock_scraper
        assert scraper.debug is True

    def test_initialization_multiple_scrapers(self):
        """Test initialization with multiple scrapers."""
        mock1, mock2 = MagicMock(), MagicMock()
        scraper = CompositeScraper(mock1, mock2)
        
        assert len(scraper.scrapers) == 2
        assert scraper.scrapers[0] == mock1
        assert scraper.scrapers[1] == mock2

    @pytest.mark.asyncio
    async def test_successful_first_scraper(self):
        """Test that first scraper success returns immediately."""
        mock_scraper = AsyncMock()
        expected_output = ScraperToolOutputSchema(
            content="Test content",
            metadata=ScrapedMetadata(url="https://example.com", title="Test", query="test")
        )
        mock_scraper.arun.return_value = expected_output
        mock_scraper.input_schema = ScraperToolInputSchema
        
        scraper = CompositeScraper(mock_scraper)
        input_params = ScraperToolInputSchema(url="https://example.com")
        
        result = await scraper._arun(input_params)
        
        assert result.content == "Test content"
        assert str(result.metadata.url) == "https://example.com/"
        mock_scraper.arun.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_second_scraper(self):
        """Test fallback when first scraper fails."""
        # Mock failing first scraper
        mock_scraper1 = AsyncMock()
        mock_scraper1.arun.side_effect = Exception("First scraper failed")
        mock_scraper1.input_schema = ScraperToolInputSchema
        mock_scraper1.__class__.__name__ = "MockScraper1"
        
        # Mock successful second scraper
        mock_scraper2 = AsyncMock()
        expected_output = ScraperToolOutputSchema(
            content="Fallback content",
            metadata=ScrapedMetadata(url="https://example.com", title="Test", query="test")
        )
        mock_scraper2.arun.return_value = expected_output
        mock_scraper2.input_schema = ScraperToolInputSchema
        
        scraper = CompositeScraper(mock_scraper1, mock_scraper2)
        input_params = ScraperToolInputSchema(url="https://example.com")
        
        result = await scraper._arun(input_params)
        
        assert result.content == "Fallback content"
        assert result.metadata.extra["scraper"] == "AsyncMock"
        mock_scraper1.arun.assert_called_once()
        mock_scraper2.arun.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_content_continues_to_next(self):
        """Test that empty content causes fallback to next scraper."""
        # Mock first scraper returning empty content
        mock_scraper1 = AsyncMock()
        empty_output = ScraperToolOutputSchema(
            content="",  # Empty content
            metadata=ScrapedMetadata(url="https://example.com", title="Test", query="test")
        )
        mock_scraper1.arun.return_value = empty_output
        mock_scraper1.input_schema = ScraperToolInputSchema
        
        # Mock second scraper with content
        mock_scraper2 = AsyncMock()
        good_output = ScraperToolOutputSchema(
            content="Good content",
            metadata=ScrapedMetadata(url="https://example.com", title="Test", query="test")
        )
        mock_scraper2.arun.return_value = good_output
        mock_scraper2.input_schema = ScraperToolInputSchema
        
        scraper = CompositeScraper(mock_scraper1, mock_scraper2)
        input_params = ScraperToolInputSchema(url="https://example.com")
        
        result = await scraper._arun(input_params)
        
        assert result.content == "Good content"
        mock_scraper1.arun.assert_called_once()
        mock_scraper2.arun.assert_called_once()


class TestWaterfallScraperConfig:
    """Test WaterfallScraperConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WaterfallScraperConfig()
        assert config.request_timeout == 30.0
        assert config.follow_redirects is True
        assert "User-Agent" in config.browser_headers
        assert config.enable_prefetching is True

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_headers = {"User-Agent": "Custom Agent"}
        config = WaterfallScraperConfig(
            request_timeout=60.0,
            follow_redirects=False,
            browser_headers=custom_headers,
            enable_prefetching=False
        )
        assert config.request_timeout == 60.0
        assert config.follow_redirects is False
        assert config.browser_headers == custom_headers
        assert config.enable_prefetching is False


class TestWaterfallScraper:
    """Test WaterfallScraper functionality."""

    def test_initialization(self):
        """Test WaterfallScraper initialization."""
        mock_scraper = MagicMock()
        scraper = WaterfallScraper(mock_scraper, debug=True)
        assert scraper.debug is True
        assert scraper.config.request_timeout == 30.0
        assert len(scraper.scrapers) == 1
        assert len(scraper.resolvers) == 3  # ArxivResolver, ADSResolver, IdentityResolver

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = WaterfallScraperConfig(request_timeout=45.0)
        mock_scraper = MagicMock()
        scraper = WaterfallScraper(mock_scraper, config=config)
        assert scraper.config.request_timeout == 45.0

    def test_initialization_with_custom_resolvers(self):
        """Test initialization with custom resolvers."""
        mock_scraper = MagicMock()
        mock_resolver = MagicMock()
        scraper = WaterfallScraper(mock_scraper, resolvers=[mock_resolver])
        assert len(scraper.resolvers) == 1
        assert scraper.resolvers[0] == mock_resolver

    @pytest.mark.asyncio
    async def test_resolve_url_identity(self):
        """Test URL resolution with identity resolver."""
        mock_scraper = MagicMock()
        scraper = WaterfallScraper(mock_scraper, debug=True)
        
        # Test URL that doesn't match any resolver should return original
        original_url = "https://example.com/paper.pdf"
        resolved_url = await scraper._resolve_url(original_url)
        assert resolved_url == original_url

    @pytest.mark.asyncio
    async def test_prefetch_disabled(self):
        """Test behavior when prefetching is disabled."""
        config = WaterfallScraperConfig(enable_prefetching=False)
        mock_scraper = MagicMock()
        scraper = WaterfallScraper(mock_scraper, config=config)
        
        result = await scraper._prefetch_and_scrape("https://example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_arun_waterfall_execution(self):
        """Test waterfall execution through scrapers."""
        # Create mock scrapers
        mock_scraper1 = AsyncMock()
        mock_scraper1.arun.side_effect = Exception("First failed")
        mock_scraper1.__class__.__name__ = "MockScraper1"
        
        mock_scraper2 = AsyncMock()
        success_output = ScraperToolOutputSchema(
            content="Success content",
            metadata=ScrapedMetadata(url="https://example.com", title="Test", query="test")
        )
        mock_scraper2.arun.return_value = success_output
        mock_scraper2.input_schema = ScraperToolInputSchema
        mock_scraper2.__class__.__name__ = "MockScraper2"
        mock_scraper2.config = {}
        
        config = WaterfallScraperConfig(enable_prefetching=False)
        scraper = WaterfallScraper(mock_scraper1, mock_scraper2, config=config)
        
        input_params = ScraperToolInputSchema(url="https://example.com")
        result = await scraper._arun(input_params)
        
        assert result.content == "Success content"
        assert result.metadata.extra["scraper"] == "MockScraper2"
        assert str(result.metadata.url) == "https://example.com/"
        mock_scraper1.arun.assert_called_once()
        mock_scraper2.arun.assert_called_once()


class TestResearchArticleResolver:
    """Test ResearchArticleResolver functionality."""

    def test_initialization(self):
        """Test ResearchArticleResolver initialization."""
        mock_resolver = MagicMock()
        resolver = ResearchArticleResolver(mock_resolver, debug=True)
        
        assert len(resolver.resolvers) == 1
        assert resolver.resolvers[0] == mock_resolver
        assert resolver.debug is True

    def test_initialization_multiple_resolvers(self):
        """Test initialization with multiple resolvers."""
        mock1, mock2 = MagicMock(), MagicMock()
        resolver = ResearchArticleResolver(mock1, mock2)
        
        assert len(resolver.resolvers) == 2
        assert resolver.resolvers[0] == mock1
        assert resolver.resolvers[1] == mock2

    @pytest.mark.asyncio
    async def test_url_validation(self):
        """Test URL validation."""
        resolver = ResearchArticleResolver()
        
        # Test valid HTTP URL
        valid_url = HttpUrl("https://example.com")
        # Note: Base class raises NotImplementedError
        with pytest.raises(NotImplementedError):
            await resolver.validate_url(valid_url)


class TestScraperIntegration:
    """Integration tests for scraper interactions."""

    def test_all_scrapers_have_required_attributes(self):
        """Test that all scrapers have required class attributes."""
        scrapers_to_test = [
            CompositeScraper,
            PyPaperBotScraper,
            WaterfallScraper,
        ]
        
        for scraper_class in scrapers_to_test:
            # Test that they can be instantiated
            if scraper_class == CompositeScraper:
                # CompositeScraper needs at least one scraper
                instance = scraper_class(DoclingScraper())
            elif scraper_class == ResearchArticleResolver:
                # ResearchArticleResolver needs at least one resolver
                mock_resolver = MagicMock()
                instance = scraper_class(mock_resolver)
            else:
                instance = scraper_class()
            
            # Check required attributes exist
            assert hasattr(instance, 'debug')
            
            # For non-composite scrapers, check schema attributes
            if scraper_class != CompositeScraper and scraper_class != ResearchArticleResolver:
                assert hasattr(scraper_class, 'input_schema')
                assert hasattr(scraper_class, 'output_schema')
                assert hasattr(scraper_class, 'config_schema')
                assert hasattr(instance, 'config')

    @pytest.mark.asyncio
    async def test_composite_with_pypaperbot(self):
        """Test CompositeScraper with PyPaperBotScraper."""
        # Create a composite scraper with PyPaperBot
        pypaperbot = PyPaperBotScraper()
        composite = CompositeScraper(pypaperbot)
        
        # Test structure is correct
        assert len(composite.scrapers) == 1
        assert isinstance(composite.scrapers[0], PyPaperBotScraper)

    def test_scraper_metadata_structure(self):
        """Test that scrapers produce properly structured metadata."""
        # Test PyPaperBot config and metadata structure
        config = PyPaperBotScraperConfig()
        scraper = PyPaperBotScraper(config=config)
        
        # Verify config structure
        assert hasattr(config, 'max_runtime_seconds')
        assert hasattr(config, 'scholar_pages')
        assert hasattr(config, 'scholar_results')
        assert hasattr(config, 'enable_query_fallback')
        assert hasattr(config, 'annas_archive_mirror')

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in scrapers."""
        # Test PyPaperBot with valid URL but no DOI/content
        scraper = PyPaperBotScraper()
        
        # Use a valid URL format that won't have a DOI
        input_params = ScraperToolInputSchema(url="https://example.com/no-doi-here")
        result = await scraper._arun(input_params)
        
        assert isinstance(result, ScraperToolOutputSchema)
        assert result.content == ""
    
    @pytest.mark.asyncio
    async def test_waterfall_prefetch_error_handling(self):
        """Test WaterfallScraper prefetch error handling."""
        mock_scraper = AsyncMock()
        empty_output = ScraperToolOutputSchema(
            content="",
            metadata=ScrapedMetadata(url="https://example.com", title="", query="")
        )
        mock_scraper.arun.return_value = empty_output
        mock_scraper.input_schema = ScraperToolInputSchema
        mock_scraper.__class__.__name__ = "MockScraper"
        mock_scraper.config = {}
        
        # Test with prefetch enabled but HTTP error
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("HTTP Error")
            
            scraper = WaterfallScraper(mock_scraper, debug=True)
            input_params = ScraperToolInputSchema(url="https://example.com")
            result = await scraper._arun(input_params)
            
            # Should fall back to regular waterfall
            assert result.content == ""
            mock_scraper.arun.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pypaperbot_command_building(self):
        """Test PyPaperBot command building with various config options."""
        config = PyPaperBotScraperConfig(
            proxy="proxy1,proxy2",
            single_proxy="http://single-proxy.com",
            selenium_chrome_version=120
        )
        scraper = PyPaperBotScraper(config=config)
        
        with patch('akd.tools.scrapers.pypaperbot_scraper._PYPAPERBOT_AVAILABLE', True):
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b"", b"")
                mock_subprocess.return_value = mock_process
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    await scraper._run_pypaperbot_with_doi("10.1000/test", Path(tmp_dir))
                    
                    # Verify command was called with correct arguments
                    mock_subprocess.assert_called_once()
                    args = mock_subprocess.call_args[0]
                    cmd_list = list(args)
                    
                    assert "--proxy" in cmd_list
                    assert "proxy1,proxy2" in cmd_list
                    assert "--single-proxy" in cmd_list
                    assert "http://single-proxy.com" in cmd_list
                    assert "--selenium-chrome-version" in cmd_list
                    assert "120" in cmd_list

    def test_schema_compliance(self):
        """Test that all scrapers follow schema compliance."""
        # Test that input/output schemas are properly defined
        from akd.tools.scrapers import ScraperToolInputSchema, ScraperToolOutputSchema
        
        # Verify schema structure
        assert hasattr(ScraperToolInputSchema, 'model_fields')
        assert hasattr(ScraperToolOutputSchema, 'model_fields')
        
        # Test that required fields exist
        input_fields = ScraperToolInputSchema.model_fields
        output_fields = ScraperToolOutputSchema.model_fields
        
        assert 'url' in input_fields
        assert 'content' in output_fields
        assert 'metadata' in output_fields


# Test fixtures
@pytest.fixture
def sample_urls():
    """Sample URLs for testing."""
    return {
        'arxiv_abs': 'https://arxiv.org/abs/2401.12345',
        'arxiv_pdf': 'https://arxiv.org/pdf/2401.12345.pdf',
        'doi_org': 'https://doi.org/10.1038/nature12345',
        'wiley': 'https://onlinelibrary.wiley.com/doi/10.1002/example',
        'sciencedirect': 'https://www.sciencedirect.com/science/article/pii/S123456789',
        'generic': 'https://example.com/paper.pdf'
    }


@pytest.fixture
def sample_scraped_metadata():
    """Sample scraped metadata for testing."""
    return ScrapedMetadata(
        url="https://example.com",
        title="Sample Paper",
        query="test query",
        doi="10.1000/example",
        keywords=["test", "example"],
        author="John Doe",
        publication_date="2024-01-01"
    )


@pytest.fixture
def mock_pypaperbot_available():
    """Mock PyPaperBot as available."""
    with patch('akd.tools.scrapers.pypaperbot_scraper._PYPAPERBOT_AVAILABLE', True):
        yield


@pytest.fixture
def mock_pypaperbot_unavailable():
    """Mock PyPaperBot as unavailable."""
    with patch('akd.tools.scrapers.pypaperbot_scraper._PYPAPERBOT_AVAILABLE', False):
        yield