import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Optional

from akd.structures import SearchResultItem
from akd.tools.resolvers._base import ResolverInputSchema, ResolverOutputSchema
from akd.tools.scrapers._base import ScrapedMetadata, ScraperToolBase, ScraperToolInputSchema, ScraperToolOutputSchema
from akd.tools.scrapers.composite import CompositeScraper
from akd.tools.resolvers import ResearchArticleResolver

# Assuming these are the imports from the module under test
from akd.tools.search import SearchPipeline, SearchPipelineConfig, SearchTool, SearchToolInputSchema, SearchToolOutputSchema


class MockSearchTool(SearchTool):
    """Mock search tool for testing"""
    
    def __init__(self, mock_results=None):
        super().__init__(config=None, debug=False)
        self.mock_results = mock_results or []
    
    async def _arun(self, params, **kwargs):
        return SearchToolOutputSchema(
            results=self.mock_results,
            category="test"
        )


class MockResolver:
    """Mock resolver for testing"""
    
    def __init__(self, should_fail=False, resolved_url=None):
        self.should_fail = should_fail
        self.resolved_url = resolved_url
        self.input_schema = ResolverInputSchema
    
    async def arun(self, params):
        if self.should_fail:
            raise Exception("Resolver failed")
        
        # Return a mock resolver output
        result_data = {
            "query": params.query,
            "title": params.title,
            "url": self.resolved_url or params.url,
            "content": params.content,
            "resolvers": ["MockResolver"]
            
        }
        return ResolverOutputSchema(**result_data)


class MockScraper:
    """Mock scraper for testing"""
    
    def __init__(self, content=None, should_fail=False, timeout=False):
        self.content = content
        self.should_fail = should_fail
        self.timeout = timeout
        self.input_schema = ScraperToolInputSchema
    
    async def arun(self, params):
        if self.timeout:
            await asyncio.sleep(10)  # Simulate timeout
        
        if self.should_fail:
            raise Exception("Scraper failed")
        
        return ScraperToolOutputSchema(
            content=self.content or "",
            metadata={
                "url": str(params.url),
                "title": "Scraped Title",
                "query": "",
            }
        )


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        SearchResultItem(
            query="test query",
            title="Test Article 1",
            url="http://example1.com/",
            content="Original content 1",
        ),
        SearchResultItem(
            query="test query",
            title="Test Article 2",
            url="http://example2.com/",
            content="Original content 2",
        )
    ]


@pytest.fixture
def basic_config():
    """Basic configuration for testing"""
    return SearchPipelineConfig(
        parallel_processing=False,
        max_concurrent_scrapes=2,
        enable_scraping=True,
        scraping_timeout=5,
        fail_on_scraping_errors=False
    )


class TestSearchPipelineConfig:
    """Tests for SearchPipelineConfig"""
    
    def test_default_config(self):
        config = SearchPipelineConfig()
        assert config.parallel_processing is True
        assert config.max_concurrent_scrapes == 5
        assert config.include_original_content is True
        assert config.enable_scraping is True
        assert config.scraping_timeout == 30
        assert config.fail_on_scraping_errors is False
        assert config.min_successful_scrapes is None
    
    def test_custom_config(self):
        config = SearchPipelineConfig(
            parallel_processing=False,
            max_concurrent_scrapes=10,
            enable_scraping=False,
            scraping_timeout=60,
            fail_on_scraping_errors=True,
            min_successful_scrapes=3
        )
        assert config.parallel_processing is False
        assert config.max_concurrent_scrapes == 10
        assert config.enable_scraping is False
        assert config.scraping_timeout == 60
        assert config.fail_on_scraping_errors is True
        assert config.min_successful_scrapes == 3


class TestSearchPipeline:
    """Tests for SearchPipeline class"""
    
    def test_init_with_defaults(self, sample_search_results):
        """Test initialization with default resolver and scraper"""
        mock_search_tool = MockSearchTool(sample_search_results)
        
        with patch('akd.tools.search.SearchPipeline._default_research_article_resolver') as mock_resolver, \
             patch('akd.tools.search.SearchPipeline.default_scraper') as mock_scraper:
            
            pipeline = SearchPipeline(mock_search_tool)
            
            assert pipeline.search_tool == mock_search_tool
            mock_resolver.assert_called_once_with(debug=False)
            mock_scraper.assert_called_once_with(debug=False)
    
    def test_init_with_custom_components(self, sample_search_results, basic_config):
        """Test initialization with custom resolver and scraper"""
        mock_search_tool = MockSearchTool(sample_search_results)
        mock_resolver = MockResolver()
        mock_scraper = MockScraper()
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            scraper=mock_scraper,
            config=basic_config,
            debug=True
        )
        
        assert pipeline.search_tool == mock_search_tool
        assert pipeline.resolver == mock_resolver
        assert pipeline.scraper == mock_scraper
        assert pipeline.debug is True
    
    @pytest.mark.asyncio
    async def test_resolve_essential_metadata_success(self, sample_search_results):
        """Test successful metadata resolution"""
        mock_search_tool = MockSearchTool()
        mock_resolver = MockResolver(resolved_url="http://resolved.example.com/")
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver
        )
        
        result = sample_search_results[0]
        resolved = await pipeline._resolve_essential_metadata(result)
        
        assert str(resolved.url) == "http://resolved.example.com/"
        assert resolved.resolvers == ["MockResolver"]
    
    @pytest.mark.asyncio
    async def test_resolve_essential_metadata_failure(self, sample_search_results):
        """Test metadata resolution failure fallback"""
        mock_search_tool = MockSearchTool()
        mock_resolver = MockResolver(should_fail=True)
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            debug=True
        )
        
        result = sample_search_results[0]
        resolved = await pipeline._resolve_essential_metadata(result)
        
        # Should fall back to original result data
        assert resolved.url == result.url
        assert resolved.title == result.title
    
    @pytest.mark.asyncio
    async def test_scrape_content_success(self):
        """Test successful content scraping"""
        mock_search_tool = MockSearchTool()
        mock_scraper = MockScraper(content="Scraped content from URL")
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            scraper=mock_scraper,
            debug=True
        )
        
        content = await pipeline._scrape_content("http://example.com/")
        
        assert content == "Scraped content from URL"
    
    @pytest.mark.asyncio
    async def test_scrape_content_empty(self):
        """Test scraping with empty content"""
        mock_search_tool = MockSearchTool()
        mock_scraper = MockScraper(content="")
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            scraper=mock_scraper,
            debug=True
        )
        
        content = await pipeline._scrape_content("http://example.com/")
        
        assert content is None
    
    @pytest.mark.asyncio
    async def test_scrape_content_failure(self):
        """Test scraping failure handling"""
        mock_search_tool = MockSearchTool()
        mock_scraper = MockScraper(should_fail=True)
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            scraper=mock_scraper,
            debug=True
        )
        
        content = await pipeline._scrape_content("http://example.com/")
        
        assert content is None
    
    @pytest.mark.asyncio
    async def test_scrape_content_timeout(self):
        """Test scraping timeout handling"""
        mock_search_tool = MockSearchTool()
        mock_scraper = MockScraper(timeout=True)
        
        config = SearchPipelineConfig(scraping_timeout=1)
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            scraper=mock_scraper,
            config=config,
            debug=True
        )
        
        content = await pipeline._scrape_content("http://example.com/")
        
        assert content is None
    
    @pytest.mark.asyncio
    async def test_process_single_result_full_pipeline(self, sample_search_results):
        """Test processing a single result through the full pipeline"""
        mock_search_tool = MockSearchTool()
        mock_resolver = MockResolver(resolved_url="http://resolved.example.com/")
        mock_scraper = MockScraper(content="Full scraped content")
        
        config = SearchPipelineConfig(
            enable_scraping=True,
            include_original_content=True
        )
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            scraper=mock_scraper,
            config=config
        )
        
        result = sample_search_results[0]
        enhanced = await pipeline._process_single_result(result)
        
        assert enhanced.title == result.title
        assert "Original content 1" in enhanced.content
        assert "Full scraped content" in enhanced.content
        assert enhanced.extra["scraping_performed"] is True
        assert enhanced.extra["full_text_scraped"] is True
        assert str(enhanced.extra["scraped_url"]) == "http://resolved.example.com/"
    
    @pytest.mark.asyncio
    async def test_process_single_result_scraping_disabled(self, sample_search_results):
        """Test processing with scraping disabled"""
        mock_search_tool = MockSearchTool()
        mock_resolver = MockResolver(resolved_url="http://resolved.example.com/")
        mock_scraper = MockScraper(content="Should not be used")
        
        config = SearchPipelineConfig(enable_scraping=False)
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            scraper=mock_scraper,
            config=config,
            debug=True
        )
        
        result = sample_search_results[0]
        enhanced = await pipeline._process_single_result(result)
        print(enhanced)
        
        assert enhanced.content == result.content  # Original content preserved
        assert enhanced.extra["scraping_performed"] is False
        assert enhanced.extra["full_text_scraped"] is False
    
    @pytest.mark.asyncio
    async def test_process_single_result_replace_content(self, sample_search_results):
        """Test processing with content replacement (not preserving original)"""
        mock_search_tool = MockSearchTool()
        mock_resolver = MockResolver()
        mock_scraper = MockScraper(content="Replacement content")
        
        config = SearchPipelineConfig(include_original_content=False)
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            scraper=mock_scraper,
            config=config
        )
        
        result = sample_search_results[0]
        enhanced = await pipeline._process_single_result(result)
        
        assert enhanced.content == "Replacement content"
        assert "Original content 1" not in enhanced.content
    
    @pytest.mark.asyncio
    async def test_process_resolver_error_handling(self, sample_search_results):
        """Test error handling in single result processing"""
        mock_search_tool = MockSearchTool()
        mock_resolver = MockResolver(should_fail=True)
        mock_scraper = MockScraper(should_fail=True)
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            scraper=mock_scraper,
            debug=True
        )
        
        result = sample_search_results[0]
        enhanced = await pipeline._process_single_result(result)
        
        assert enhanced.title == result.title
        assert enhanced.extra["resolver_used"] == []
        # scraping is performed (on original url) but fails
        assert enhanced.extra["scraping_performed"] is True
        assert enhanced.extra["full_text_scraped"] is False
    
    @pytest.mark.asyncio
    async def test_arun_no_search_results(self):
        """Test pipeline with no search results"""
        mock_search_tool = MockSearchTool([])  # Empty results
        
        pipeline = SearchPipeline(search_tool=mock_search_tool, debug=True)
        
        params = SearchToolInputSchema(queries=["test query"])
        result = await pipeline._arun(params)
        
        assert len(result.results) == 0
        assert result.category == "test"
    
    @pytest.mark.asyncio
    async def test_arun_sequential_processing(self, sample_search_results):
        """Test pipeline with sequential processing"""
        mock_search_tool = MockSearchTool(sample_search_results)
        mock_resolver = MockResolver()
        mock_scraper = MockScraper(content="Scraped content")
        
        config = SearchPipelineConfig(parallel_processing=False)
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            scraper=mock_scraper,
            config=config,
            debug=True
        )
        
        params = SearchToolInputSchema(queries=["test query"])
        result = await pipeline._arun(params)
        
        assert len(result.results) == 2
        for enhanced_result in result.results:
            assert enhanced_result.extra["full_text_scraped"] is True
    
    @pytest.mark.asyncio
    async def test_arun_parallel_processing(self, sample_search_results):
        """Test pipeline with parallel processing"""
        mock_search_tool = MockSearchTool(sample_search_results)
        mock_resolver = MockResolver()
        mock_scraper = MockScraper(content="Scraped content")
        
        config = SearchPipelineConfig(
            parallel_processing=True,
            max_concurrent_scrapes=2
        )
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            scraper=mock_scraper,
            config=config,
            debug=True
        )
        
        params = SearchToolInputSchema(queries=["test query"])
        result = await pipeline._arun(params)
        
        assert len(result.results) == 2
        for enhanced_result in result.results:
            assert enhanced_result.extra["full_text_scraped"] is True
    
    @pytest.mark.asyncio
    async def test_arun_fail_on_scraping_errors_false(self, sample_search_results):
        """Test pipeline with fail_on_scraping_errors=False"""
        mock_search_tool = MockSearchTool(sample_search_results)
        mock_resolver = MockResolver()
        mock_scraper = MockScraper(should_fail=True)
        
        config = SearchPipelineConfig(
            fail_on_scraping_errors=False,
            parallel_processing=True
        )
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            scraper=mock_scraper,
            config=config,
            debug=True
        )
        
        params = SearchToolInputSchema(queries=["test query"])
        result = await pipeline._arun(params)
        
        # Should still return results despite scraping failures
        assert len(result.results) == 2
    
    @pytest.mark.asyncio
    async def test_arun_fail_on_scraping_errors_true(self, sample_search_results):
        """Test pipeline with fail_on_scraping_errors=True"""
        mock_search_tool = MockSearchTool(sample_search_results)
        mock_resolver = MockResolver()
        mock_scraper = MockScraper(should_fail=True)
        
        config = SearchPipelineConfig(
            fail_on_scraping_errors=True,
            parallel_processing=False
        )
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            scraper=mock_scraper,
            config=config,
            debug=True
        )
        
        params = SearchToolInputSchema(queries=["test query"])


        
        # Should raise an exception due to scraping failure
        with pytest.raises(Exception):
            result = await pipeline._arun(params)
            print(result)

    

    def test_default_research_article_resolver(self):
        """Test default resolver creation"""
        mock_search_tool = MockSearchTool()
        pipeline = SearchPipeline(search_tool=mock_search_tool)
        
        resolver = pipeline._default_research_article_resolver(debug=True)
        
        # Should return a ResearchArticleResolver instance
        assert isinstance(resolver, ResearchArticleResolver)
    
    def test_default_scraper(self):
        """Test default scraper creation"""
        mock_search_tool = MockSearchTool()
        pipeline = SearchPipeline(search_tool=mock_search_tool)
        
        scraper = pipeline.default_scraper(debug=True)
        
        # Should return a CompositeScraper instance
        assert isinstance(scraper, CompositeScraper)


class TestIntegration:
    """Integration tests for SearchPipeline"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test the full pipeline with realistic components"""
        # Create sample data
        search_results = [
            SearchResultItem(
                query="machine learning healthcare",
                title="Machine Learning in Healthcare",
                url="http://arxiv.org/abs/1234.5678",
                content="Abstract: This paper discusses...",
                snippet="Machine learning applications"
            )
        ]
        
        # Mock components
        mock_search_tool = MockSearchTool(search_results)
        mock_resolver = MockResolver(resolved_url="http://arxiv.org/pdf/1234.5678.pdf")
        mock_scraper = MockScraper(content="Full paper content from PDF...")
        
        config = SearchPipelineConfig(
            parallel_processing=True,
            max_concurrent_scrapes=3,
            include_original_content=True,
            enable_scraping=True,
            scraping_timeout=10,
            fail_on_scraping_errors=False
        )
        
        pipeline = SearchPipeline(
            search_tool=mock_search_tool,
            resolver=mock_resolver,
            scraper=mock_scraper,
            config=config,
            debug=True
        )
        
        # Run the pipeline
        params = SearchToolInputSchema(queries=["machine learning healthcare"])
        result = await pipeline._arun(params)
        
        # Verify results
        assert len(result.results) == 1
        enhanced_result = result.results[0]
        
        assert enhanced_result.title == "Machine Learning in Healthcare"
        assert "Abstract: This paper discusses..." in enhanced_result.content
        assert "Full paper content from PDF..." in enhanced_result.content
        assert enhanced_result.extra["full_text_scraped"] is True
        assert str(enhanced_result.extra["scraped_url"]) == "http://arxiv.org/pdf/1234.5678.pdf"
        assert enhanced_result.extra["resolver_used"] == ["MockResolver"]

