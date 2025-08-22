"""
Comprehensive tests for AKD mapping system.

This test suite validates the waterfall mapping system with real agent schemas
from the AKD framework, ensuring proper data transformation between different
agent types in realistic scenarios.
"""

from typing import List

import pytest
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents.extraction import ExtractionInputSchema

# Import akd agent schemas
from akd.agents.query import QueryAgentInputSchema, QueryAgentOutputSchema
from akd.agents.relevancy import RelevancyAgentInputSchema
from akd.agents.search import LitSearchAgentInputSchema, LitSearchAgentOutputSchema
from akd.mapping.mappers import (
    DirectFieldMapper,
    LLMFallbackMapper,
    MapperConfig,
    MapperInput,
    SemanticFieldMapper,
    WaterfallMapper,
)


# Test schemas for mock scenarios
class LiteratureSearchInput(InputSchema):
    """Literature search input schema."""

    query: str = Field(description="Search query for literature")
    num_results: int = Field(default=5, description="Number of results to return")


class LiteratureSearchOutput(OutputSchema):
    """Literature search output schema."""

    documents: List[dict] = Field(description="List of found documents")
    search_query: str = Field(description="Original search query")
    total_results: int = Field(description="Total number of results found")


class ExtractionInput(InputSchema):
    """Data extraction input schema."""

    text_content: str = Field(description="Text content to extract from")
    document_title: str = Field(default="", description="Title of the document")


class ExtractionOutput(OutputSchema):
    """Data extraction output schema."""

    extractions: List[dict] = Field(description="List of extracted data points")
    confidence_score: float = Field(description="Confidence in extractions")


class QueryInput(InputSchema):
    """Query processing input schema."""

    query: str = Field(description="User's query")
    context: str = Field(default="", description="Additional context")


class QueryOutput(OutputSchema):
    """Query processing output schema."""

    response: str = Field(description="Generated response")
    query_intent: str = Field(description="Detected query intent")


class TestDirectFieldMapping:
    """Test direct field name matching with typed models."""

    @pytest.mark.asyncio
    async def test_exact_field_match(self):
        """Test mapping with exact field name matches."""
        mapper = DirectFieldMapper()

        # Create source model with matching field names
        source = QueryInput(
            query="carbon capture technology",
            context="materials science",
        )

        # Map to target schema with same field names
        result = await mapper.map_models(
            source_model=source,
            target_schema=QueryInput,  # Using same schema to test exact match
        )

        assert result["mapped"]["query"] == "carbon capture technology"
        assert result["mapped"]["context"] == "materials science"
        assert result["confidence"] == 1.0
        assert len(result["unmapped"]) == 0

    @pytest.mark.asyncio
    async def test_partial_field_match(self):
        """Test mapping with partial field matches."""
        mapper = DirectFieldMapper()

        # Create source model
        source = LiteratureSearchInput(query="renewable energy", num_results=10)

        # Map to schema with some matching fields
        result = await mapper.map_models(source_model=source, target_schema=QueryInput)

        assert result["mapped"]["query"] == "renewable energy"
        assert "context" not in result["mapped"]  # No matching field
        assert result["confidence"] == 0.5  # Only 1 of 2 source fields mapped
        assert "num_results" in result["unmapped"]

    @pytest.mark.asyncio
    async def test_mapping_with_hints(self):
        """Test mapping with explicit field hints."""
        mapper = DirectFieldMapper()

        source = LiteratureSearchOutput(
            documents=[{"title": "Paper 1"}],
            search_query="test query",
            total_results=1,
        )

        result = await mapper.map_models(
            source_model=source,
            target_schema=ExtractionInput,
            mapping_hints={"search_query": "text_content"},
        )

        assert result["mapped"]["text_content"] == "test query"
        assert result["confidence"] > 0.0


class TestSemanticFieldMapping:
    """Test semantic similarity matching."""

    @pytest.mark.asyncio
    async def test_semantic_similarity(self):
        """Test mapping with semantically similar field names."""
        config = MapperConfig(semantic_threshold=0.4)
        mapper = SemanticFieldMapper(config=config)

        # Create model with similar but not exact field names
        source = LiteratureSearchOutput(
            documents=[{"title": "Test"}],
            search_query="renewable energy",
            total_results=5,
        )

        # Map to schema with semantically similar fields
        result = await mapper.map_models(source_model=source, target_schema=QueryInput)

        # Should map search_query to query due to semantic similarity
        assert "query" in result["mapped"]
        assert result["mapped"]["query"] == "renewable energy"
        assert result["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_semantic_threshold_behavior(self):
        """Test semantic mapping with different thresholds."""
        # High threshold - more selective
        high_config = MapperConfig(semantic_threshold=0.9)
        mapper_high = SemanticFieldMapper(config=high_config)

        # Low threshold - more permissive
        low_config = MapperConfig(semantic_threshold=0.3)
        mapper_low = SemanticFieldMapper(config=low_config)

        source = LiteratureSearchOutput(
            documents=[],
            search_query="test",
            total_results=0,
        )

        result_high = await mapper_high.map_models(
            source_model=source,
            target_schema=QueryInput,
        )

        result_low = await mapper_low.map_models(
            source_model=source,
            target_schema=QueryInput,
        )

        # Low threshold should map more fields
        assert len(result_low["mapped"]) >= len(result_high["mapped"])


class TestLLMFallbackMapping:
    """Test LLM-based mapping fallback."""

    @pytest.mark.asyncio
    async def test_llm_fallback_no_api_key(self):
        """Test LLM fallback behavior without API key."""
        mapper = LLMFallbackMapper()

        if mapper.llm is None:
            source = ExtractionOutput(
                extractions=[{"field": "value"}],
                confidence_score=0.9,
            )

            result = await mapper.map_models(
                source_model=source,
                target_schema=QueryInput,
            )

            # Should fall back gracefully
            assert result["confidence"] == 0.1
            assert result["metadata"]["strategy"] == "string_fallback"

    @pytest.mark.asyncio
    async def test_llm_with_api_key(self):
        """Test LLM mapping with API key available."""
        mapper = LLMFallbackMapper()

        if mapper.llm is not None:
            # Create a complex source model
            source = LiteratureSearchOutput(
                documents=[
                    {
                        "title": "Solar Cells",
                        "abstract": "Advanced photovoltaic research",
                    },
                ],
                search_query="renewable energy technology",
                total_results=1,
            )

            # Test LLM transformation to a different schema
            result = await mapper.map_models(
                source_model=source,
                target_schema=ExtractionInput,
            )

            # Should use LLM successfully
            assert result["confidence"] > 0.5
            assert result["metadata"]["strategy"] == "llm_assisted"
            assert "text_content" in result["mapped"]
        else:
            pytest.skip("OpenAI API key not available for LLM testing")

    @pytest.mark.asyncio
    async def test_llm_with_complex_transformation(self):
        """Test LLM mapping with complex schema transformation."""
        mapper = LLMFallbackMapper()

        if mapper.llm is not None:
            # Test transformation between very different schemas
            source = QueryOutput(
                response="Solar panels achieve 22% efficiency in laboratory conditions",
                query_intent="information_seeking",
            )

            result = await mapper.map_models(
                source_model=source,
                target_schema=ExtractionOutput,
            )

            # LLM should intelligently extract information
            assert result["confidence"] > 0.3
            assert "extractions" in result["mapped"]
            assert "confidence_score" in result["mapped"]
        else:
            pytest.skip("OpenAI API key not available for LLM testing")

    @pytest.mark.asyncio
    async def test_llm_prompt_creation(self):
        """Test LLM prompt creation for complex mappings."""
        mapper = LLMFallbackMapper()

        if mapper.llm is not None:
            source = QueryInput(
                query="What is the efficiency of modern solar panels?",
                context="renewable energy research",
            )

            # Test the prompt creation method directly
            source_dict = mapper._convert_model_to_dict(source)
            prompt = mapper._create_transformation_prompt(
                source_dict,
                ExtractionInput,
                {"query": "text_content"},
            )

            # Verify prompt contains necessary information
            assert "text_content" in prompt
            assert "ExtractionInput" in prompt
            assert "What is the efficiency" in prompt
        else:
            pytest.skip("OpenAI API key not available for LLM testing")


class TestWaterfallMapper:
    """Test the main waterfall mapping orchestrator."""

    @pytest.mark.asyncio
    async def test_waterfall_direct_success(self):
        """Test waterfall with successful direct mapping."""
        mapper = WaterfallMapper()

        # Create models with exact field matches
        source = QueryInput(query="test query", context="test context")

        result = await mapper.arun(
            MapperInput(source_model=source, target_schema=QueryInput),
        )

        assert result.used_strategy == "DirectFieldMapper"
        assert result.mapping_confidence == 1.0
        assert result.mapped_model.query == "test query"
        assert result.mapped_model.context == "test context"

    @pytest.mark.asyncio
    async def test_waterfall_with_hints(self):
        """Test waterfall with mapping hints."""
        mapper = WaterfallMapper()

        source = LiteratureSearchOutput(
            documents=[{"title": "Paper"}],
            search_query="carbon materials",
            total_results=1,
        )

        result = await mapper.arun(
            MapperInput(
                source_model=source,
                target_schema=ExtractionInput,
                mapping_hints={
                    "search_query": "text_content",
                    "total_results": "document_title",
                },
            ),
        )

        assert result.mapping_confidence > 0.0
        assert result.mapped_model.text_content == "carbon materials"

    @pytest.mark.asyncio
    async def test_waterfall_caching(self):
        """Test caching functionality."""
        mapper = WaterfallMapper()

        source = QueryInput(query="test", context="cache test")

        # First call
        result1 = await mapper.arun(
            MapperInput(source_model=source, target_schema=QueryOutput),
        )

        # Second call should use cache
        result2 = await mapper.arun(
            MapperInput(source_model=source, target_schema=QueryOutput),
        )

        # Results should be identical
        assert result1.mapped_model.model_dump() == result2.mapped_model.model_dump()
        assert result2.transformation_metadata.get("from_cache") is True


class TestRealAgentMappings:
    """Test mappings between real AKD agent schemas."""

    @pytest.mark.asyncio
    async def test_query_agent_to_lit_agent(self):
        """Test mapping from QueryAgent output to LitAgent input."""
        mapper = WaterfallMapper()

        # QueryAgent generates search queries
        query_output = QueryAgentOutputSchema(
            queries=["solar cell efficiency", "perovskite photovoltaics"],
            category="science",
        )

        # Map to LitAgent input (should use first query)
        result = await mapper.arun(
            MapperInput(
                source_model=query_output,
                target_schema=LitSearchAgentInputSchema,
                mapping_hints={"queries": "query"},  # Map queries list to single query
            ),
        )

        assert result.mapping_confidence > 0.0
        assert hasattr(result.mapped_model, "query")

    @pytest.mark.asyncio
    async def test_lit_agent_to_extraction_agent(self):
        """Test mapping from LitAgent output to ExtractionAgent input."""
        mapper = WaterfallMapper()

        # Create realistic LitAgent output

        lit_output = LitSearchAgentOutputSchema(
            results=[
                {
                    "source": "https://example.com/solar-paper",
                    "result": {
                        "title": "Advanced Solar Cell Technologies",
                        "content": "Recent breakthroughs in perovskite solar cells...",
                    },
                },
            ],
            category="science",
        )

        result = await mapper.arun(
            MapperInput(source_model=lit_output, target_schema=ExtractionInputSchema),
        )

        assert result.mapping_confidence > 0.0
        assert hasattr(result.mapped_model, "content")

    @pytest.mark.asyncio
    async def test_extraction_input_to_relevancy_agent(self):
        """Test mapping from ExtractionAgent input to RelevancyAgent input."""
        mapper = WaterfallMapper()

        # Create extraction input with query and content
        extraction_input = ExtractionInputSchema(
            query="What is the efficiency of perovskite solar cells?",
            content="Recent studies show perovskite solar cells achieve 25% efficiency...",
        )

        result = await mapper.arun(
            MapperInput(
                source_model=extraction_input,
                target_schema=RelevancyAgentInputSchema,
            ),
        )

        # Should have high confidence for direct field mapping
        assert result.mapping_confidence > 0.8
        assert result.used_strategy == "DirectFieldMapper"
        assert (
            result.mapped_model.query
            == "What is the efficiency of perovskite solar cells?"
        )
        assert (
            result.mapped_model.content
            == "Recent studies show perovskite solar cells achieve 25% efficiency..."
        )

    @pytest.mark.asyncio
    async def test_full_agent_pipeline_mapping(self):
        """Test mapping through a complete agent pipeline."""
        mapper = WaterfallMapper()

        # Step 1: QueryAgent output -> LitAgent input
        query_output = QueryAgentOutputSchema(
            queries=["solar cell efficiency record", "photovoltaic maximum efficiency"],
            category="science",
        )

        lit_result = await mapper.arun(
            MapperInput(
                source_model=query_output,
                target_schema=LitSearchAgentInputSchema,
            ),
        )

        # Step 2: LitAgent output -> ExtractionAgent input
        from akd.structures import ExtractionDTO

        lit_output = LitSearchAgentOutputSchema(
            results=[
                ExtractionDTO(
                    source="research_paper.pdf",
                    result={
                        "content": "Solar cell efficiency has reached 47.1% using concentrated photovoltaics",
                    },
                ),
            ],
        )

        extraction_result = await mapper.arun(
            MapperInput(source_model=lit_output, target_schema=ExtractionInputSchema),
        )

        # Verify the pipeline works
        assert lit_result.mapping_confidence > 0.0
        assert extraction_result.mapping_confidence > 0.0
        assert hasattr(extraction_result.mapped_model, "content")

    @pytest.mark.asyncio
    async def test_real_agent_schema_compatibility(self):
        """Test compatibility between all real agent schemas."""
        mapper = WaterfallMapper()

        # Test all combinations of real agent schemas
        # removed unused variable real_schemas

        # Create sample instances for each schema
        sample_data = {
            QueryAgentInputSchema: QueryAgentInputSchema(
                query="test query",
                num_queries=3,
            ),
            QueryAgentOutputSchema: QueryAgentOutputSchema(
                queries=["query1", "query2"],
            ),
            LitSearchAgentInputSchema: LitSearchAgentInputSchema(
                query="literature search",
            ),
            ExtractionInputSchema: ExtractionInputSchema(
                query="test",
                content="content",
            ),
            RelevancyAgentInputSchema: RelevancyAgentInputSchema(
                query="test",
                content="content",
            ),
        }

        # Test mappings between compatible schemas
        compatible_pairs = [
            (QueryAgentInputSchema, LitSearchAgentInputSchema),  # Both have query field
            (
                ExtractionInputSchema,
                RelevancyAgentInputSchema,
            ),  # Both have query and content
        ]  # TODO: Add more pairs

        for source_schema, target_schema in compatible_pairs:
            if source_schema in sample_data:
                result = await mapper.arun(
                    MapperInput(
                        source_model=sample_data[source_schema],
                        target_schema=target_schema,
                    ),
                )

                # Should have reasonable confidence for compatible schemas
                assert result.mapping_confidence >= 0.5
                assert result.used_strategy == "DirectFieldMapper"


class TestAgentPairMappings:
    """Test realistic agent-to-agent mappings with mock schemas."""

    @pytest.mark.asyncio
    async def test_literature_to_extraction(self):
        """Test mapping from literature search to extraction."""
        mapper = WaterfallMapper()

        # Literature search output
        lit_output = LiteratureSearchOutput(
            documents=[
                {
                    "title": "Solar Cell Efficiency",
                    "content": "Recent advances in perovskite...",
                },
            ],
            search_query="solar cell efficiency",
            total_results=1,
        )

        result = await mapper.arun(
            MapperInput(source_model=lit_output, target_schema=ExtractionInput),
        )

        assert result.mapping_confidence > 0.0
        assert hasattr(result.mapped_model, "text_content")

    @pytest.mark.asyncio
    async def test_query_to_literature(self):
        """Test mapping from query to literature search."""
        mapper = WaterfallMapper()

        query_input = QueryInput(
            query="find papers on quantum computing",
            context="focus on recent algorithms",
        )

        result = await mapper.arun(
            MapperInput(source_model=query_input, target_schema=LiteratureSearchInput),
        )

        assert result.mapped_model.query == "find papers on quantum computing"
        assert result.mapping_confidence > 0.0


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_incompatible_schemas(self):
        """Test handling of incompatible schemas."""
        mapper = WaterfallMapper(config=MapperConfig(enable_llm_fallback=False))

        # Create source with no matching fields
        source = ExtractionOutput(extractions=[{"data": "value"}], confidence_score=0.8)

        result = await mapper.arun(
            MapperInput(source_model=source, target_schema=LiteratureSearchInput),
        )

        # Should handle gracefully even without matches
        assert result.mapping_confidence >= 0.0
        assert len(result.unmapped_fields) > 0

    @pytest.mark.asyncio
    async def test_strategy_disabling(self):
        """Test with disabled strategies."""
        config = MapperConfig(
            enable_direct_matching=False,
            enable_llm_fallback=False,
            semantic_threshold=0.4,
        )
        mapper = WaterfallMapper(config=config)

        source = LiteratureSearchOutput(
            documents=[],
            search_query="test",
            total_results=0,
        )

        result = await mapper.arun(
            MapperInput(source_model=source, target_schema=QueryInput),
        )

        # Should only use semantic mapping
        assert result.used_strategy in ["SemanticFieldMapper", "none"]


class TestConfigurationScenarios:
    """Test different configuration scenarios with real agents."""

    @pytest.mark.asyncio
    async def test_high_confidence_direct_mapping(self):
        """Test scenarios where direct mapping should have high confidence."""
        config = MapperConfig(
            enable_semantic_matching=False,
            enable_llm_fallback=False,
        )
        mapper = WaterfallMapper(config=config)

        # Same schema mapping should have perfect confidence
        source = QueryAgentInputSchema(query="renewable energy research", num_queries=5)

        result = await mapper.arun(
            MapperInput(source_model=source, target_schema=QueryAgentInputSchema),
        )

        assert result.used_strategy == "DirectFieldMapper"
        assert result.mapping_confidence == 1.0

    @pytest.mark.asyncio
    async def test_semantic_only_mapping(self):
        """Test mapping with only semantic matching enabled."""
        config = MapperConfig(
            enable_direct_matching=False,
            enable_llm_fallback=False,
            semantic_threshold=0.4,
        )
        mapper = WaterfallMapper(config=config)

        # Should use semantic mapping for similar field names
        source = LitSearchAgentOutputSchema(
            results=[],  # Empty results for simplicity
            category="science",
        )

        result = await mapper.arun(
            MapperInput(source_model=source, target_schema=QueryAgentInputSchema),
        )

        assert result.used_strategy in ["SemanticFieldMapper", "none"]

    @pytest.mark.asyncio
    async def test_llm_only_mapping(self):
        """Test mapping with only LLM fallback enabled."""
        config = MapperConfig(
            enable_direct_matching=False,
            enable_semantic_matching=False,
            enable_llm_fallback=True,
        )
        mapper = WaterfallMapper(config=config)

        source = ExtractionInputSchema(
            query="What is solar cell efficiency?",
            content="Solar cells convert sunlight to electricity...",
        )

        result = await mapper.arun(
            MapperInput(source_model=source, target_schema=RelevancyAgentInputSchema),
        )

        # Should use LLM mapping if available, or fail gracefully
        assert result.used_strategy in ["LLMFallbackMapper", "none"]
        assert result.mapping_confidence >= 0.0
        assert result.mapping_confidence >= 0.0
