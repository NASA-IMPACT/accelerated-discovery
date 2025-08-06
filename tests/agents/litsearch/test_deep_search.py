"""Tests for the DeepLitSearchAgent."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Optional

from akd.agents.search import (
    DeepLitSearchAgent,
    DeepLitSearchAgentConfig,
    LitSearchAgentInputSchema,
    LitSearchAgentOutputSchema,
)
from akd.agents.relevancy import (
    MultiRubricRelevancyOutputSchema,
    TopicAlignmentLabel,
    ContentDepthLabel,
    RecencyRelevanceLabel,
    MethodologicalRelevanceLabel,
    EvidenceQualityLabel,
    ScopeRelevanceLabel,
)
from akd.structures import SearchResultItem
from akd.agents.query import QueryAgentOutputSchema, FollowUpQueryAgentOutputSchema


class TestDeepLitSearchAgentConfig:
    """Test the DeepLitSearchAgentConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DeepLitSearchAgentConfig()
        assert config.max_research_iterations == 5
        assert config.quality_threshold == 0.7
        assert config.auto_clarify is True
        assert config.max_clarifying_rounds == 1
        assert config.enable_streaming is True
        assert config.use_semantic_scholar is True
        assert config.enable_per_link_assessment is True
        assert config.min_relevancy_score == 0.3
        assert config.full_content_threshold == 0.7
        assert config.enable_full_content_scraping is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DeepLitSearchAgentConfig(
            max_research_iterations=10,
            quality_threshold=0.8,
            auto_clarify=False,
            max_clarifying_rounds=3,
            enable_streaming=False,
            use_semantic_scholar=False,
            enable_per_link_assessment=False,
            min_relevancy_score=0.5,
            enable_full_content_scraping=False
        )
        assert config.max_research_iterations == 10
        assert config.quality_threshold == 0.8
        assert config.auto_clarify is False
        assert config.max_clarifying_rounds == 3
        assert config.enable_streaming is False
        assert config.use_semantic_scholar is False
        assert config.enable_per_link_assessment is False
        assert config.min_relevancy_score == 0.5
        assert config.enable_full_content_scraping is False
    
    def test_config_validation(self):
        """Test configuration validation constraints."""
        # Test valid ranges
        config = DeepLitSearchAgentConfig(
            quality_threshold=0.0,
            min_relevancy_score=1.0,
            full_content_threshold=0.5
        )
        assert config.quality_threshold == 0.0
        assert config.min_relevancy_score == 1.0
        
        # Test invalid ranges
        with pytest.raises(ValueError):
            DeepLitSearchAgentConfig(quality_threshold=1.5)
        
        with pytest.raises(ValueError):
            DeepLitSearchAgentConfig(min_relevancy_score=-0.1)
        
        with pytest.raises(ValueError):
            DeepLitSearchAgentConfig(full_content_threshold=2.0)


class TestDeepLitSearchAgent:
    """Test the DeepLitSearchAgent."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        agent = DeepLitSearchAgent()
        assert isinstance(agent.config, DeepLitSearchAgentConfig)
        assert agent.config.max_research_iterations == 5
        assert agent.search_tool is not None
        assert agent.semantic_scholar_tool is not None  # enabled by default
        assert agent.query_agent is not None
        assert agent.followup_query_agent is not None
        assert agent.relevancy_agent is not None
        assert agent.link_relevancy_assessor is not None  # enabled by default
        assert agent.web_scraper is not None  # enabled by default
        assert agent.pdf_scraper is not None  # enabled by default
        assert agent.triage_component is not None
        assert agent.clarification_component is not None
        assert agent.instruction_component is not None
        assert agent.research_synthesis_component is not None
        assert agent.research_history == []
        assert agent.clarification_history == []
    
    def test_initialization_minimal_config(self):
        """Test initialization with minimal features enabled."""
        config = DeepLitSearchAgentConfig(
            use_semantic_scholar=False,
            enable_per_link_assessment=False,
            enable_full_content_scraping=False
        )
        agent = DeepLitSearchAgent(config=config)
        assert agent.semantic_scholar_tool is None
        assert agent.link_relevancy_assessor is None
        assert agent.web_scraper is None
        assert agent.pdf_scraper is None
    
    def test_initialization_custom_tools(self):
        """Test initialization with custom tools."""
        mock_search_tool = Mock()
        mock_relevancy_agent = Mock()
        
        agent = DeepLitSearchAgent(
            search_tool=mock_search_tool,
            relevancy_agent=mock_relevancy_agent
        )
        
        assert agent.search_tool is mock_search_tool
        assert agent.relevancy_agent is mock_relevancy_agent
    
    def test_initialization_custom_query_agents(self):
        """Test initialization with custom query agents."""
        from akd.agents.query import QueryAgent, FollowUpQueryAgent
        
        mock_query_agent = Mock(spec=QueryAgent)
        mock_followup_agent = Mock(spec=FollowUpQueryAgent)
        
        agent = DeepLitSearchAgent(
            query_agent=mock_query_agent,
            followup_query_agent=mock_followup_agent
        )
        
        assert agent.query_agent is mock_query_agent
        assert agent.followup_query_agent is mock_followup_agent


class TestDeepLitSearchAgentComponents:
    """Test embedded component functionality."""
    
    @pytest.mark.asyncio
    async def test_handle_triage(self):
        """Test triage handling with embedded component."""
        # Mock triage component
        mock_triage_component = AsyncMock()
        mock_triage_output = Mock()
        mock_triage_output.routing_decision = "clarification"
        mock_triage_output.needs_clarification = True
        mock_triage_output.reasoning = "Query is too broad and needs clarification"
        mock_triage_component.process.return_value = mock_triage_output
        
        agent = DeepLitSearchAgent()
        agent.triage_component = mock_triage_component
        
        result = await agent._handle_triage("broad research topic")
        
        assert result["routing_decision"] == "clarification"
        assert result["needs_clarification"] is True
        assert result["reasoning"] == "Query is too broad and needs clarification"
        mock_triage_component.process.assert_called_once_with("broad research topic")
    
    @pytest.mark.asyncio
    async def test_handle_clarification(self):
        """Test clarification handling with embedded component."""
        # Mock clarification component
        mock_clarification_component = AsyncMock()
        mock_clarification_component.process.return_value = (
            "enriched research query with specific parameters",
            ["What time period?", "Which methodology?", "What domain?"]
        )
        
        agent = DeepLitSearchAgent()
        agent.clarification_component = mock_clarification_component
        
        enriched_query, clarifications = await agent._handle_clarification("vague query")
        
        assert enriched_query == "enriched research query with specific parameters"
        assert len(clarifications) == 3
        assert "What time period?" in clarifications
        assert len(agent.clarification_history) == 3
        mock_clarification_component.process.assert_called_once_with("vague query", None)
    
    @pytest.mark.asyncio
    async def test_handle_clarification_with_mock_answers(self):
        """Test clarification handling with mock answers."""
        mock_clarification_component = AsyncMock()
        mock_clarification_component.process.return_value = (
            "refined query based on answers",
            ["Refined clarification"]
        )
        
        agent = DeepLitSearchAgent()
        agent.clarification_component = mock_clarification_component
        
        mock_answers = {"time_period": "2020-2024", "methodology": "systematic review"}
        enriched_query, clarifications = await agent._handle_clarification("query", mock_answers)
        
        assert enriched_query == "refined query based on answers"
        mock_clarification_component.process.assert_called_once_with("query", mock_answers)
    
    @pytest.mark.asyncio
    async def test_build_research_instructions(self):
        """Test research instruction building."""
        mock_instruction_component = AsyncMock()
        mock_instruction_component.process.return_value = "Detailed research instructions for comprehensive literature review"
        
        agent = DeepLitSearchAgent()
        agent.instruction_component = mock_instruction_component
        
        instructions = await agent._build_research_instructions(
            "climate change adaptation",
            ["Focus on urban areas", "Include recent studies"]
        )
        
        assert instructions == "Detailed research instructions for comprehensive literature review"
        mock_instruction_component.process.assert_called_once_with(
            "climate change adaptation",
            ["Focus on urban areas", "Include recent studies"]
        )


class TestDeepLitSearchAgentQueryGeneration:
    """Test query generation and refinement."""
    
    @pytest.mark.asyncio
    async def test_generate_initial_queries(self):
        """Test initial query generation from instructions."""
        # Mock query agent
        mock_query_agent = AsyncMock()
        mock_query_output = QueryAgentOutputSchema(
            queries=[
                "climate change urban adaptation strategies",
                "urban resilience climate impacts",
                "city-level climate adaptation planning",
                "urban heat island mitigation measures",
                "climate-resilient urban infrastructure"
            ]
        )
        
        with patch('akd.agents.query.QueryAgent', return_value=mock_query_agent):
            mock_query_agent.arun.return_value = mock_query_output
            
            agent = DeepLitSearchAgent()
            instructions = "Research urban climate adaptation strategies with focus on recent developments"
            
            queries = await agent._generate_initial_queries(instructions)
            
            assert len(queries) == 5
            assert "climate change urban adaptation strategies" in queries
            assert "urban resilience climate impacts" in queries
            mock_query_agent.arun.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_refined_queries(self):
        """Test refined query generation based on previous results."""
        # Mock follow-up query agent
        mock_followup_agent = AsyncMock()
        mock_followup_output = FollowUpQueryAgentOutputSchema(
            followup_queries=[
                "urban climate adaptation best practices",
                "climate resilience policy implementation",
                "nature-based urban adaptation solutions"
            ]
        )
        
        with patch('akd.agents.query.FollowUpQueryAgent', return_value=mock_followup_agent):
            mock_followup_agent.arun.return_value = mock_followup_output
            
            agent = DeepLitSearchAgent()
            
            previous_queries = ["climate adaptation", "urban planning"]
            mock_results = [
                SearchResultItem(
                    query="test",
                    url="http://example.com/1",
                    title="Urban Climate Adaptation",
                    content="This paper discusses various adaptation strategies for urban environments in the context of climate change.",
                    category="science"
                )
            ]
            instructions = "Research urban climate adaptation"
            
            refined_queries = await agent._generate_refined_queries(
                previous_queries, mock_results, instructions
            )
            
            assert len(refined_queries) == 3
            assert "urban climate adaptation best practices" in refined_queries
            mock_followup_agent.arun.assert_called_once()


class TestDeepLitSearchAgentSearchExecution:
    """Test search execution and result processing."""
    
    @pytest.mark.asyncio
    async def test_execute_searches_primary_tool_only(self):
        """Test search execution with primary tool only."""
        # Mock search tool
        mock_search_tool = AsyncMock()
        mock_search_result = Mock()
        mock_search_result.results = [
            SearchResultItem(
                query="test",
                url="http://example.com/1",
                title="Research Paper 1",
                content="Content of research paper 1",
                category="science"
            )
        ]
        mock_search_tool.arun.return_value = mock_search_result
        mock_search_tool.input_schema = LitSearchAgentInputSchema
        
        # Create agent with semantic scholar disabled
        config = DeepLitSearchAgentConfig(
            use_semantic_scholar=False,
            enable_per_link_assessment=False,
            enable_full_content_scraping=False
        )
        agent = DeepLitSearchAgent(config=config, search_tool=mock_search_tool)
        
        queries = ["artificial intelligence applications", "machine learning research"]
        results = await agent._execute_searches(queries)
        
        assert len(results) == 1
        assert results[0].title == "Research Paper 1"
        mock_search_tool.arun.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_searches_with_semantic_scholar(self):
        """Test search execution with both primary and semantic scholar tools."""
        # Mock primary search tool
        mock_search_tool = AsyncMock()
        mock_search_result = Mock()
        mock_search_result.results = [
            SearchResultItem(
                query="test",
                url="http://example.com/1",
                title="Primary Paper",
                content="Content from primary search",
                category="science"
            )
        ]
        mock_search_tool.arun.return_value = mock_search_result
        mock_search_tool.input_schema = LitSearchAgentInputSchema
        
        # Mock semantic scholar tool
        mock_semantic_scholar_tool = AsyncMock()
        mock_ss_result = Mock()
        mock_ss_result.results = [
            SearchResultItem(
                query="test",
                url="http://semanticscholar.com/1",
                title="Semantic Scholar Paper",
                content="Content from semantic scholar",
                category="science"
            )
        ]
        mock_semantic_scholar_tool.arun.return_value = mock_ss_result
        
        # Create agent with semantic scholar enabled
        config = DeepLitSearchAgentConfig(
            use_semantic_scholar=True,
            enable_per_link_assessment=False,
            enable_full_content_scraping=False
        )
        agent = DeepLitSearchAgent(
            config=config,
            search_tool=mock_search_tool
        )
        agent.semantic_scholar_tool = mock_semantic_scholar_tool
        
        queries = ["machine learning research"]
        results = await agent._execute_searches(queries)
        
        assert len(results) == 2
        assert any(r.title == "Primary Paper" for r in results)
        assert any(r.title == "Semantic Scholar Paper" for r in results)
        mock_search_tool.arun.assert_called_once()
        mock_semantic_scholar_tool.arun.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_searches_with_relevancy_assessment(self):
        """Test search execution with per-link relevancy assessment."""
        # Mock search tool
        mock_search_tool = AsyncMock()
        mock_search_result = Mock()
        mock_search_result.results = [
            SearchResultItem(
                query="test",
                url="http://example.com/1",
                title="Research Paper",
                content="Research content",
                category="science"
            )
        ]
        mock_search_tool.arun.return_value = mock_search_result
        mock_search_tool.input_schema = LitSearchAgentInputSchema
        
        # Mock link relevancy assessor
        mock_assessor = AsyncMock()
        mock_assessment_output = Mock()
        mock_assessment_output.filtered_results = [
            SearchResultItem(
                query="test",
                url="http://example.com/1",
                title="Research Paper",
                content="Research content",
                category="science"
            )
        ]
        mock_assessment_output.assessment_summary = "1 relevant result found"
        mock_assessor.arun.return_value = mock_assessment_output
        
        # Create agent with relevancy assessment enabled
        config = DeepLitSearchAgentConfig(
            use_semantic_scholar=False,
            enable_per_link_assessment=True,
            enable_full_content_scraping=False
        )
        agent = DeepLitSearchAgent(config=config, search_tool=mock_search_tool)
        agent.link_relevancy_assessor = mock_assessor
        
        queries = ["machine learning"]
        results = await agent._execute_searches(queries, original_query="machine learning")
        
        assert len(results) == 1
        assert results[0].title == "Research Paper"
        mock_assessor.arun.assert_called_once()
    
    def test_deduplicate_results(self):
        """Test result deduplication by URL and title."""
        agent = DeepLitSearchAgent()
        
        existing_results = [
            SearchResultItem(
                query="test",
                url="http://example.com/1",
                title="Paper 1",
                content="Content 1"
            ),
            SearchResultItem(
                query="test",
                url="http://example.com/2",
                title="Paper 2",
                content="Content 2"
            )
        ]
        
        new_results = [
            SearchResultItem(  # Duplicate URL
                query="test",
                url="http://example.com/1",
                title="Paper 1 Updated",
                content="Updated content"
            ),
            SearchResultItem(  # Duplicate title (case insensitive)
                query="test",
                url="http://example.com/3",
                title="PAPER 2",
                content="Different content"
            ),
            SearchResultItem(  # Truly new result
                query="test",
                url="http://example.com/4",
                title="Paper 3",
                content="New content"
            )
        ]
        
        deduplicated = agent._deduplicate_results(new_results, existing_results)
        
        assert len(deduplicated) == 1
        assert deduplicated[0].url == "http://example.com/4"
        assert deduplicated[0].title == "Paper 3"


class TestDeepLitSearchAgentQualityEvaluation:
    """Test research quality evaluation."""
    
    @pytest.mark.asyncio
    async def test_evaluate_research_quality_high(self):
        """Test quality evaluation with high-quality results."""
        # Mock relevancy agent
        mock_relevancy_agent = AsyncMock()
        mock_rubric_output = MultiRubricRelevancyOutputSchema(
            topic_alignment=TopicAlignmentLabel.ALIGNED,
            content_depth=ContentDepthLabel.COMPREHENSIVE,
            evidence_quality=EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE,
            methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
            recency_relevance=RecencyRelevanceLabel.CURRENT,
            scope_relevance=ScopeRelevanceLabel.IN_SCOPE,
            reasoning_steps=["High quality assessment"]
        )
        mock_relevancy_agent.arun.return_value = mock_rubric_output
        
        agent = DeepLitSearchAgent(relevancy_agent=mock_relevancy_agent)
        
        results = [
            SearchResultItem(
                query="test",
                url="http://example.com/1",
                title="High Quality Paper",
                content="Comprehensive research with strong methodology",
                category="science"
            )
        ]
        
        quality_score = await agent._evaluate_research_quality(results, "test query")
        
        assert quality_score == 1.0  # 6/6 positive rubrics
        mock_relevancy_agent.arun.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_evaluate_research_quality_mixed(self):
        """Test quality evaluation with mixed-quality results."""
        # Mock relevancy agent
        mock_relevancy_agent = AsyncMock()
        mock_rubric_output = MultiRubricRelevancyOutputSchema(
            topic_alignment=TopicAlignmentLabel.ALIGNED,  # positive
            content_depth=ContentDepthLabel.SURFACE_LEVEL,  # negative
            evidence_quality=EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE,  # positive
            methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_UNSOUND,  # negative
            recency_relevance=RecencyRelevanceLabel.CURRENT,  # positive
            scope_relevance=ScopeRelevanceLabel.OUT_OF_SCOPE,  # negative
            reasoning_steps=["Mixed quality assessment"]
        )
        mock_relevancy_agent.arun.return_value = mock_rubric_output
        
        agent = DeepLitSearchAgent(relevancy_agent=mock_relevancy_agent)
        
        results = [
            SearchResultItem(
                query="test",
                url="http://example.com/1",
                title="Mixed Quality Paper",
                content="Some good aspects but also issues",
                category="science"
            )
        ]
        
        quality_score = await agent._evaluate_research_quality(results, "test query")
        
        assert quality_score == 0.5  # 3/6 positive rubrics
    
    @pytest.mark.asyncio
    async def test_evaluate_research_quality_empty_results(self):
        """Test quality evaluation with empty results."""
        agent = DeepLitSearchAgent()
        
        quality_score = await agent._evaluate_research_quality([], "test query")
        
        assert quality_score == 0.0


class TestDeepLitSearchAgentIntegration:
    """Integration tests for DeepLitSearchAgent."""
    
    @pytest.mark.asyncio
    async def test_basic_research_workflow_without_guardrails(self):
        """Test basic research workflow without guardrails."""
        # Mock all embedded components
        mock_triage_component = AsyncMock()
        mock_triage_output = Mock()
        mock_triage_output.routing_decision = "direct_research"
        mock_triage_output.needs_clarification = False
        mock_triage_output.reasoning = "Query is clear enough for direct research"
        mock_triage_component.process.return_value = mock_triage_output
        
        mock_instruction_component = AsyncMock()
        mock_instruction_component.process.return_value = "Detailed research instructions for AI applications"
        
        mock_synthesis_component = AsyncMock()
        mock_synthesis_output = Mock()
        mock_synthesis_output.research_report = "Comprehensive research report on AI applications"
        mock_synthesis_output.key_findings = ["Finding 1", "Finding 2"]
        mock_synthesis_output.sources_consulted = ["Source 1", "Source 2"]
        mock_synthesis_output.evidence_quality_score = 0.85
        mock_synthesis_output.citations = ["Citation 1", "Citation 2"]
        mock_synthesis_component.synthesize.return_value = mock_synthesis_output
        
        # Mock search tools
        mock_search_tool = AsyncMock()
        mock_search_result = Mock()
        mock_search_result.results = [
            SearchResultItem(
                query="AI applications",
                url="http://example.com/ai1",
                title="AI Applications in Healthcare",
                content="This paper explores various AI applications in healthcare settings.",
                category="science"
            )
        ]
        mock_search_tool.arun.return_value = mock_search_result
        mock_search_tool.input_schema = LitSearchAgentInputSchema
        
        # Mock relevancy agent for quality evaluation
        mock_relevancy_agent = AsyncMock()
        mock_rubric_output = MultiRubricRelevancyOutputSchema(
            topic_alignment=TopicAlignmentLabel.ALIGNED,
            content_depth=ContentDepthLabel.COMPREHENSIVE,
            evidence_quality=EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE,
            methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
            recency_relevance=RecencyRelevanceLabel.CURRENT,
            scope_relevance=ScopeRelevanceLabel.IN_SCOPE,
            reasoning_steps=["High quality research"]
        )
        mock_relevancy_agent.arun.return_value = mock_rubric_output
        
        # Create agent with minimal configuration
        config = DeepLitSearchAgentConfig(
            max_research_iterations=2,
            quality_threshold=0.8,
            auto_clarify=False,
            use_semantic_scholar=False,
            enable_per_link_assessment=False,
            enable_full_content_scraping=False
        )
        agent = DeepLitSearchAgent(
            config=config,
            search_tool=mock_search_tool,
            relevancy_agent=mock_relevancy_agent
        )
        
        # Set mock components
        agent.triage_component = mock_triage_component
        agent.instruction_component = mock_instruction_component
        agent.research_synthesis_component = mock_synthesis_component
        
        # Run the agent
        input_params = LitSearchAgentInputSchema(
            query="artificial intelligence applications in healthcare",
            max_results=10
        )
        
        result = await agent._arun(input_params)
        
        # Verify results
        assert isinstance(result, LitSearchAgentOutputSchema)
        assert len(result.results) >= 1
        
        # Check that the research report is included as first result
        first_result = result.results[0]
        assert first_result["url"] == "deep-research://report"
        assert first_result["title"] == "Deep Research Report"
        assert first_result["content"] == "Comprehensive research report on AI applications"
        assert first_result["key_findings"] == ["Finding 1", "Finding 2"]
        assert first_result["quality_score"] == 0.85
        
        # Verify components were called
        mock_triage_component.process.assert_called_once()
        mock_instruction_component.process.assert_called_once()
        mock_synthesis_component.synthesize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_research_workflow_with_clarification(self):
        """Test research workflow that requires clarification."""
        # Mock triage component to require clarification
        mock_triage_component = AsyncMock()
        mock_triage_output = Mock()
        mock_triage_output.routing_decision = "clarification"
        mock_triage_output.needs_clarification = True
        mock_triage_output.reasoning = "Query needs clarification for better results"
        mock_triage_component.process.return_value = mock_triage_output
        
        # Mock clarification component
        mock_clarification_component = AsyncMock()
        mock_clarification_component.process.return_value = (
            "enriched AI applications query with specific healthcare focus",
            ["What specific healthcare domain?", "What time period?"]
        )
        
        # Mock instruction component
        mock_instruction_component = AsyncMock()
        mock_instruction_component.process.return_value = "Enhanced research instructions based on clarifications"
        
        # Mock synthesis component
        mock_synthesis_component = AsyncMock()
        mock_synthesis_output = Mock()
        mock_synthesis_output.research_report = "Enhanced research report with clarifications"
        mock_synthesis_output.key_findings = ["Enhanced finding 1"]
        mock_synthesis_output.sources_consulted = ["Enhanced source 1"]
        mock_synthesis_output.evidence_quality_score = 0.9
        mock_synthesis_output.citations = ["Enhanced citation 1"]
        mock_synthesis_component.synthesize.return_value = mock_synthesis_output
        
        # Mock search and relevancy as before
        mock_search_tool = AsyncMock()
        mock_search_result = Mock()
        mock_search_result.results = [
            SearchResultItem(
                query="enhanced AI query",
                url="http://example.com/enhanced",
                title="Enhanced AI Research",
                content="Enhanced research content",
                category="science"
            )
        ]
        mock_search_tool.arun.return_value = mock_search_result
        mock_search_tool.input_schema = LitSearchAgentInputSchema
        
        mock_relevancy_agent = AsyncMock()
        mock_rubric_output = MultiRubricRelevancyOutputSchema(
            topic_alignment=TopicAlignmentLabel.ALIGNED,
            content_depth=ContentDepthLabel.COMPREHENSIVE,
            evidence_quality=EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE,
            methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
            recency_relevance=RecencyRelevanceLabel.CURRENT,
            scope_relevance=ScopeRelevanceLabel.IN_SCOPE,
            reasoning_steps=["Enhanced quality"]
        )
        mock_relevancy_agent.arun.return_value = mock_rubric_output
        
        # Create agent with clarification enabled
        config = DeepLitSearchAgentConfig(
            auto_clarify=True,
            use_semantic_scholar=False,
            enable_per_link_assessment=False,
            enable_full_content_scraping=False
        )
        agent = DeepLitSearchAgent(
            config=config,
            search_tool=mock_search_tool,
            relevancy_agent=mock_relevancy_agent
        )
        
        # Set mock components
        agent.triage_component = mock_triage_component
        agent.clarification_component = mock_clarification_component
        agent.instruction_component = mock_instruction_component
        agent.research_synthesis_component = mock_synthesis_component
        
        # Run the agent
        input_params = LitSearchAgentInputSchema(query="vague AI query")
        
        result = await agent._arun(input_params)
        
        # Verify clarification was performed
        assert len(agent.clarification_history) == 2
        assert "What specific healthcare domain?" in agent.clarification_history
        
        # Verify enhanced research report
        first_result = result.results[0]
        assert first_result["content"] == "Enhanced research report with clarifications"
        
        # Verify all components were called
        mock_triage_component.process.assert_called_once()
        mock_clarification_component.process.assert_called_once()
        mock_instruction_component.process.assert_called_once()
        mock_synthesis_component.synthesize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_iterative_research_with_quality_threshold(self):
        """Test iterative research that meets quality threshold early."""
        # Mock components for simple workflow
        mock_triage_component = AsyncMock()
        mock_triage_output = Mock()
        mock_triage_output.needs_clarification = False
        mock_triage_component.process.return_value = mock_triage_output
        
        mock_instruction_component = AsyncMock()
        mock_instruction_component.process.return_value = "Research instructions"
        
        mock_synthesis_component = AsyncMock()
        mock_synthesis_output = Mock()
        mock_synthesis_output.research_report = "Quality research report"
        mock_synthesis_output.key_findings = ["Quality finding"]
        mock_synthesis_output.sources_consulted = ["Quality source"]
        mock_synthesis_output.evidence_quality_score = 0.95
        mock_synthesis_output.citations = ["Quality citation"]
        mock_synthesis_component.synthesize.return_value = mock_synthesis_output
        
        # Mock search tool to return different results per iteration
        mock_search_tool = AsyncMock()
        first_result = Mock()
        first_result.results = [
            SearchResultItem(
                query="iter1",
                url="http://example.com/1",
                title="First Iteration Paper",
                content="First iteration content",
                category="science"
            )
        ]
        second_result = Mock()
        second_result.results = [
            SearchResultItem(
                query="iter2",
                url="http://example.com/2",
                title="Second Iteration Paper",
                content="Second iteration content",
                category="science"
            )
        ]
        mock_search_tool.arun.side_effect = [first_result, second_result]
        mock_search_tool.input_schema = LitSearchAgentInputSchema
        
        # Mock relevancy agent to show quality improvement
        mock_relevancy_agent = AsyncMock()
        # First evaluation - moderate quality
        first_rubric = MultiRubricRelevancyOutputSchema(
            topic_alignment=TopicAlignmentLabel.ALIGNED,
            content_depth=ContentDepthLabel.SURFACE_LEVEL,
            evidence_quality=EvidenceQualityLabel.MEDIUM_QUALITY_EVIDENCE,
            methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
            recency_relevance=RecencyRelevanceLabel.CURRENT,
            scope_relevance=ScopeRelevanceLabel.IN_SCOPE,
            reasoning_steps=["Moderate quality"]
        )
        # Second evaluation - high quality (meets threshold)
        second_rubric = MultiRubricRelevancyOutputSchema(
            topic_alignment=TopicAlignmentLabel.ALIGNED,
            content_depth=ContentDepthLabel.COMPREHENSIVE,
            evidence_quality=EvidenceQualityLabel.HIGH_QUALITY_EVIDENCE,
            methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
            recency_relevance=RecencyRelevanceLabel.CURRENT,
            scope_relevance=ScopeRelevanceLabel.IN_SCOPE,
            reasoning_steps=["High quality achieved"]
        )
        mock_relevancy_agent.arun.side_effect = [first_rubric, second_rubric]
        
        # Create agent with quality threshold
        config = DeepLitSearchAgentConfig(
            max_research_iterations=5,
            quality_threshold=0.8,  # High threshold
            auto_clarify=False,
            use_semantic_scholar=False,
            enable_per_link_assessment=False,
            enable_full_content_scraping=False
        )
        agent = DeepLitSearchAgent(
            config=config,
            search_tool=mock_search_tool,
            relevancy_agent=mock_relevancy_agent
        )
        
        # Set mock components
        agent.triage_component = mock_triage_component
        agent.instruction_component = mock_instruction_component
        agent.research_synthesis_component = mock_synthesis_component
        
        # Run the agent
        input_params = LitSearchAgentInputSchema(query="quality research topic")
        
        result = await agent._arun(input_params)
        
        # Should stop after 2 iterations due to quality threshold
        # (First iteration: 4/6 = 0.67, Second iteration: average = (0.67 + 1.0)/2 = 0.835 > 0.8)
        assert result.iterations_performed >= 2
        
        # Verify both search results are included
        research_results = result.results[1:]  # Exclude the report
        assert len(research_results) >= 2
        
        # Verify quality threshold was met
        first_result = result.results[0]
        assert first_result["quality_score"] == 0.95


class TestDeepLitSearchAgentErrorHandling:
    """Test error handling in DeepLitSearchAgent."""
    
    @pytest.mark.asyncio
    async def test_component_failure_graceful_degradation(self):
        """Test graceful handling when embedded components fail."""
        # Mock triage component to fail
        mock_triage_component = AsyncMock()
        mock_triage_component.process.side_effect = Exception("Triage failed")
        
        # Mock other components to work normally
        mock_instruction_component = AsyncMock()
        mock_instruction_component.process.return_value = "Fallback instructions"
        
        mock_synthesis_component = AsyncMock()
        mock_synthesis_output = Mock()
        mock_synthesis_output.research_report = "Fallback report"
        mock_synthesis_output.key_findings = ["Fallback finding"]
        mock_synthesis_output.sources_consulted = ["Fallback source"]
        mock_synthesis_output.evidence_quality_score = 0.5
        mock_synthesis_output.citations = ["Fallback citation"]
        mock_synthesis_component.synthesize.return_value = mock_synthesis_output
        
        # Mock search tool
        mock_search_tool = AsyncMock()
        mock_search_result = Mock()
        mock_search_result.results = [
            SearchResultItem(
                query="test",
                url="http://example.com/fallback",
                title="Fallback Paper",
                content="Fallback content",
                category="science"
            )
        ]
        mock_search_tool.arun.return_value = mock_search_result
        mock_search_tool.input_schema = LitSearchAgentInputSchema
        
        # Mock relevancy agent
        mock_relevancy_agent = AsyncMock()
        mock_rubric_output = MultiRubricRelevancyOutputSchema(
            topic_alignment=TopicAlignmentLabel.ALIGNED,
            content_depth=ContentDepthLabel.SURFACE_LEVEL,
            evidence_quality=EvidenceQualityLabel.MEDIUM_QUALITY_EVIDENCE,
            methodological_relevance=MethodologicalRelevanceLabel.METHODOLOGICALLY_SOUND,
            recency_relevance=RecencyRelevanceLabel.CURRENT,
            scope_relevance=ScopeRelevanceLabel.IN_SCOPE,
            reasoning_steps=["Fallback quality"]
        )
        mock_relevancy_agent.arun.return_value = mock_rubric_output
        
        config = DeepLitSearchAgentConfig(
            auto_clarify=False,
            use_semantic_scholar=False,
            enable_per_link_assessment=False,
            enable_full_content_scraping=False
        )
        agent = DeepLitSearchAgent(
            config=config,
            search_tool=mock_search_tool,
            relevancy_agent=mock_relevancy_agent
        )
        
        # Set mock components
        agent.triage_component = mock_triage_component
        agent.instruction_component = mock_instruction_component
        agent.research_synthesis_component = mock_synthesis_component
        
        # Should handle triage failure and continue with degraded functionality
        input_params = LitSearchAgentInputSchema(query="test query")
        
        # This should not raise an exception despite triage failure
        result = await agent._arun(input_params)
        
        # Should still return results
        assert len(result.results) >= 1
        assert result.results[0]["content"] == "Fallback report"


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v"])