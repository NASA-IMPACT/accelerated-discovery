"""
Pytest tests for the field mapping system.

Tests:
1. FieldMappingRegistry - explicit and LLM mapping loading/saving
2. FieldMappingGenerator - LLM-based semantic mapping
3. WorkflowBuilder - three-tier mapping integration
4. End-to-end workflow with automatic field mapping
"""

import pytest
from loguru import logger

from akd.planner.registry import get_agent_registry, AgentRegistry
from akd.planner.field_mapping_registry import FieldMappingRegistry
from akd.planner.field_mapping_generator import FieldMappingGenerator
from akd.planner.workflow_builder import WorkflowBuilder
from akd.planner.structures import WorkflowPlan, AgentSuggestion


@pytest.fixture
def agent_registry():
    """Shared agent registry fixture."""
    return get_agent_registry()


@pytest.fixture
def mapping_registry():
    """Shared mapping registry fixture."""
    return FieldMappingRegistry()


@pytest.fixture
def workflow_builder(agent_registry, mapping_registry):
    """Shared workflow builder fixture."""
    return WorkflowBuilder(agent_registry, mapping_registry)


@pytest.fixture
def reset_singleton():
    """Reset AgentRegistry singleton after each test."""
    yield
    AgentRegistry._reset_singleton()


class TestFieldMappingRegistry:
    """Test FieldMappingRegistry basic functionality."""

    def test_explicit_mapping_loading(self, mapping_registry):
        """Test loading explicit mappings from JSON."""
        mapping = mapping_registry.get_mapping("deep_search", "gap_analysis")

        assert mapping is not None
        assert "search_results" in mapping
        assert mapping["search_results"] == "results"

    def test_save_and_retrieve_llm_mapping(self, mapping_registry):
        """Test saving and retrieving LLM mappings."""
        # Save test LLM mapping
        mapping_registry.save_llm_mapping(
            source_agent_id="test_source",
            target_agent_id="test_target",
            mapping={"output_field": "input_field"},
            confidence=0.95,
            user_approved=True,
            reasoning={"output_field": "Semantic match based on descriptions"}
        )

        # Retrieve saved LLM mapping
        llm_mapping = mapping_registry.get_mapping("test_source", "test_target")

        assert llm_mapping is not None
        assert llm_mapping.get("output_field") == "input_field"

    def test_mapping_info(self, mapping_registry):
        """Test retrieving mapping metadata."""
        info = mapping_registry.get_mapping_info("deep_search", "gap_analysis")

        assert info is not None
        assert info["type"] == "explicit"

    def test_invalid_agent_pair(self, mapping_registry):
        """Test that invalid agent pairs return None."""
        mapping = mapping_registry.get_mapping("nonexistent_agent", "gap_analysis")
        assert mapping is None


class TestFieldMappingGenerator:
    """Test FieldMappingGenerator LLM-based mapping."""

    @pytest.mark.asyncio
    async def test_llm_mapping_generation(self, agent_registry):
        """Test LLM-based field mapping generation (REAL API)."""
        generator = FieldMappingGenerator(temperature=0.0)

        # Get test agents for a realistic scenario
        deep_search = agent_registry.get_agent("deep_search")
        code_search = agent_registry.get_agent("code_search")

        assert deep_search is not None
        assert code_search is not None

        # Test: Can LLM map deep_search fields to code_search.query?
        required_fields = ["query"]

        result = await generator.generate_mapping(
            deep_search,
            code_search,
            required_fields
        )

        # Validate result
        assert result.overall_confidence > 0.0
        assert len(result.mappings) > 0

        # Check query mapping exists
        query_mapping = next((m for m in result.mappings if m.target_field == "query"), None)
        assert query_mapping is not None

        # Validate LLM chose a reasonable source field
        sensible_sources = ["report", "category", "results", "answer"]
        assert query_mapping.source_field in sensible_sources

    @pytest.mark.asyncio
    async def test_approval_threshold(self, agent_registry):
        """Test confidence-based approval threshold."""
        generator = FieldMappingGenerator(temperature=0.0)

        deep_search = agent_registry.get_agent("deep_search")
        code_search = agent_registry.get_agent("code_search")

        result = await generator.generate_mapping(
            deep_search,
            code_search,
            ["query"]
        )

        # Test approval threshold
        needs_approval = generator.should_request_approval(result, threshold=0.8)

        # Verify it returns a boolean
        assert isinstance(needs_approval, bool)

    @pytest.mark.asyncio
    async def test_approval_message_formatting(self, agent_registry):
        """Test formatting of approval messages."""
        generator = FieldMappingGenerator(temperature=0.0)

        deep_search = agent_registry.get_agent("deep_search")
        code_search = agent_registry.get_agent("code_search")

        result = await generator.generate_mapping(
            deep_search,
            code_search,
            ["query"]
        )

        # Test formatting
        approval_msg = generator.format_approval_message(
            deep_search,
            code_search,
            result
        )

        assert isinstance(approval_msg, str)
        assert len(approval_msg) > 0
        assert "Deep Search" in approval_msg or "deep_search" in approval_msg


class TestWorkflowBuilderMapping:
    """Test WorkflowBuilder three-tier mapping integration."""

    def test_explicit_mapping_integration(self, workflow_builder):
        """Test that explicit mappings are used in workflow building."""
        plan = WorkflowPlan(
            workflow_description="Test workflow",
            research_goal="Test mapping",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Literature search",
                    confidence=1.0,
                    required_inputs=["query"],
                    expected_outputs=["synthesis", "results"]
                ),
                AgentSuggestion(
                    agent_id="gap_analysis",
                    agent_name="Gap Analysis",
                    reason="Find gaps",
                    confidence=1.0,
                    required_inputs=["gap", "search_results"],
                    expected_outputs=["gaps"]
                )
            ],
        )

        filled_inputs = {
            "deep_search": {"query": "test query"},
            "gap_analysis": {"gap": "methodology"}
        }

        # Build workflow
        workflow = workflow_builder.build(plan, filled_inputs)

        assert len(workflow.nodes) == 2

        # Check io_map in gap_analysis node
        gap_node = next((n for n in workflow.nodes if n.type == "gap_analysis"), None)

        assert gap_node is not None
        assert gap_node.io_map is not None
        assert "search_results" in gap_node.io_map
        assert gap_node.io_map["search_results"] == "$.deep_search_1.outputs.results"

    def test_unmapped_field_detection(self, workflow_builder):
        """Test detection of unmapped fields."""
        plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="Test",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Search",
                    confidence=1.0,
                    required_inputs=["query"],
                    expected_outputs=["results"]
                ),
                AgentSuggestion(
                    agent_id="gap_analysis",
                    agent_name="Gap Analysis",
                    reason="Test",
                    confidence=1.0,
                    required_inputs=["search_results", "gap"],
                    expected_outputs=["output"]
                )
            ],
        )

        # Only deep_search query filled
        filled_inputs = {"deep_search": {"query": "test"}}
        unmapped = workflow_builder.identify_unmapped_fields(plan, filled_inputs)

        # Should detect gap as unmapped (search_results has explicit mapping)
        assert isinstance(unmapped, list)

    def test_partial_field_mapping(self, workflow_builder):
        """Test partial field mapping (some filled, some via io_map)."""
        plan = WorkflowPlan(
            workflow_description="Test",
            research_goal="Test",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Search",
                    confidence=1.0,
                    required_inputs=["query"],
                    expected_outputs=["results"]
                ),
                AgentSuggestion(
                    agent_id="gap_analysis",
                    agent_name="Gap Analysis",
                    reason="Analysis",
                    confidence=1.0,
                    required_inputs=["search_results", "gap"],
                    expected_outputs=["output"]
                )
            ],
        )

        # Partial inputs: query filled, gap filled, search_results should use io_map
        filled_inputs = {
            "deep_search": {"query": "test"},
            "gap_analysis": {"gap": "methodology"}
        }

        workflow = workflow_builder.build(plan, filled_inputs)
        gap_node = next((n for n in workflow.nodes if n.type == "gap_analysis"), None)

        assert gap_node is not None
        assert gap_node.io_map is not None
        assert "search_results" in gap_node.io_map
        assert gap_node.io_map["search_results"] == "$.deep_search_1.outputs.results"
        # Check that gap is NOT in io_map (it's filled directly)
        assert "gap" not in gap_node.io_map


class TestConfidenceBasedApproval:
    """Test confidence-based mapping approval."""

    def test_low_confidence_not_approved(self, mapping_registry):
        """Test that low-confidence mappings are not auto-approved."""
        test_mapping = {"field1": "field2"}

        # Low confidence - should not be auto-approved
        mapping_registry.save_llm_mapping(
            "low_conf_source", "low_conf_target",
            test_mapping,
            confidence=0.6,
            user_approved=False,
            reasoning={"field1": "Low confidence"}
        )

        low_info = mapping_registry.get_mapping_info("low_conf_source", "low_conf_target")

        assert low_info is not None
        assert not low_info.get("user_approved", True)

    def test_high_confidence_approved(self, mapping_registry):
        """Test that high-confidence mappings are auto-approved."""
        test_mapping = {"field1": "field2"}

        # High confidence - should be auto-approved
        mapping_registry.save_llm_mapping(
            "high_conf_source", "high_conf_target",
            test_mapping,
            confidence=0.95,
            user_approved=True,
            reasoning={"field1": "High confidence"}
        )

        high_info = mapping_registry.get_mapping_info("high_conf_source", "high_conf_target")

        assert high_info is not None
        assert high_info.get("user_approved", False)


class TestSpecificAgentWorkflows:
    """Test specific agent workflow combinations."""

    def test_deep_search_to_code_search(self, workflow_builder):
        """Test deep_search -> code_search mapping (report -> query)."""
        plan = WorkflowPlan(
            workflow_description="Literature then code search",
            research_goal="Find papers then related code",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="deep_search",
                    agent_name="Deep Search",
                    reason="Search literature",
                    confidence=1.0,
                    required_inputs=["query"],
                    expected_outputs=["results", "report"]
                ),
                AgentSuggestion(
                    agent_id="code_search",
                    agent_name="Code Search",
                    reason="Find related code",
                    confidence=1.0,
                    required_inputs=["query"],
                    expected_outputs=["results"]
                )
            ],
        )

        filled_inputs = {
            "deep_search": {"query": "AlphaFold protein structure"},
            "code_search": {}
        }

        workflow = workflow_builder.build(plan, filled_inputs)

        assert len(workflow.nodes) == 2

        # Check io_map in code_search node
        code_node = next((n for n in workflow.nodes if n.type == "code_search"), None)

        assert code_node is not None
        assert code_node.io_map is not None
        assert code_node.io_map.get("query") == "$.deep_search_1.outputs.report"

    def test_code_search_to_gap_analysis(self, workflow_builder):
        """Test code_search -> gap_analysis mapping."""
        plan = WorkflowPlan(
            workflow_description="Code search workflow",
            research_goal="Search code and analyze gaps",
            suggested_agents=[
                AgentSuggestion(
                    agent_id="code_search",
                    agent_name="Code Search",
                    reason="Search for code examples",
                    confidence=1.0,
                    required_inputs=["query"],
                    expected_outputs=["results"]
                ),
                AgentSuggestion(
                    agent_id="gap_analysis",
                    agent_name="Gap Analysis",
                    reason="Find gaps in code coverage",
                    confidence=1.0,
                    required_inputs=["gap", "search_results"],
                    expected_outputs=["gaps"]
                )
            ],
        )

        filled_inputs = {
            "code_search": {"query": "async patterns"},
            "gap_analysis": {"gap": "implementation"}
        }

        workflow = workflow_builder.build(plan, filled_inputs)

        assert len(workflow.nodes) == 2

        # Check io_map in gap_analysis node
        gap_node = next((n for n in workflow.nodes if n.type == "gap_analysis"), None)

        assert gap_node is not None
        assert gap_node.io_map is not None
        assert gap_node.io_map.get("search_results") == "$.code_search_1.outputs.results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
