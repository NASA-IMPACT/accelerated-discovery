"""Tests for the QuestionAnsweringAgent to diagnose field name capitalization issues."""

import pytest

from akd.agents.search.answer import (
    QuestionAnsweringAgent,
    QuestionAnsweringAgentInputSchema,
    QuestionAnsweringAgentOutputSchema,
)
from akd.structures import SearchResultItem


class TestQuestionAnsweringAgent:
    """Test the QuestionAnsweringAgent in isolation."""

    @pytest.mark.asyncio
    async def test_basic_answer_generation(self):
        """Test that QuestionAnsweringAgent generates answers correctly."""
        agent = QuestionAnsweringAgent()

        # Create input similar to what the failing test provides
        input_data = QuestionAnsweringAgentInputSchema(
            query="artificial intelligence applications in healthcare",
            search_results=[
                SearchResultItem(
                    query="AI applications",
                    url="http://example.com/ai1",
                    title="AI Applications in Healthcare",
                    content="This paper explores various AI applications in healthcare settings, including diagnostic tools, treatment recommendations, and patient monitoring systems.",
                    category="science",
                ),
            ],
            additional_context="Comprehensive research report on AI applications in the healthcare sector.",
        )

        # Run the agent
        result = await agent.arun(input_data)

        # Verify the output structure
        assert isinstance(result, QuestionAnsweringAgentOutputSchema)
        assert hasattr(result, "answer")
        assert hasattr(result, "reasoning_traces")
        assert isinstance(result.answer, str)
        assert isinstance(result.reasoning_traces, list)
        assert len(result.answer) > 0
        assert len(result.reasoning_traces) > 0

        print("\n✓ QuestionAnsweringAgent test passed!")
        print(f"  Answer: {result.answer[:150]}...")
        print(f"  Reasoning traces: {len(result.reasoning_traces)}")

    @pytest.mark.asyncio
    async def test_answer_with_multiple_results(self):
        """Test QuestionAnsweringAgent with multiple search results."""
        agent = QuestionAnsweringAgent()

        input_data = QuestionAnsweringAgentInputSchema(
            query="machine learning in climate science",
            search_results=[
                SearchResultItem(
                    query="ML climate",
                    url="http://example.com/1",
                    title="Machine Learning for Climate Prediction",
                    content="Deep learning models are being used to improve climate predictions.",
                    category="science",
                ),
                SearchResultItem(
                    query="ML climate",
                    url="http://example.com/2",
                    title="Neural Networks in Weather Forecasting",
                    content="Neural networks have shown promising results in weather pattern recognition.",
                    category="science",
                ),
            ],
            additional_context=None,
        )

        result = await agent.arun(input_data)

        assert isinstance(result, QuestionAnsweringAgentOutputSchema)
        assert len(result.answer) > 0
        assert len(result.reasoning_traces) > 0

    @pytest.mark.asyncio
    async def test_answer_field_names_are_lowercase(self):
        """Test that the output fields use lowercase field names, not capitalized titles."""
        agent = QuestionAnsweringAgent()

        input_data = QuestionAnsweringAgentInputSchema(
            query="test query",
            search_results=[
                SearchResultItem(
                    query="test",
                    url="http://example.com/test",
                    title="Test Paper",
                    content="Test content for verification.",
                    category="science",
                ),
            ],
        )

        result = await agent.arun(input_data)

        # Verify the model dump uses lowercase field names
        result_dict = result.model_dump()
        print(f"\nResult dict keys: {list(result_dict.keys())}")

        # These should be lowercase
        assert "answer" in result_dict, f"Expected 'answer' but got keys: {list(result_dict.keys())}"
        assert "reasoning_traces" in result_dict, (
            f"Expected 'reasoning_traces' but got keys: {list(result_dict.keys())}"
        )

        # These should NOT exist (capitalized versions)
        assert "Answer" not in result_dict, "Field name should be lowercase 'answer', not 'Answer'"
        assert "Reasoning_Traces" not in result_dict, (
            "Field name should be lowercase 'reasoning_traces', not 'Reasoning_Traces'"
        )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
