#!/usr/bin/env python3
"""
Test script to validate the field prioritization improvements in guardrails.py
"""

import asyncio

from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents import InstructorBaseAgent
from akd.configs.guardrails_config import GuardrailsConfig
from akd.guardrails import apply_guardrails
from akd.tools.granite_guardian_tool import RiskDefinition


class TestInputSchema(InputSchema):
    """Test input schema with multiple potential text fields."""

    query: str = Field(..., description="Primary query field")
    content: str = Field(..., description="Secondary content field")
    extra_field: str = Field(
        ...,
        description="Extra field that should be lower priority",
    )


class TestOutputSchema(OutputSchema):
    """Test output schema with multiple potential text fields."""

    response: str = Field(..., description="Primary response field")
    answer: str = Field(..., description="Secondary answer field")
    result: str = Field(..., description="Result field")


class TestAgent(InstructorBaseAgent[TestInputSchema, TestOutputSchema]):
    """Test agent for field prioritization testing."""

    input_schema = TestInputSchema
    output_schema = TestOutputSchema

    async def _arun(self, params: TestInputSchema, **kwargs) -> TestOutputSchema:
        """Simple test implementation."""
        return TestOutputSchema(
            response=f"Processed: {params.query}",
            answer=f"Answer: {params.content}",
            result=f"Result: {params.extra_field}",
        )


async def test_field_prioritization():
    """Test that field prioritization works correctly."""
    print("Testing field prioritization in guardrails...")

    # Get default field values from config
    default_config = GuardrailsConfig()

    # Create test agent with custom field priorities
    agent = TestAgent()

    # Test 1: Default field priorities
    print("\n1. Testing default field priorities...")
    guarded_agent_default = apply_guardrails(
        component=agent,
        input_guardrails=[
            RiskDefinition.JAILBREAK,
        ],  # Need non-empty list to apply guardrails
        output_guardrails=[RiskDefinition.ANSWER_RELEVANCE],
    )

    print(f"Default input fields: {guarded_agent_default.input_fields}")
    print(f"Default output fields: {guarded_agent_default.output_fields}")
    assert guarded_agent_default.input_fields == default_config.input_fields
    assert guarded_agent_default.output_fields == default_config.output_fields

    # Test 2: Custom field priorities
    print("\n2. Testing custom field priorities...")
    custom_input_fields = ["content", "query"]  # Reverse priority
    custom_output_fields = ["answer", "response"]  # Reverse priority

    guarded_agent_custom = apply_guardrails(
        component=agent,
        input_guardrails=[RiskDefinition.JAILBREAK],
        output_guardrails=[RiskDefinition.ANSWER_RELEVANCE],
        input_fields=custom_input_fields,
        output_fields=custom_output_fields,
    )

    print(f"Custom input fields: {guarded_agent_custom.input_fields}")
    print(f"Custom output fields: {guarded_agent_custom.output_fields}")
    assert guarded_agent_custom.input_fields == custom_input_fields
    assert guarded_agent_custom.output_fields == custom_output_fields

    # Test 3: Field extraction with different priorities
    print("\n3. Testing field extraction...")

    # Create test data where different fields have different content
    test_input = TestInputSchema(
        query="PRIMARY_QUERY_CONTENT",
        content="SECONDARY_CONTENT",
        extra_field="EXTRA_FIELD_CONTENT",
    )

    # Test default extraction (should prioritize 'query' first)
    default_text = guarded_agent_default._extract_text_content(
        test_input,
        default_config.input_fields,
    )
    print(f"Default extraction result: {default_text}")
    assert "PRIMARY_QUERY_CONTENT" in default_text

    # Test custom extraction (should prioritize 'content' first)
    custom_text = guarded_agent_custom._extract_text_content(
        test_input,
        custom_input_fields,
    )
    print(f"Custom extraction result: {custom_text}")
    assert "SECONDARY_CONTENT" in custom_text

    print("\n✅ All field prioritization tests passed!")


if __name__ == "__main__":
    asyncio.run(test_field_prioritization())
