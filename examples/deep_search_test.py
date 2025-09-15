#!/usr/bin/env python3
"""
Test script to verify the refactored DeepLitSearchAgent works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from akd.agents.search._base import LitSearchAgentInputSchema
from akd.agents.search.deep_search import DeepLitSearchAgent, DeepLitSearchAgentConfig


async def test_deep_lit_search_agent():
    """Test the DeepLitSearchAgent with a simple query."""
    print("🧪 Testing DeepLitSearchAgent...")

    # Create a simple configuration
    config = DeepLitSearchAgentConfig(
        max_research_iterations=2,  # Keep it short for testing
        quality_threshold=0.5,
        auto_clarify=False,  # Skip clarification for this test
    )

    # Initialize the agent
    try:
        agent = DeepLitSearchAgent(config=config, debug=True)
        print("✅ Agent initialization successful")
        print(f"   - Agent has search_tool: {type(agent.search_tool).__name__}")
        print(f"   - Agent has query_agent: {type(agent.query_agent).__name__}")
        print(f"   - Agent has relevancy_agent: {type(agent.relevancy_agent).__name__}")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

    # Create a simple test query
    test_query = "machine learning applications in drug discovery"
    input_schema = LitSearchAgentInputSchema(
        query=test_query, category="science", max_results=5
    )

    print(f"\n🔍 Testing query: '{test_query}'")

    try:
        # Run the agent
        result = await agent.arun(input_schema)

        print("✅ Agent execution successful!")
        print(f"   - Number of results: {len(result.results)}")
        print(f"   - Iterations performed: {result.iterations_performed}")
        print(
            f"   - Has research report: {result.extra.get('research_report') is not None}"
        )
        print(f"   - Has key findings: {result.extra.get('key_findings') is not None}")
        print(
            f"   - Has evidence quality score: {result.extra.get('evidence_quality_score') is not None}"
        )
        print(f"   - Has citations: {result.extra.get('citations') is not None}")

        # Show first result if available
        if result.results:
            first_result = result.results[0]
            print("\n📄 First result preview:")
            print(f"   - Title: {first_result.get('title', 'N/A')[:100]}...")
            print(f"   - URL: {first_result.get('url', 'N/A')}")
            print(f"   - Has extra fields: {'extra' in first_result}")

        # Show synthesis summary if available
        if result.extra.get("research_report"):
            print("\n📊 Research synthesis preview:")
            print(
                f"   - Report length: {len(result.extra['research_report'])} characters"
            )
            if result.extra.get("key_findings"):
                print(f"   - Key findings count: {len(result.extra['key_findings'])}")
            print(
                f"   - Evidence quality score: {result.extra.get('evidence_quality_score', 'N/A')}"
            )

        return True

    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    print("🚀 Starting DeepLitSearchAgent test...")
    success = asyncio.run(test_deep_lit_search_agent())

    if success:
        print(
            "\n🎉 Test completed successfully! The refactored DeepLitSearchAgent is working."
        )
    else:
        print("\n💥 Test failed. There are issues with the refactored agent.")
        sys.exit(1)
