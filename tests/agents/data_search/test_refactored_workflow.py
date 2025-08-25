#!/usr/bin/env python3
"""
Test script for the refactored CMR Data Search Agent workflow.

Tests the new LLM-driven scientific angle generation and CMR query formulation.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from akd.agents.data_search import CMRDataSearchAgent
from akd.agents.data_search._base import DataSearchAgentInputSchema


async def test_scientific_angles_workflow():
    """Test the scientific angles generation workflow."""
    print("🧪 Testing Scientific Angles Generation Workflow")
    print("=" * 60)
    
    # Test query about soil moisture and tropical cyclones
    test_query = ("Can high levels of soil moisture and the land cover across coastal Louisiana "
                  "sustain or affect the intensity of land falling tropical cyclones in the area?")
    
    print(f"🔍 Test Query: {test_query}")
    print()
    
    try:
        # Initialize the agent
        agent = CMRDataSearchAgent(debug=True)
        
        # Test just the scientific expansion component
        print("📚 Step 1: Testing Scientific Expansion Component")
        documents = await agent.scientific_expansion_component.process(test_query)
        print(f"   Retrieved {len(documents)} documents (stub implementation)")
        print()
        
        # Test the scientific angles component
        print("🧠 Step 2: Testing Scientific Angles Component")
        angles_output = await agent.scientific_angles_component.process(test_query, documents)
        print(f"   Generated {len(angles_output.angles)} scientific angles:")
        for i, angle in enumerate(angles_output.angles, 1):
            print(f"   {i}. {angle.title}")
            print(f"      Justification: {angle.scientific_justification[:100]}...")
        print()
        
        # Test CMR query generation for first angle
        if angles_output.angles:
            print("⚙️ Step 3: Testing CMR Query Generation Component")
            first_angle = angles_output.angles[0]
            cmr_output = await agent.cmr_query_generation_component.process(first_angle, test_query)
            print(f"   Generated {len(cmr_output.search_queries)} CMR search queries:")
            for i, query in enumerate(cmr_output.search_queries, 1):
                query_summary = []
                if query.keyword:
                    query_summary.append(f"keyword: {query.keyword}")
                if query.platform:
                    query_summary.append(f"platform: {query.platform}")
                if query.instrument:
                    query_summary.append(f"instrument: {query.instrument}")
                if query.processing_level:
                    query_summary.append(f"level: {query.processing_level}")
                print(f"   {i}. {', '.join(query_summary) if query_summary else 'empty query'}")
            print(f"   Reasoning: {cmr_output.reasoning[:100]}...")
        print()
        
        print("✅ Individual component tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_full_workflow():
    """Test the complete end-to-end workflow."""
    print("\n" + "=" * 60)
    print("🔬 Testing Complete End-to-End Workflow")
    print("=" * 60)
    
    test_query = ("Can high levels of soil moisture and the land cover across coastal Louisiana "
                  "sustain or affect the intensity of land falling tropical cyclones in the area?")
    
    print(f"🔍 Test Query: {test_query}")
    print()
    
    try:
        # Initialize the agent
        agent = CMRDataSearchAgent(debug=True)
        
        # Create input schema
        input_params = DataSearchAgentInputSchema(
            query=test_query,
            max_results=10
        )
        
        print("🚀 Running full workflow...")
        print("   Note: This will make actual requests to CMR MCP server")
        print("   Make sure the MCP server is running at http://localhost:8080")
        print()
        
        # Run the complete workflow
        # Note: This will likely fail if MCP server is not running
        # but we can still test the LLM components
        result = await agent.arun(input_params)
        
        print("📊 Workflow Results:")
        print(f"   Total granules found: {result.total_results}")
        print(f"   Collections searched: {len(result.collections_searched)}")
        print(f"   Search metadata: {result.search_metadata}")
        
        print("✅ Full workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Full workflow failed (expected if MCP server not running): {e}")
        print("   This is normal if the MCP server is not available")


async def main():
    """Main test function."""
    print("🧪 CMR Data Search Agent Refactor Tests")
    print("Testing LLM-driven scientific angle generation and CMR query formulation")
    print()
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Make sure your .env file contains the API key")
        return
    
    print("✅ OpenAI API key found")
    print()
    
    # Test individual components first
    await test_scientific_angles_workflow()
    
    # Then test full workflow (will likely fail without MCP server)
    await test_full_workflow()
    
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("✅ LLM components should work if OpenAI API is configured")
    print("❌ Full workflow requires MCP server at http://localhost:8080")
    print("📝 Check the generated scientific angles and CMR queries above")


if __name__ == "__main__":
    asyncio.run(main())