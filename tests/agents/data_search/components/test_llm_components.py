#!/usr/bin/env python3
"""
Test script for LLM-driven data search components.

Tests the new scientific angle generation and CMR query generation workflow.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from akd.agents.data_search.components import (
    ScientificExpansionComponent,
    ScientificAnglesComponent, 
    CMRQueryGenerationComponent
)
from akd.agents.data_search.schemas import ScientificAngle


async def test_scientific_angles():
    """Test scientific angles generation."""
    print("🧠 Testing Scientific Angles Generation")
    print("-" * 50)
    
    # Test query from the example
    query = ("Can high levels of soil moisture and the land cover across coastal Louisiana "
             "sustain or affect the intensity of land falling tropical cyclones in the area?")
    
    print(f"Query: {query}")
    print()
    
    try:
        # Test scientific expansion (stub)
        expansion_component = ScientificExpansionComponent(debug=True)
        documents = await expansion_component.process(query)
        print(f"📚 Scientific expansion returned {len(documents)} documents (stub)")
        print()
        
        # Test scientific angles generation
        angles_component = ScientificAnglesComponent(debug=True)
        angles_result = await angles_component.process(query, documents)
        
        print(f"🎯 Generated {len(angles_result.angles)} scientific angles:")
        for i, angle in enumerate(angles_result.angles, 1):
            print(f"\n{i}. {angle.title}")
            print(f"   Justification: {angle.scientific_justification}")
        
        return angles_result.angles
        
    except Exception as e:
        print(f"❌ Error in angles generation: {e}")
        return []


async def test_cmr_query_generation(angles):
    """Test CMR query generation for the angles."""
    print("\n" + "=" * 50)
    print("⚙️ Testing CMR Query Generation")
    print("-" * 50)
    
    if not angles:
        print("❌ No angles to test with")
        return
    
    original_query = ("Can high levels of soil moisture and the land cover across coastal Louisiana "
                     "sustain or affect the intensity of land falling tropical cyclones in the area?")
    
    try:
        cmr_component = CMRQueryGenerationComponent(debug=True)
        
        # Test with first angle
        test_angle = angles[0]
        print(f"Testing with angle: {test_angle.title}")
        print()
        
        cmr_result = await cmr_component.process(test_angle, original_query)
        
        print(f"🔍 Generated {len(cmr_result.search_queries)} CMR search queries:")
        print(f"Reasoning: {cmr_result.reasoning}")
        print()
        
        for i, query in enumerate(cmr_result.search_queries, 1):
            print(f"Query {i}:")
            if query.keyword:
                print(f"  keyword: {query.keyword}")
            if query.platform:
                print(f"  platform: {query.platform}")
            if query.instrument:
                print(f"  instrument: {query.instrument}")
            if query.processing_level:
                print(f"  processing_level: {query.processing_level}")
            if query.temporal:
                print(f"  temporal: {query.temporal}")
            if query.bounding_box:
                print(f"  bounding_box: {query.bounding_box}")
            print()
            
    except Exception as e:
        print(f"❌ Error in CMR query generation: {e}")


async def test_schemas():
    """Test the new schemas work correctly."""
    print("\n" + "=" * 50)
    print("📋 Testing Schemas")
    print("-" * 50)
    
    try:
        # Test ScientificAngle
        angle = ScientificAngle(
            title="Soil Moisture Data",
            scientific_justification="Soil moisture affects evapotranspiration rates which influence atmospheric moisture content."
        )
        print(f"✅ ScientificAngle: {angle.title}")
        
        # Test CMRCollectionSearchParams
        from akd.agents.data_search.schemas import CMRCollectionSearchParams
        cmr_params = CMRCollectionSearchParams(
            keyword="soil moisture",
            platform="Terra",
            instrument="MODIS",
            processing_level="L3",
            temporal="2023-01-01T00:00:00Z,2023-12-31T23:59:59Z",
            bounding_box="-94.0,28.0,-88.0,32.0"
        )
        print(f"✅ CMRCollectionSearchParams: {cmr_params.keyword}, {cmr_params.platform}")
        
        print("✅ All schemas working correctly")
        
    except Exception as e:
        print(f"❌ Schema error: {e}")


async def main():
    """Run all tests."""
    print("🧪 LLM-Driven Data Search Components Test")
    print("=" * 50)
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Make sure your .env file contains the API key")
        return
    
    print("✅ OpenAI API key found")
    print()
    
    # Test schemas first
    await test_schemas()
    
    # Test scientific angles generation
    angles = await test_scientific_angles()
    
    # Test CMR query generation
    await test_cmr_query_generation(angles)
    
    print("\n" + "=" * 50)
    print("🎯 Test Complete!")
    print("This tests the new LLM-driven workflow components.")
    print("For full end-to-end testing, you'll need the CMR MCP server running.")


if __name__ == "__main__":
    asyncio.run(main())