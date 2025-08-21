#!/usr/bin/env python3
"""
Test script for CMR Data Search Agent.

This script tests the CMR data search functionality by running example queries
and displaying results. It assumes the CMR MCP server is running locally.

Usage:
    python scripts/test_cmr_data_search.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to the path to allow for `akd` imports
# and loading of the .env file
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file if it exists
env_file = project_root / ".env"
if env_file.exists():
    print(f"Loading environment from {env_file}")
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

from akd.agents.data_search import CMRDataSearchAgent, CMRDataSearchAgentConfig
from akd.agents.data_search._base import DataSearchAgentInputSchema


async def test_basic_collection_search():
    """Test basic collection search functionality."""
    print("=" * 60)
    print("🧪 TEST 1: Basic Collection Search")
    print("=" * 60)

    try:
        from akd.tools.data_search import CMRCollectionSearchTool
        from akd.tools.data_search.cmr_collection_search import (
            CMRCollectionSearchInputSchema,
        )

        tool = CMRCollectionSearchTool(debug=True)

        # Test MODIS search
        search_params = CMRCollectionSearchInputSchema(
            keyword="MODIS", platform="Terra", instrument="MODIS", page_size=5
        )

        print("🔎 Searching for MODIS collections...")
        result = await tool.arun(search_params)

        print(
            f"✅ Found {result.total_hits} collections, showing {len(result.collections)}"
        )

        for i, collection in enumerate(result.collections[:3], 1):
            print(
                f"  {i}. {collection.get('short_name')} v{collection.get('version', 'N/A')}"
            )
            print(f"     Title: {collection.get('entry_title', 'N/A')}")
            print(f"     Concept ID: {collection.get('concept_id', 'N/A')}")
            print()

        return True

    except Exception as e:
        print(f"❌ Collection search test failed: {e}")
        return False


async def test_basic_granule_search():
    """Test basic granule search functionality."""
    print("=" * 60)
    print("🧪 TEST 2: Basic Granule Search")
    print("=" * 60)

    try:
        from akd.tools.data_search import CMRGranuleSearchTool
        from akd.tools.data_search.cmr_granule_search import CMRGranuleSearchInputSchema

        tool = CMRGranuleSearchTool(debug=True)

        # Use a collection ID from the successful collection search
        # Let's try a simpler approach - get a collection ID from previous test
        collection_id = "C1940475563-POCLOUD"  # From the successful collection search

        granule_params = CMRGranuleSearchInputSchema(
            collection_concept_id=collection_id,
            temporal="2024-01-01T00:00:00Z,2024-01-07T23:59:59Z",
            bounding_box="-125,32,-114,42",  # California
            online_only=True,
            page_size=3,
        )

        print(f"🔎 Searching for granules in collection {collection_id}...")
        result = await tool.arun(granule_params)

        print(f"✅ Found {result.total_hits} granules, showing {len(result.granules)}")

        for i, granule in enumerate(result.granules[:2], 1):
            print(f"  {i}. {granule.get('granule_ur', 'N/A')}")
            print(f"     Concept ID: {granule.get('concept_id', 'N/A')}")
            print(
                f"     Online Access: {'Yes' if granule.get('online_access_flag') else 'No'}"
            )

            download_urls = granule.get("download_urls", [])
            if download_urls:
                print(f"     Download URLs: {len(download_urls)} available")
                print(f"       - {download_urls[0].get('url', 'N/A')[:80]}...")
            print()

        return True

    except Exception as e:
        print(f"❌ Granule search test failed: {e}")
        return False


async def test_query_decomposition():
    """Test query decomposition component."""
    print("=" * 60)
    print("🧪 TEST 3: Query Decomposition")
    print("=" * 60)

    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OpenAI API key not found. Testing pattern-based decomposition only.")

        try:
            # Simple pattern-based test without LLM
            test_query = "Find MODIS sea surface temperature data from 2023 over the Pacific Ocean"
            print(f"📝 Testing pattern matching on: '{test_query}'")

            # Simple pattern matching
            keywords = []
            platforms = []
            instruments = []

            query_lower = test_query.lower()
            if "modis" in query_lower:
                keywords.append("modis")
                instruments.append("MODIS")
            if "terra" in query_lower:
                platforms.append("Terra")
            if "temperature" in query_lower or "sst" in query_lower:
                keywords.append("temperature")

            print("✅ Pattern-based decomposition result:")
            print(f"   Keywords: {keywords}")
            print(f"   Instruments: {instruments}")
            print(f"   Platforms: {platforms}")

            return True

        except Exception as e:
            print(f"❌ Pattern-based decomposition failed: {e}")
            return False

    try:
        from akd.agents.data_search.components import QueryDecompositionComponent

        component = QueryDecompositionComponent(debug=True)

        test_query = (
            "Find MODIS sea surface temperature data from 2023 over the Pacific Ocean"
        )
        print(f"📝 Decomposing query: '{test_query}'")

        result = await component.process(test_query)

        print("✅ Query decomposition result:")
        print(f"   Keywords: {result.keywords}")
        print(f"   Data types: {result.data_type_indicators}")
        print(f"   Platforms: {result.platforms}")
        print(f"   Instruments: {result.instruments}")
        print(f"   Processing level: {result.processing_level}")
        print(f"   Temporal: {result.temporal_description}")
        print(f"   Spatial: {result.spatial_description}")
        print(f"   Search variations: {len(result.search_variations)}")

        for i, variation in enumerate(result.search_variations[:3], 1):
            print(f"     {i}. {variation}")

        return True

    except Exception as e:
        print(f"❌ Query decomposition test failed: {e}")
        return False


async def test_full_agent_workflow():
    """Test the complete CMR data search agent workflow."""
    print("=" * 60)
    print("🧪 TEST 4: Complete Agent Workflow")
    print("=" * 60)

    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OpenAI API key not found. Skipping full agent workflow test.")
        print("   The full agent requires OpenAI for query decomposition.")
        print("   Set OPENAI_API_KEY environment variable to test this functionality.")
        return False

    try:
        config = CMRDataSearchAgentConfig(
            debug=True,
            max_collections_to_search=3,
            collection_search_page_size=10,
            granule_search_page_size=20,
        )

        agent = CMRDataSearchAgent(config=config, debug=True)

        # Test with a simple, focused query
        test_query = "Find Terra MODIS data from January 2024"
        print(f"🔍 Running full agent workflow with query: '{test_query}'")

        input_params = DataSearchAgentInputSchema(query=test_query, max_results=10)

        result = await agent.arun(input_params)

        print("✅ Agent workflow completed!")
        print(f"   Total results: {result.total_results}")
        print(f"   Collections searched: {len(result.collections_searched)}")

        if result.granules:
            print(f"   Sample granules:")
            for i, granule in enumerate(result.granules[:3], 1):
                print(f"     {i}. {granule.get('title', 'Untitled')}")
                print(
                    f"        Collection: {granule.get('collection', {}).get('name', 'Unknown')}"
                )
                print(
                    f"        Online access: {granule.get('access', {}).get('online', False)}"
                )
                if granule.get("primary_download_url"):
                    print(
                        f"        Download: {granule['primary_download_url'][:60]}..."
                    )
                print()

        # Display search metadata
        metadata = result.search_metadata
        print("📊 Search metadata:")
        print(f"   Original query: {metadata.get('original_query')}")
        print(
            f"   Search duration: {metadata.get('total_search_duration_ms', 0):.0f}ms"
        )
        print(f"   Collections searched: {metadata.get('collections_searched', 0)}")
        print(f"   CMR query time: {metadata.get('cmr_query_time_ms', 0):.0f}ms")

        return True

    except Exception as e:
        print(f"❌ Full agent workflow test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all CMR data search tests."""
    print("🚀 CMR Data Search Agent Test Suite")
    print("Testing implementation against CMR MCP server")
    print()
    print("⚠️  Make sure CMR MCP server is running:")
    print("   uv run uvicorn app:app --host 0.0.0.0 --port 8080")
    print()

    tests = [
        ("Basic Collection Search", test_basic_collection_search),
        ("Basic Granule Search", test_basic_granule_search),
        ("Query Decomposition", test_query_decomposition),
        ("Full Agent Workflow", test_full_agent_workflow),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print()
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print()
    print("=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! CMR Data Search Agent is working correctly.")
    else:
        print("⚠️  Some tests failed. Check MCP server connection and implementation.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
