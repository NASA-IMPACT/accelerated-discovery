#!/usr/bin/env python3
"""
Simple test script to verify CMR MCP server connection and basic tool functionality.

This script tests the MCP HTTP connection without requiring OpenAI API keys.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the akd package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_mcp_raw_connection():
    """Test raw HTTP connection to MCP server."""
    print("=" * 60)
    print("🧪 TEST: Raw MCP Server Connection")
    print("=" * 60)
    
    try:
        import httpx
        
        mcp_endpoint = "http://localhost:8080/mcp/cmr/mcp/"
        
        request_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_collections",
                "arguments": {
                    "keyword": "MODIS",
                    "platform": "Terra",
                    "page_size": 3
                }
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "AKD-Test/1.0.0"
        }
        
        print(f"🔗 Connecting to: {mcp_endpoint}")
        print(f"📤 Request: {json.dumps(request_data, indent=2)}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(mcp_endpoint, json=request_data, headers=headers)
            
            print(f"📥 Response status: {response.status_code}")
            print(f"📥 Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                content = response.text.strip()
                print(f"📄 Response content preview: {content[:200]}...")
                
                # Try to parse as SSE
                for line in content.split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line.split("data: ", 1)[1])
                        
                        if "error" in data:
                            print(f"❌ MCP Error: {data['error']}")
                            return False
                        
                        if "result" in data and "content" in data["result"]:
                            tool_result = json.loads(data["result"]["content"][0]["text"])
                            print(f"✅ MCP Response received!")
                            print(f"   Total hits: {tool_result.get('total_hits', 'N/A')}")
                            print(f"   Collections: {len(tool_result.get('collections', []))}")
                            return True
                            
                print("❌ Could not parse MCP response")
                return False
            else:
                print(f"❌ HTTP Error: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


async def test_collection_tool_mcp():
    """Test CMR Collection Search Tool using MCP."""
    print("=" * 60)
    print("🧪 TEST: CMR Collection Search Tool (MCP)")
    print("=" * 60)
    
    try:
        from akd.tools.data_search import CMRCollectionSearchTool
        from akd.tools.data_search.cmr_collection_search import CMRCollectionSearchInputSchema
        
        # Create tool with explicit MCP endpoint
        tool = CMRCollectionSearchTool(debug=True)
        
        search_params = CMRCollectionSearchInputSchema(
            keyword="MODIS",
            platform="Terra",
            page_size=3
        )
        
        print("🔎 Testing CMR Collection Search via MCP...")
        result = await tool.arun(search_params)
        
        print(f"✅ Tool succeeded!")
        print(f"   Total hits: {result.total_hits}")
        print(f"   Collections returned: {len(result.collections)}")
        print(f"   Query time: {result.query_time_ms}ms")
        
        if result.collections:
            print("   Sample collections:")
            for i, col in enumerate(result.collections[:2], 1):
                print(f"     {i}. {col.get('short_name', 'N/A')} - {col.get('concept_id', 'N/A')}")
                
        return result.collections[0] if result.collections else None
        
    except Exception as e:
        print(f"❌ Collection tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_granule_tool_mcp(collection_concept_id: str):
    """Test CMR Granule Search Tool using MCP."""
    print("=" * 60)
    print("🧪 TEST: CMR Granule Search Tool (MCP)")
    print("=" * 60)
    
    try:
        from akd.tools.data_search import CMRGranuleSearchTool
        from akd.tools.data_search.cmr_granule_search import CMRGranuleSearchInputSchema
        
        tool = CMRGranuleSearchTool(debug=True)
        
        granule_params = CMRGranuleSearchInputSchema(
            collection_concept_id=collection_concept_id,
            temporal="2024-01-01T00:00:00Z,2024-01-02T23:59:59Z",  # Shorter time range
            online_only=True,
            page_size=2
        )
        
        print(f"🔎 Testing granule search for collection: {collection_concept_id}")
        result = await tool.arun(granule_params)
        
        print(f"✅ Granule search succeeded!")
        print(f"   Total hits: {result.total_hits}")
        print(f"   Granules returned: {len(result.granules)}")
        print(f"   Query time: {result.query_time_ms}ms")
        
        if result.granules:
            print("   Sample granules:")
            for i, granule in enumerate(result.granules[:2], 1):
                print(f"     {i}. {granule.get('granule_ur', 'N/A')[:50]}...")
                print(f"        Online: {'Yes' if granule.get('online_access_flag') else 'No'}")
                
        return True
        
    except Exception as e:
        print(f"❌ Granule tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_query_decomposition():
    """Test query decomposition without requiring OpenAI."""
    print("=" * 60)
    print("🧪 TEST: Simple Query Decomposition (Pattern-based)")
    print("=" * 60)
    
    try:
        from akd.agents.data_search.components.query_decomposition import QueryDecompositionComponent
        
        # Create a mock component that doesn't use QueryAgent
        class SimpleQueryDecomposition:
            def __init__(self):
                self.debug = True
                
            async def process(self, query: str):
                """Simple pattern-based decomposition without LLM."""
                from akd.agents.data_search.components.query_decomposition import DecomposedQuery
                
                query_lower = query.lower()
                
                # Simple pattern matching
                keywords = []
                platforms = []
                instruments = []
                
                if 'modis' in query_lower:
                    keywords.append('modis')
                    instruments.append('MODIS')
                if 'terra' in query_lower:
                    platforms.append('Terra')
                if 'aqua' in query_lower:
                    platforms.append('Aqua')
                if 'temperature' in query_lower or 'sst' in query_lower:
                    keywords.append('temperature')
                if '2024' in query:
                    temporal_start = "2024-01-01T00:00:00Z"
                    temporal_end = "2024-12-31T23:59:59Z"
                else:
                    temporal_start = None
                    temporal_end = None
                    
                # Create basic search variations
                search_variations = []
                if keywords and platforms:
                    search_variations.append({
                        "keyword": " ".join(keywords),
                        "platform": platforms[0] if platforms else None
                    })
                    
                return DecomposedQuery(
                    keywords=keywords,
                    platforms=platforms,
                    instruments=instruments,
                    temporal_start=temporal_start,
                    temporal_end=temporal_end,
                    search_variations=search_variations
                )
        
        component = SimpleQueryDecomposition()
        
        test_query = "Find Terra MODIS sea surface temperature data from 2024"
        print(f"📝 Decomposing: '{test_query}'")
        
        result = await component.process(test_query)
        
        print("✅ Query decomposition succeeded!")
        print(f"   Keywords: {result.keywords}")
        print(f"   Platforms: {result.platforms}")
        print(f"   Instruments: {result.instruments}")
        print(f"   Temporal start: {result.temporal_start}")
        print(f"   Search variations: {len(result.search_variations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query decomposition test failed: {e}")
        return False


async def main():
    """Run simplified CMR MCP tests."""
    print("🚀 CMR MCP Connection Test Suite")
    print("Testing basic MCP server functionality")
    print()
    print("⚠️  Make sure CMR MCP server is running:")
    print("   uv run uvicorn app:app --host 0.0.0.0 --port 8080")
    print()
    
    # Test 1: Raw MCP connection
    print()
    mcp_works = await test_mcp_raw_connection()
    
    if not mcp_works:
        print()
        print("❌ MCP server connection failed. Cannot proceed with other tests.")
        print("   Please ensure the CMR MCP server is running on localhost:8080")
        return
    
    # Test 2: Collection search tool
    print()
    sample_collection = await test_collection_tool_mcp()
    
    # Test 3: Granule search tool (if we have a collection)
    if sample_collection:
        concept_id = sample_collection.get('concept_id')
        if concept_id:
            print()
            await test_granule_tool_mcp(concept_id)
    
    # Test 4: Simple query decomposition
    print()
    await test_simple_query_decomposition()
    
    print()
    print("=" * 60)
    print("🎯 Basic MCP functionality testing completed!")
    print("   If all tests passed, the CMR data search tools are working.")
    print("   To test the full agent, set OPENAI_API_KEY environment variable.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())