#!/usr/bin/env python
"""
Simple script to test CMR queries with MCP parameters.

Usage:
1. Paste your mcp_parameters_sent dict below (replace the PARAMS variable)
2. Run: uv run test_cmr_query.py
"""

import asyncio
import json
import warnings

from akd.tools.data_search.cmr_collection_search import CMRCollectionSearchTool

# Suppress Pydantic HttpUrl serialization warning
warnings.filterwarnings("ignore", message=".*HttpUrl.*", category=UserWarning)

# PASTE YOUR MCP PARAMETERS HERE (copy directly from mcp_parameters_sent)
PARAMS = {
    "bounding_box": "-124.48,32.53,-114.13,42.01",
    "keyword": "precipitation",
    "page_size": 20,
}


async def test_query():
    """Execute CMR query and print results."""
    print("🔍 Testing CMR query with parameters:")
    print(json.dumps(PARAMS, indent=2))
    print("\n" + "=" * 60 + "\n")

    # Create search tool
    tool = CMRCollectionSearchTool()

    try:
        # Execute search
        result = await tool.arun(PARAMS)

        # Parse results
        collections = result.collections
        total = len(collections)

        print(f"✅ Found {total} collections\n")

        # Print first 5 collection titles
        for i, collection in enumerate(collections[:5], 1):
            concept_id = collection.get("concept_id", "No ID")
            short_name = collection.get("short_name", "No short name")
            entry_title = collection.get("entry_title", "No title")

            print(f"{i}. {short_name} ({concept_id})")
            print(f"   {entry_title}")

            # Show summary if available
            if summary := collection.get("summary"):
                summary_preview = summary[:450] + "..." if len(summary) > 450 else summary
                print(f"   Summary: {summary_preview}")
            print()

        if total > 5:
            print(f"... and {total - 5} more collections")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_query())
