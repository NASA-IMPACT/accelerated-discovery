"""
Test script to debug SearxNG search tool directly.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from akd.tools.search import (
    SearxNGSearchTool,
    SearxNGSearchToolConfig,
    SearxNGSearchToolInputSchema,
)

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


async def test_search_tool():
    """Test the SearxNG search tool directly."""
    
    print("=== Testing SearxNG Search Tool ===\n")
    
    # Create config
    base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
    print(f"Using SearxNG base URL: {base_url}")
    
    config = SearxNGSearchToolConfig(
        base_url=base_url,
        max_results=10,
        debug=True  # Enable debug mode
    )
    
    # Create tool
    tool = SearxNGSearchTool(config=config)
    
    # Test query
    query = "climate change coral reefs"
    print(f"\nSearching for: {query}")
    
    try:
        # Create input
        tool_input = SearxNGSearchToolInputSchema(
            queries=[query],
            category="science",
            max_results=10
        )
        
        # Run search
        result = await tool.arun(tool_input)
        
        print(f"\nResult type: {type(result)}")
        print(f"Has results attribute: {hasattr(result, 'results')}")
        
        if hasattr(result, 'results'):
            print(f"Number of results: {len(result.results)}")
            print("\nFirst 3 results:")
            for i, item in enumerate(result.results[:3]):
                print(f"\n{i+1}. Title: {getattr(item, 'title', 'No title')}")
                print(f"   URL: {getattr(item, 'url', 'No URL')}")
                print(f"   Content: {str(getattr(item, 'content', ''))[:100]}...")
        else:
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


async def test_searxng_connection():
    """Test if SearxNG is accessible."""
    import aiohttp
    
    base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
    print(f"\n=== Testing SearxNG Connection ===")
    print(f"Trying to connect to: {base_url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/") as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    print("SearxNG is accessible!")
                else:
                    print("SearxNG returned non-200 status")
                    text = await response.text()
                    print(f"Response: {text[:200]}...")
    except Exception as e:
        print(f"Connection failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # First test connection
    asyncio.run(test_searxng_connection())
    
    # Then test search
    asyncio.run(test_search_tool())