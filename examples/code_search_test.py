import sys
import os

# Add the parent directory (the project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from akd.tools.code_search import CodeSearchTool, CodeSearchToolConfig, CodeSearchToolInputSchema

async def main():
    """An async function to run the tool."""
    print("Initializing the tool...")
    cfg = CodeSearchToolConfig()
    tool = CodeSearchTool(config=cfg)

    search_input = CodeSearchToolInputSchema(
        queries=["landslide nepal"], 
        top_k=5
    )

    print("Running the search...")
    output = await tool._arun(search_input)

    print("\n--- Search Results ---")
    for result in output.results:
        print(result.url)
        print(result.content)
        print("-"*100)

if __name__ == "__main__":
    asyncio.run(main())










