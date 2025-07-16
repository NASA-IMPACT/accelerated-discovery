import sys
import os

# Add the parent directory (the project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from akd.tools.search import SearxNGSearchToolConfig
from akd.tools.code_search import (
    CodeSearchToolInputSchema,
    LocalRepoCodeSearchTool, 
    LocalRepoCodeSearchToolConfig, 
    LocalRepoCodeSearchToolInputSchema, 
    GitHubCodeSearchTool,
    SDECodeSearchTool, 
    SDECodeSearchToolConfig
)


# Code Search Tool
async def local_repo_search_test():
    """An async function to run the tool."""

    print("Initializing the tool...")
    cfg = LocalRepoCodeSearchToolConfig()
    tool = LocalRepoCodeSearchTool(config=cfg)

    search_input = LocalRepoCodeSearchToolInputSchema(
        queries=["landslide nepal"], 
        max_results=5
    )

    print("Running the search...")
    output = await tool._arun(search_input)

    print("\n--- Search Results ---")
    for result in output.results:
        print(result.url)
        print(result.content)
        print("-"*100)


# GitHub Search Tool
async def github_search_test():
    """An async function to run the tool."""

    print("Initializing the tool...")
    cfg = SearxNGSearchToolConfig(score_cutoff=0.1)
    tool = GitHubCodeSearchTool(config=cfg)

    search_input = CodeSearchToolInputSchema(
        queries=["landslide nepal"], 
        max_results=10
    )

    print("Running the search...")
    output = await tool._arun(search_input)

    print("\n--- Search Results ---")
    for result in output.results:
        print(result.url)
        print(result.content)
        print("-"*100)
 

async def sde_search_test():
    """An async function to run the tool."""

    print("Initializing the tool...")
    cfg = SDECodeSearchToolConfig()
    tool = SDECodeSearchTool(config=cfg)

    search_input = CodeSearchToolInputSchema(
        queries=["landslide nepal"], 
        max_results=5
    )

    print("Running the search...")
    output = await tool._arun(search_input)

    print("\n--- Search Results ---")
    for result in output.results:
        print(result.url)
        print(result.content)
        print("-"*100)


if __name__ == "__main__":
    print("Running local repo search test...")
    asyncio.run(local_repo_search_test())
    print("Running GitHub search test...")
    asyncio.run(github_search_test())
    print("Running SDE search test...")
    asyncio.run(sde_search_test())










