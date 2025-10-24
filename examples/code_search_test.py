import os
import sys

# Add the parent directory (the project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from akd.agents.search import (
    CodeSearchAgent,
    CodeSearchAgentConfig,
    LitSearchAgentInputSchema,
)
from akd.tools.search import SearxNGSearchToolConfig
from akd.tools.search.code_search import (
    CodeSearchToolInputSchema,
    CompositeCodeSearchTool,
    CompositeCodeSearchToolConfig,
    GitHubCodeSearchTool,
    LocalRepoCodeSearchTool,
    LocalRepoCodeSearchToolConfig,
    SDECodeSearchTool,
    SDECodeSearchToolConfig,
)


# Local Code Search Tool
async def local_repo_search_test():
    """An async function to run the tool."""

    print("Initializing the tool...")
    cfg = LocalRepoCodeSearchToolConfig()
    tool = LocalRepoCodeSearchTool(config=cfg)

    search_input = CodeSearchToolInputSchema(queries=["landslide nepal"], max_results=5)

    print("Running the search...")
    output = await tool._arun(search_input)

    print("\n--- Search Results ---")
    for result in output.results:
        print(result.url)
        print(result.content)
        print("-" * 100)


# GitHub Search Tool
async def github_search_test():
    """An async function to run the tool."""

    print("Initializing the tool...")
    cfg = SearxNGSearchToolConfig(score_cutoff=0.1)
    tool = GitHubCodeSearchTool(config=cfg)

    search_input = CodeSearchToolInputSchema(
        queries=["landslide nepal"],
        max_results=10,
    )

    print("Running the search...")
    output = await tool._arun(search_input)

    print("\n--- Search Results ---")
    for result in output.results:
        print(result.url)
        print(result.content)
        print("-" * 100)


# SDE Code Search Tool
async def sde_search_test():
    """An async function to run the tool."""

    print("Initializing the tool...")
    cfg = SDECodeSearchToolConfig()
    tool = SDECodeSearchTool(config=cfg)

    search_input = CodeSearchToolInputSchema(
        queries=["Weather Prediction"],
        max_results=5,
    )

    print("Running the search...")
    output = await tool._arun(search_input)

    print("\n--- Search Results ---")
    for result in output.results:
        print(result.url)
        print(result.content)
        print("-" * 100)


# Combined Code Search Tool
async def combined_code_search_test():
    """An async function to run the tool."""

    print("Initializing the tool...")
    cfg = CompositeCodeSearchToolConfig()
    tool = CompositeCodeSearchTool(config=cfg)

    search_input = CodeSearchToolInputSchema(
        queries=["landslide nepal"],
        max_results=10,
    )

    print("Running the search...")
    output = await tool._arun(search_input)

    print("\n--- Search Results ---")
    for result in output.results:
        print(result.url)
        print(result.content)
        print(result.extra["tool"])
        print(result.extra["score"])
        print("-" * 100)


# Code Search Agent
async def code_search_agent_test():
    """An async function to run the agent."""

    print("Initializing the code search agent...")
    cfg = CodeSearchAgentConfig()
    agent = CodeSearchAgent(config=cfg)

    search_input = LitSearchAgentInputSchema(query="landslide nepal", max_results=10)

    print("Running the search...")
    output = await agent.arun(search_input)

    print("\n--- Search Results ---")
    for result in output.results:
        print(result.url)
        print(result.title)
        print(result.content[:100])
        print("-" * 100)


if __name__ == "__main__":
    print("Running local repo search test...")
    asyncio.run(local_repo_search_test())
    print("Running GitHub search test...")
    asyncio.run(github_search_test())
    print("Running SDE search test...")
    asyncio.run(sde_search_test())
    print("Running combined code search test...")
    asyncio.run(combined_code_search_test())
    print("Running code search agent test...")
    asyncio.run(code_search_agent_test())
