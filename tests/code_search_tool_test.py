import sys
import os
import pytest

# Add the parent directory (the project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from akd.tools.search import SearxNGSearchToolConfig, SearxNGSearchToolInputSchema
from akd.tools.code_search import (
    CodeSearchToolInputSchema,
    LocalRepoCodeSearchTool, 
    LocalRepoCodeSearchToolConfig, 
    LocalRepoCodeSearchToolInputSchema, 
    GitHubCodeSearchTool,
    SDECodeSearchTool, 
    SDECodeSearchToolConfig
)


# Initialize the tools and run the tests
@pytest.fixture
def local_tool():
    config = LocalRepoCodeSearchToolConfig(debug=True)
    return LocalRepoCodeSearchTool(config=config)


@pytest.fixture
def github_tool():
    config = SearxNGSearchToolConfig(score_cutoff=0.1)
    return GitHubCodeSearchTool(config=config)


@pytest.fixture
def sde_tool():
    config = SDECodeSearchToolConfig(debug=True)
    return SDECodeSearchTool(config=config)


@pytest.mark.asyncio
async def test_local_repo_search(local_tool):
    input_params = LocalRepoCodeSearchToolInputSchema(
        queries=["landslide nepal"],
        top_k=3,
    )
    output = await local_tool._arun(input_params)
    
    assert output is not None
    assert len(output.results) > 0
    for result in output.results:
        assert "http" in str(result.url)
        assert result.content is not None
        assert result.content.strip() != ""


@pytest.mark.asyncio
async def test_github_code_search(github_tool):
    input_params = CodeSearchToolInputSchema(
        queries=["flood detection"],
        max_results=5,
    )
    output = await github_tool._arun(input_params)

    assert output is not None
    assert len(output.results) > 0
    for result in output.results:
        assert "http" in str(result.url)
        assert result.content is not None
        assert result.content.strip() != ""


@pytest.mark.asyncio
async def test_sde_code_search(sde_tool):
    input_params = CodeSearchToolInputSchema(
        queries=["weather prediction"],
        max_results=5,
    )
    output = await sde_tool._arun(input_params)

    assert output is not None
    assert len(output.results) > 0
    for result in output.results:
        assert "http" in str(result.url)
        assert result.content is not None
        assert result.content.strip() != ""








