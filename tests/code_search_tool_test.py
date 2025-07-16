import sys
import os
import pytest
import requests

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


def validate_output_structure(output):
    assert hasattr(output, "results")
    assert isinstance(output.results, list)
    assert len(output.results) > 0

    for result in output.results:
        assert hasattr(result, "url")
        assert hasattr(result, "content")
        assert result.content and result.content.strip()


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
        max_results=3,
    )
    # Input structure validation
    assert input_params.queries == ["landslide nepal"]
    assert input_params.max_results == 3

    # Output structure validation
    output = await local_tool._arun(input_params)
    validate_output_structure(output)


@pytest.mark.asyncio
async def test_github_code_search(github_tool):
    input_params = CodeSearchToolInputSchema(
        queries=["flood detection"],
        max_results=10,
    )
    # Input structure validation
    assert input_params.queries == ["flood detection"]
    assert input_params.max_results == 10

    # Output structure validation
    output = await github_tool._arun(input_params)
    validate_output_structure(output)


@pytest.mark.asyncio
async def test_sde_code_search(sde_tool):
    input_params = CodeSearchToolInputSchema(
        queries=["weather prediction"],
        max_results=5,
    )
    # Input structure validation
    assert input_params.queries == ["weather prediction"]
    assert input_params.max_results == 5

    # Output structure validation
    output = await sde_tool._arun(input_params)
    validate_output_structure(output)


def test_google_drive_link():
    config = LocalRepoCodeSearchToolConfig()
    file_id = config.google_drive_file_id
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.head(url, allow_redirects=True)
    assert response.status_code == 200






