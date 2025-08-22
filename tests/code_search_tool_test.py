import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import requests

# Add the parent directory (the project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from akd.tools.code_search import (
    CodeSearchToolInputSchema,
    GitHubCodeSearchTool,
    LocalRepoCodeSearchTool,
    LocalRepoCodeSearchToolConfig,
    SDECodeSearchTool,
    SDECodeSearchToolConfig,
)
from akd.tools.misc import Embedder
from akd.tools.search import SearxNGSearchToolConfig

"""Validate the output structure"""


def validate_output_structure(output):
    assert hasattr(output, "results")
    assert isinstance(output.results, list)
    assert len(output.results) > 0

    for result in output.results:
        assert hasattr(result, "url")
        assert hasattr(result, "content")
        assert result.content and result.content.strip()


"""Initialize the tools"""


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


"""Test1: Google Drive Link"""


def test_google_drive_link():
    config = LocalRepoCodeSearchToolConfig()
    file_id = config.google_drive_file_id
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.head(url, allow_redirects=True)
    assert response.status_code == 200


"""Test2: Data file validation"""


def test_data_file_validation():
    config = LocalRepoCodeSearchToolConfig()
    df = pd.read_csv(config.data_file)
    assert df is not None
    assert not df.empty
    assert "embeddings" in df.columns


"""Test3: Vector Embedding"""


def test_vector_embedding(embedder=Embedder(model_name="all-MiniLM-L6-v2")):
    texts = ["flood prediction", "earthquake classification"]
    embeddings = embedder.embed_texts(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == embedder.get_embedding_dimensions()
    assert not np.isnan(embeddings).any()


"""Test4: Local Repo Search"""


@pytest.mark.asyncio
async def test_local_repo_search(local_tool):
    input_params = CodeSearchToolInputSchema(
        queries=["landslide nepal"],
        max_results=3,
    )
    # Input structure validation
    assert input_params.queries == ["landslide nepal"]
    assert input_params.max_results == 3

    # Output structure validation
    output = await local_tool._arun(input_params)
    validate_output_structure(output)


"""Test5: SearxNG server"""


@pytest.mark.asyncio
async def test_searxng_server():
    url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
    response = requests.head(url)
    assert response.status_code == 200
    assert response.headers.get("Content-Type") == "text/html; charset=utf-8"


"""Test6: GitHub Search"""


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


"""Test7: SDE API"""


@pytest.mark.asyncio
async def test_sde_api():
    url = "https://d2kqty7z3q8ugg.cloudfront.net/api/code/search"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "page": 0,
        "pageSize": 1,
        "search_term": "test",
        "search_type": "keyword",
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data


"""Test8: SDE Search"""


@pytest.mark.asyncio
async def test_sde_code_search(sde_tool):
    input_params = CodeSearchToolInputSchema(
        queries=["weather prediction"], max_results=5, search_mode="keyword"
    )
    # Input structure validation
    assert input_params.queries == ["weather prediction"]
    assert input_params.max_results == 5

    # Output structure validation
    output = await sde_tool._arun(input_params)
    validate_output_structure(output)
