import shutil
from pathlib import Path
from typing import Dict, List

import pytest

from akd.tools.vector_db_tool import (
    VectorDBIndexInputSchema,
    VectorDBQueryInputSchema,
    VectorDBQueryOutputSchema,
    VectorDBTool,
    VectorDBToolConfig,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Create a temporary directory for the ChromaDB database."""
    db_path = str(tmp_path / "test_chroma_db")
    yield db_path
    shutil.rmtree(db_path, ignore_errors=True)


@pytest.fixture
def sample_data() -> Dict[str, List]:
    """Provides sample data as a dictionary of lists."""
    return {
        "ids": ["doc1", "doc2", "doc3"],
        "documents": [
            "The sky is blue. It is thirty degrees Celsius.",
            "The ingredients include apples, pears and grapes.",
            "A computer is an electronic device. Keyboards are used for input.",
        ],
        "metadatas": [
            {"source": "weather.txt", "title": "weather_report"},
            {"source": "ingredients.txt", "title": "ingredients_list"},
            {"source": "tech.txt", "url": "www.tech-example.com"},
        ],
    }


@pytest.fixture
def configured_db_tool(
    temp_db_path: str,
    sample_data: Dict[str, List],
) -> VectorDBTool:
    """
    Provides a fully configured and pre-populated VectorDBTool instance.
    """
    config = VectorDBToolConfig(
        db_path=temp_db_path,
        collection_name="test_collection",
    )
    db_tool = VectorDBTool(config=config)

    # Index the sample documents
    index_params = VectorDBIndexInputSchema(**sample_data)
    db_tool.index(index_params)

    return db_tool


def test_vectordb_initialization(temp_db_path: str):
    """
    Tests that the VectorDBTool initializes correctly.
    """
    config = VectorDBToolConfig(db_path=temp_db_path, collection_name="init_test")
    db_tool = VectorDBTool(config=config)

    assert db_tool.client is not None
    assert db_tool.collection is not None
    assert db_tool.collection.name == "init_test"
    assert Path(temp_db_path).exists()


def test_index_method(temp_db_path: str, sample_data: Dict[str, List]):
    """
    Tests that the `index` method correctly adds documents.
    """
    config = VectorDBToolConfig(db_path=temp_db_path, collection_name="index_test")
    db_tool = VectorDBTool(config=config)

    index_params = VectorDBIndexInputSchema(**sample_data)
    db_tool.index(index_params)

    assert db_tool.collection.count() == len(sample_data["documents"])

    retrieved = db_tool.collection.get(ids=["doc2"], include=["metadatas", "documents"])
    assert (
        retrieved["documents"][0] == "The ingredients include apples, pears and grapes."
    )
    assert retrieved["metadatas"][0]["source"] == "ingredients.txt"
    assert retrieved["metadatas"][0]["title"] == "ingredients_list"


async def test_arun_retrieval(configured_db_tool: VectorDBTool):
    """
    Tests that the `_arun` method correctly retrieves relevant documents.
    """
    query = "What color is the sky?"
    input_params = VectorDBQueryInputSchema(query=query, k=1)

    output = await configured_db_tool._arun(input_params)

    assert isinstance(output, VectorDBQueryOutputSchema)
    assert len(output.results) == 1

    retrieved_doc = output.results[0]
    assert isinstance(retrieved_doc, dict)

    assert retrieved_doc["metadata"]["source"] == "weather.txt"
    assert "The sky is blue" in retrieved_doc["page_content"]
