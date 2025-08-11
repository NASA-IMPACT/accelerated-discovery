import shutil
from pathlib import Path

import pytest
from langchain_core.documents import Document

from akd.tools.vector_db_tool import (
    VectorDBInputSchema,
    VectorDBOutputSchema,
    VectorDBTool,
    VectorDBToolConfig,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Create a temporary directory for the ChromaDB database."""
    db_path = str(tmp_path / "test_chroma_db")
    yield db_path
    # Clean up the database directory after the test runs
    shutil.rmtree(db_path, ignore_errors=True)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provides a list of sample documents for indexing."""
    return [
        Document(
            page_content="The sky is blue. It is a beautiful day.",
            metadata={"id": "doc1", "source": "weather.txt"},
        ),
        Document(
            page_content="Apples are a type of fruit. They are often red or green.",
            metadata={"id": "doc2", "source": "fruits.txt"},
        ),
        Document(
            page_content="A computer is an electronic device. Keyboards are used for input.",
            metadata={"id": "doc3", "source": "tech.txt"},
        ),
    ]


@pytest.fixture
def configured_db_tool(
    temp_db_path: str,
    sample_documents: list[Document],
) -> VectorDBTool:
    """
    Provides a fully configured and pre-populated VectorDBTool instance
    for testing retrieval.
    """
    # Use a unique collection name for each test run to ensure isolation
    config = VectorDBToolConfig(
        db_path=temp_db_path,
        collection_name="test_collection",
    )
    db_tool = VectorDBTool(config=config)

    # Index the sample documents into the fresh database
    db_tool.index(sample_documents)

    return db_tool


def test_vectordb_initialization(temp_db_path: str):
    """
    Tests that the VectorDBTool initializes correctly and creates the
    database directory and collection.
    """
    config = VectorDBToolConfig(db_path=temp_db_path, collection_name="init_test")
    db_tool = VectorDBTool(config=config)

    # Assert that the client and collection were created
    assert db_tool.client is not None
    assert db_tool.collection is not None
    assert db_tool.collection.name == "init_test"

    # Assert that the database directory was created on disk
    assert Path(temp_db_path).exists()


def test_index_method(temp_db_path: str, sample_documents: list[Document]):
    """
    Tests that the `index` method correctly adds documents to the collection.
    """
    config = VectorDBToolConfig(db_path=temp_db_path, collection_name="index_test")
    db_tool = VectorDBTool(config=config)

    # Index the documents
    db_tool.index(sample_documents)

    # Verify the documents were added to the collection
    assert db_tool.collection.count() == len(sample_documents)

    # Retrieve one document by ID to confirm its content
    retrieved = db_tool.collection.get(ids=["doc2"], include=["metadatas", "documents"])
    assert (
        retrieved["documents"][0]
        == "Apples are a type of fruit. They are often red or green."
    )
    assert retrieved["metadatas"][0]["source"] == "fruits.txt"


async def test_arun_retrieval(configured_db_tool: VectorDBTool):
    """
    Tests that the `_arun` method correctly retrieves the most relevant
    documents for a given query.
    """

    query = "What color is the sky?"
    input_params = VectorDBInputSchema(query=query, k=1)

    output = await configured_db_tool._arun(input_params)

    assert isinstance(output, VectorDBOutputSchema)
    assert len(output.results) == 1

    retrieved_doc = output.results[0]
    assert isinstance(retrieved_doc, Document)

    # Check that the most relevant document was returned
    assert retrieved_doc.metadata["source"] == "weather.txt"
    assert "The sky is blue" in retrieved_doc.page_content
