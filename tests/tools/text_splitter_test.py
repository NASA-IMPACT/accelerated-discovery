import pytest
from langchain_core.documents import Document

from akd.tools.text_splitter import (
    TextSplitterInputSchema,
    TextSplitterOutputSchema,
    TextSplitterTool,
    TextSplitterToolConfig,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_documents():
    """Provides sample documents for testing."""
    long_text = " ".join(["This is sentence " + str(i) + "." for i in range(200)])

    return [
        Document(
            page_content=long_text,
            metadata={"source": "doc1.txt", "url": "www.example1.com"},
        ),
        Document(
            page_content="This is a short document that should not be split.",
            metadata={"source": "doc2.txt", "url": "www.example2.com"},
        ),
    ]


async def test_text_splitter_with_default_config(sample_documents):
    """
    Tests the TextSplitterTool with its default configuration to ensure
    it splits long documents and assigns new IDs.
    """

    splitter_tool = TextSplitterTool()
    input_data = TextSplitterInputSchema(documents=sample_documents)

    # Run the tool
    output = await splitter_tool._arun(input_data)

    assert isinstance(output, TextSplitterOutputSchema)

    # The long document should be split, and the short one should remain as one chunk.
    assert len(output.chunks) > len(sample_documents)

    # Check the properties of each chunk
    for chunk in output.chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) <= splitter_tool.config.chunk_size
        # Verify that a new, unique chunk ID has been added to the metadata
        assert "id" in chunk.metadata
        assert chunk.metadata["id"].startswith(chunk.metadata["source"])


async def test_text_splitter_with_custom_config():
    """
    Tests the TextSplitterTool with a custom configuration (smaller chunk size)
    to verify it produces more chunks.
    """
    small_text = (
        "The cat sat on the mat. Rug is another word for mat. This is a third sentence."
    )
    doc = Document(page_content=small_text, metadata={"source": "custom.txt"})

    # Use a very small chunk size to force splitting
    custom_config = TextSplitterToolConfig(chunk_size=30, chunk_overlap=5)
    splitter_tool = TextSplitterTool(config=custom_config)

    input_data = TextSplitterInputSchema(documents=[doc])

    # Run the tool
    output = await splitter_tool._arun(input_data)

    # Assert the output
    assert isinstance(output, TextSplitterOutputSchema)

    assert len(output.chunks) > 1

    first_chunk = output.chunks[0]
    second_chunk = output.chunks[1]

    # Assert the index of the start index of the second chunk
    overlap_start_index = first_chunk.page_content.find(
        second_chunk.page_content[: custom_config.chunk_overlap],
    )
    assert overlap_start_index != -1, "Chunks should have overlapping content"
