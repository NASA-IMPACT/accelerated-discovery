from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig


class TextSplitterInputSchema(InputSchema):
    """Input schema for the Text Splitter Tool."""

    documents: List[Document] = Field(
        ...,
        description="A list of Langchain Document objects to split.",
    )


class TextSplitterOutputSchema(OutputSchema):
    """Output schema for the Text Splitter Tool."""

    chunks: List[Document] = Field(
        ...,
        description="A list of smaller Langchain Document objects (chunks).",
    )


class TextSplitterToolConfig(BaseToolConfig):
    """Configuration for the TextSplitterTool."""

    chunk_size: int = Field(
        default=1000,
        description="The maximum size of each text chunk.",
    )
    chunk_overlap: int = Field(
        default=100,
        description="The number of characters to overlap between chunks.",
    )


class TextSplitterTool(
    BaseTool[TextSplitterInputSchema, TextSplitterOutputSchema],
):
    """
    A tool for splitting large documents into smaller, more manageable chunks.
    """

    name = "text_splitter_tool"
    description = "Splits a list of documents into smaller text chunks."
    input_schema = TextSplitterInputSchema
    output_schema = TextSplitterOutputSchema
    config_schema = TextSplitterToolConfig

    def __init__(
        self,
        config: Optional[TextSplitterToolConfig] = None,
        debug: bool = False,
    ):
        """Initializes the TextSplitterTool."""
        config = config or self.config_schema()
        super().__init__(config, debug)

        logger.info("Initializing TextSplitterTool...")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    async def _arun(
        self,
        params: TextSplitterInputSchema,
        **kwargs,
    ) -> TextSplitterOutputSchema:
        """
        Splits the provided documents into smaller chunks.
        """
        logger.info(f"Splitting {len(params.documents)} document(s)...")
        all_chunks = []
        for doc in params.documents:
            chunks = self._splitter.split_documents([doc])
            # Add unique IDs to each chunk's metadata
            for i, chunk in enumerate(chunks):
                source_id = chunk.metadata.get(
                    "id",
                    chunk.metadata.get("source", "unknown"),
                )
                chunk.metadata["id"] = f"{source_id}_{i}"
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks.")
        return TextSplitterOutputSchema(chunks=all_chunks)
