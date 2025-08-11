from typing import List, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain_core.documents import Document
from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig


class VectorDBInputSchema(InputSchema):
    """Input schema for querying documents from the Vector Database."""

    query: str = Field(..., description="The query string for retrieval.")
    k: int = Field(3, description="Number of documents to retrieve.")


class VectorDBOutputSchema(OutputSchema):
    """Output schema for the Vector Database tool's query results."""

    results: List[Document] = Field(
        ...,
        description="List of retrieved Langchain Document objects.",
    )


class VectorDBToolConfig(BaseToolConfig):
    """Configuration for the VectorDBTool, loaded from environment variables."""

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="The name of the Hugging Face embedding model to use.",
    )
    embedding_model_api_key: Optional[str] = Field(
        default=None,
        description="The API key for the embedding model provider, currently using HuggingFace.",
    )
    db_path: str = Field(
        default="./chroma_db",
        description="Path to the persistent ChromaDB directory.",
    )
    collection_name: str = Field(
        default="litagent_demo",
        description="Name of the collection within ChromaDB.",
    )


class VectorDBTool(
    BaseTool[VectorDBInputSchema, VectorDBOutputSchema],
):
    """
    A tool for indexing and retrieving documents from a Chroma vector database.
    """

    name = "vector_db_tool"
    description = (
        "Indexes documents into a vector database and retrieves them based on a query."
    )
    input_schema = VectorDBInputSchema
    output_schema = VectorDBOutputSchema
    config_schema = VectorDBToolConfig

    def __init__(
        self,
        config: VectorDBToolConfig | None = None,
        debug: bool = False,
    ):
        """Initializes the VectorDBTool and its ChromaDB client."""
        config = config or VectorDBToolConfig()
        super().__init__(config, debug)

        logger.info("Initializing VectorDBTool...")

        self.client = chromadb.PersistentClient(path=self.config.db_path)

        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model_name,
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            embedding_function=embedding_function,
        )
        logger.info(
            f"Connected to ChromaDB collection '{self.config.collection_name}'.",
        )

    def index(self, documents: List[Document]):
        """
        Adds or updates documents in the vector database collection from Langchain Documents.
        """
        logger.info(f"Indexing {len(documents)} documents...")

        # Extract components from the Document objects for ChromaDB
        ids = [doc.metadata.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
        contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
        )
        logger.info("Indexing complete.")

    async def _arun(
        self,
        params: VectorDBInputSchema,
    ) -> VectorDBOutputSchema:
        """
        Retrieves documents and returns them as a list of Langchain Document objects.
        """
        logger.info(
            f"Querying collection with query: '{params.query}', retrieving top-{params.k} documents",
        )

        # Include metadatas and documents to reconstruct the Document objects
        results = self.collection.query(
            query_texts=[params.query],
            n_results=params.k,
            include=["metadatas", "documents"],
        )

        retrieved_docs = []
        # The result is batched; we process the first (and only) query's results
        if results and results.get("ids") and results["ids"][0]:
            result_ids = results["ids"][0]
            result_documents = results["documents"][0]
            result_metadatas = results["metadatas"][0]

            for i in range(len(result_ids)):
                # Reconstruct the Langchain Document object
                doc = Document(
                    page_content=result_documents[i],
                    metadata=result_metadatas[i]
                    if result_metadatas and result_metadatas[i]
                    else {},
                )
                retrieved_docs.append(doc)

        return VectorDBOutputSchema(results=retrieved_docs)
