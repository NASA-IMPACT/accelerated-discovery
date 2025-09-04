import os
from typing import Any, Dict, List, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig
from akd.utils import get_akd_root


class VectorDBIndexInputSchema(InputSchema):
    """Input schema for indexing documents into the Vector Database."""

    ids: List[str] = Field(..., description="A unique list of document IDs.")
    documents: List[str] = Field(
        ...,
        description="A list of document contents to index.",
    )
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Optional list of metadata for each document.",
    )


class VectorDBQueryInputSchema(InputSchema):
    """Input schema for querying documents from the Vector Database."""

    query: str = Field(..., description="The query string for retrieval.")
    k: int = Field(3, description="Number of documents to retrieve.")


class VectorDBQueryOutputSchema(OutputSchema):
    """Output schema for the Vector Database tool's query results."""

    results: List[Dict[str, Any]] = Field(
        ...,
        description="List of retrieved documents, each as a dictionary with 'page_content' and 'metadata'.",
    )


class VectorDBToolConfig(BaseToolConfig):
    """Configuration for the VectorDBTool, loaded from environment variables."""

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="The name of the Hugging Face embedding model to use.",
    )
    embedding_model_api_key: Optional[str] = Field(
        default=os.getenv("EMBEDDING_MODEL_API_KEY", None),
        description="The API key for the embedding model provider, if required.",
    )
    db_path: str = Field(
        default=os.getenv("VECTOR_DB_PATH", str(get_akd_root() / "chroma_db")),
        description="Path to the persistent ChromaDB directory.",
    )
    collection_name: str = Field(
        default="akd_vdb",
        description="Name of the collection within ChromaDB.",
    )


class VectorDBTool(
    BaseTool[VectorDBQueryInputSchema, VectorDBQueryOutputSchema],
):
    """
    A tool for indexing and retrieving documents from a Chroma vector database.
    """

    name = "vector_db_tool"
    description = (
        "Indexes documents into a vector database and retrieves them based on a query."
    )
    input_schema = VectorDBQueryInputSchema
    output_schema = VectorDBQueryOutputSchema
    config_schema = VectorDBToolConfig

    def __init__(
        self,
        config: Optional[VectorDBToolConfig] = None,
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

    def index(self, params: VectorDBIndexInputSchema):
        """
        Adds or updates documents in the vector database collection.
        """
        logger.info(f"Indexing {len(params.documents)} documents...")
        self.collection.add(
            ids=params.ids,
            documents=params.documents,
            metadatas=params.metadatas,
        )
        logger.info("Indexing complete.")

    async def _arun(
        self,
        params: VectorDBQueryInputSchema,
    ) -> VectorDBQueryOutputSchema:
        """
        Retrieves documents and returns them as a list of dictionaries.
        """
        logger.info(
            f"Querying collection with query: '{params.query}', retrieving top-{params.k} documents",
        )

        results = self.collection.query(
            query_texts=[params.query],
            n_results=params.k,
            include=["metadatas", "documents"],
        )

        retrieved_docs = []
        if results and results.get("ids") and results["ids"][0]:
            result_documents = results["documents"][0]
            result_metadatas = results["metadatas"][0]

            for i in range(len(result_documents)):
                doc = {
                    "page_content": result_documents[i],
                    "metadata": result_metadatas[i]
                    if result_metadatas and result_metadatas[i]
                    else {},
                }
                retrieved_docs.append(doc)

        return VectorDBQueryOutputSchema(results=retrieved_docs)
