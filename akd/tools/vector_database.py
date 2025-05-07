from __future__ import annotations

import asyncio
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from loguru import logger
from pydantic import Field

from ._base import BaseIOSchema, BaseTool, BaseToolConfig


class VectorDBToolConfig(BaseToolConfig):
    embedding_model: str = os.getenv(
        "EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
    )
    milvus_uri: str = os.getenv("MILVUS_URI", "./milvus/litagent.db")
    collection_name: str = os.getenv("MILVUS_COLLECTION", "litagent_demo")
    drop_old: bool = False
    k: int = 3


class VectorDBIndexInputSchema(BaseIOSchema):
    """Schema for input of a tool for indexing documents into a vector database."""

    documents: List[Document] = Field(
        ..., description="List of LangChain Document objects to index."
    )


class VectorDBQueryInputSchema(BaseIOSchema):
    """Schema for input of a tool for querying a vector database."""

    query: str = Field(..., description="The query string for retrieval.")
    k: int = Field(3, description="Number of top documents to retrieve.")


class VectorDBQueryOutputSchema(BaseIOSchema):
    """Schema for output of a tool for querying a vector database."""

    results: List[str] = Field(..., description="Retrieved document contents.")


class VectorDBSearchTool(BaseTool):
    input_schema = VectorDBQueryInputSchema
    output_schema = VectorDBQueryOutputSchema

    def __init__(
        self, config: Optional[VectorDBToolConfig] = None, debug: bool = False
    ):
        config = config or VectorDBToolConfig()
        super().__init__(config, debug)
        self.embedding: Embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model
        )
        self.vectorstore: Milvus = Milvus(
            embedding_function=self.embedding,
            collection_name=config.collection_name,
            connection_args={"uri": config.milvus_uri},
        )

    @classmethod
    def from_params(
        cls,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        milvus_uri: str = "milvus://localhost:19530",
        collection_name: str = "docling_demo",
        drop_old: bool = False,
        k: int = 3,
        debug: bool = False,
    ) -> VectorDBSearchTool:
        config = VectorDBToolConfig(
            embedding_model=embedding_model,
            milvus_uri=milvus_uri,
            collection_name=collection_name,
            drop_old=drop_old,
            k=k,
        )
        return cls(config, debug)

    async def arun_index(self, params: VectorDBIndexInputSchema) -> str:
        if self.config.drop_old:
            if self.debug:
                logger.debug("Dropping old index...")

            self.vectorstore = Milvus.from_documents(
                documents=params.documents,
                embedding=self.embedding,
                collection_name=self.config.collection_name,
                connection_args={"uri": self.config.milvus_uri},
                index_params={"index_type": "FLAT"},
                drop_old=self.config.drop_old, # set to True if seeking to drop the collection with that name if it exists
            )
        else:
            await asyncio.to_thread(self.vectorstore.add_documents, params.documents)

        return f"Indexed {len(params.documents)} documents into collection '{self.config.collection_name}'."

    async def arun(self, params: VectorDBQueryInputSchema) -> VectorDBQueryOutputSchema:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": params.k})
        docs = await asyncio.to_thread(retriever.get_relevant_documents, params.query)

        if self.debug:
            logger.debug(f"Retrieved {len(docs)} docs for query: {params.query}")

        return VectorDBQueryOutputSchema(results=[doc.page_content for doc in docs])
