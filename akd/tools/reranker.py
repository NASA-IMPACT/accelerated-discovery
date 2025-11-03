from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
from pydantic.fields import Field
from sentence_transformers import CrossEncoder

from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig
from akd.tools.search.utils import deduplicate_results, sort_results

# Reranker type options for factory function
RerankerType = Literal["cross_encoder", "identity", "no_op", "nope", "none"]


class RerankerToolConfig(BaseToolConfig):
    """
    Base configuration for reranker tools.
    This can be extended by specific reranker tool configurations.
    """

    deduplication: bool = Field(default=True, description="Whether to use deduplication of results.")
    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L12-v2",
        description="The name of the reranker model to use.",
    )
    deduplication_keys: list[str] = Field(default=["url"], description="The keys to use for deduplication of results.")
    sort_key: str = Field(default="score", description="The key to use for sorting of results.")


class RerankerToolInputSchema(InputSchema):
    """
    Schema for input to a tool for reranking search results.
    """

    query: str = Field(..., description="Reranking query.")
    results: list[SearchResultItem] = Field(..., description="List of search results to rerank.")


class RerankerToolOutputSchema(OutputSchema):
    """Schema for output of a tool for reranking search results."""

    query: str = Field(..., description="Reranking query.")
    results: list[SearchResultItem] = Field(..., description="List of reranked search results.")


class RerankerTool(BaseTool[RerankerToolInputSchema, RerankerToolOutputSchema]):
    """
    Tool for performing reranking of search results based on the provided queries.

    Attributes:
        input_schema (RerankerToolInputSchema): The schema for the input data.
        output_schema (RerankerToolOutputSchema): The schema for the output data.
    """

    input_schema = RerankerToolInputSchema
    output_schema = RerankerToolOutputSchema
    config_schema = RerankerToolConfig

    async def _deduplicate_results(
        self,
        results: list[SearchResultItem],
        deduplication_keys: list[str],
    ) -> list[SearchResultItem]:
        """
        Deduplicate results based on a list of keys.
        """
        deduped = deduplicate_results(
            results,
            keys=deduplication_keys,
            debug=self.debug,
        )
        return deduped[0]

    async def _sort_results(
        self,
        results: list[SearchResultItem],
        sort_key: str,
    ) -> list[SearchResultItem]:
        """
        Sort results by the specified key. First checks for the key directly in the dict,
        then checks in the 'extra' field if it exists. Returns unsorted if key not found.
        """
        return sort_results(
            results,
            sort_by=sort_key,
            debug=self.debug,
        )

    # abstract method to be implemented by the subclass
    @abstractmethod
    async def _rerank_results(self, query: str, results: list[SearchResultItem]) -> list[SearchResultItem]:
        raise NotImplementedError("Subclass must implement this method")

    async def _arun(self, params: RerankerToolInputSchema) -> RerankerToolOutputSchema:
        if not params.results:
            return RerankerToolOutputSchema(query=params.query, results=[])
        # rerank results
        ranked_results = await self._rerank_results(params.query, params.results)

        # deduplicate results
        if self.config.deduplication:
            ranked_results = await self._deduplicate_results(
                ranked_results,
                deduplication_keys=self.config.deduplication_keys,
            )

        return RerankerToolOutputSchema(query=params.query, results=ranked_results)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} | (model_name={self.config.model_name}, deduplication={self.config.deduplication}, sort_key={self.config.sort_key})"  # type: ignore

    def __repr__(self) -> str:
        return str(self)


class CrossEncoderRerankerTool(RerankerTool):
    """
    Tool for performing reranking of search results using a cross-encoder model.
    """

    def __init__(self, config: RerankerToolConfig | None = None, debug: bool = False):
        super().__init__(config=config, debug=debug)
        self.reranker_model = CrossEncoder(self.config.model_name)
        self.debug = debug

    async def _rerank_results(self, query: str, results: list[SearchResultItem]) -> list[SearchResultItem]:
        # create pairs of query and results
        pairs = [(query, result.content) for result in results]

        # get similarity scores from CrossEncoder
        scores = self.reranker_model.predict(pairs)
        scores = 1 / (1 + np.exp(-scores))

        # attach scores
        for score, result in zip(scores, results):
            score = float(score)
            result.score = score
            result.extra["score"] = score

        # sort results
        return await self._sort_results(results, sort_key=self.config.sort_key)


class NoOpRerankerTool(RerankerTool):
    async def _rerank_results(self, query: str, results: list[SearchResultItem]) -> list[SearchResultItem]:
        return results

    def __str__(self) -> str:
        return self.__class__.__name__


def create_reranker(
    reranker_type: RerankerType,
    config: RerankerToolConfig | None = None,
    debug: bool = False,
) -> RerankerTool:
    """
    Factory function to create reranker instances by type.

    This function provides a clean way to instantiate different reranker
    implementations without hardcoded if/elif chains. New reranker types
    can be added by implementing the RerankerTool class and adding a
    branch here.

    Args:
        reranker_type: Type of reranker to create. Options:
            - "cross_encoder": CrossEncoderRerankerTool using cross-encoder models
            - "identity": NoOpRerankerTool (pass-through, returns results unchanged)
            - "no_op": NoOpRerankerTool (pass-through, returns results unchanged)
            - "nope": NoOpRerankerTool (pass-through, returns results unchanged)
            - "none": NoOpRerankerTool (pass-through, returns results unchanged)
        config: Optional reranker configuration. If None, uses default config.
        debug: Enable debug mode for logging.

    Returns:
        RerankerTool instance (never None - uses NoOpRerankerTool as default)

    Raises:
        ValueError: If reranker_type is not recognized.

    Example:
        >>> # Create cross-encoder reranker
        >>> reranker = create_reranker("cross_encoder")
        >>>
        >>> # Create with custom config
        >>> config = RerankerToolConfig(model_name="custom-model")
        >>> reranker = create_reranker("cross_encoder", config=config)
        >>>
        >>> # No reranking - returns NoOpRerankerTool
        >>> reranker = create_reranker("none")
        >>> reranker = create_reranker("nope")
        >>>
        >>> # Identity/pass-through (for testing)
        >>> reranker = create_reranker("identity")
    """
    # Cross-encoder reranking
    if reranker_type == "cross_encoder":
        return CrossEncoderRerankerTool(config=config, debug=debug)

    # No-op/identity reranking - pass-through that returns original results
    if reranker_type in ("identity", "no_op", "nope", "none"):
        return NoOpRerankerTool(config=config, debug=debug)

    # Unknown type
    raise ValueError(
        f"Unknown reranker type: '{reranker_type}'. Supported types: cross_encoder, identity, no_op, none, nope",
    )


# Export public API
__all__ = [
    # Type definitions
    "RerankerType",
    # Config and schemas
    "RerankerToolConfig",
    "RerankerToolInputSchema",
    "RerankerToolOutputSchema",
    # Base and implementations
    "RerankerTool",
    "CrossEncoderRerankerTool",
    "NoOpRerankerTool",
    # Factory
    "create_reranker",
]
