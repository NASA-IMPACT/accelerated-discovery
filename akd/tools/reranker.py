from __future__ import annotations

from abc import abstractmethod
from pydantic.fields import Field

from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig
from loguru import logger
from sentence_transformers import CrossEncoder
import numpy as np


class RerankerToolConfig(BaseToolConfig):
    """
    Base configuration for reranker tools.
    This can be extended by specific reranker tool configurations.
    """

    deduplication: bool = Field(default=True, description="Whether to use deduplication of results.")
    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L12-v2", description="The name of the reranker model to use."
    )
    deduplication_key: str = Field(default="url", description="The key to use for deduplication of results.")
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

    def _deduplicate_results(
        self,
        results: list[SearchResultItem],
        deduplication_key: str,
    ) -> list[SearchResultItem]:
        """
        Deduplicate results based on a unique key (default is URL).
        """
        seen = set()
        deduped = []
        for result in results:
            val = str(getattr(result, deduplication_key, ""))
            if val and val not in seen:
                seen.add(val)
                deduped.append(result)
        return deduped

    def _sort_results(
        self,
        results: list[SearchResultItem],
        sort_key: str,
    ) -> list[SearchResultItem]:
        """
        Sort results by the specified key. First checks for the key directly in the dict,
        then checks in the 'extra' field if it exists. Returns unsorted if key not found.
        """

        def __get_sort_key(result):
            # First check if sort_by key exists directly in the dict
            if sort_key in result:
                return result[sort_key]

            # Then check if 'extra' field exists and contains the sort_by key
            if result.extra and isinstance(result.extra, dict) and sort_key in result.extra:
                return result.extra[sort_key]

            # If key not found anywhere, return a default value that will sort last
            # Using float('inf') for numerical sorting or empty string for string sorting
            if self.debug:
                logger.warning(f"Sort key {sort_key} not found in results")
            return float("-inf")

        try:
            # Sort in descending order (highest score first)
            # Change reverse=False if you want ascending order
            return sorted(results, key=__get_sort_key, reverse=True)
        except TypeError:
            # If sorting fails (mixed types), return as is
            return results

    # abstract method to be implemented by the subclass
    @abstractmethod
    def _rerank_results(self, query: str, results: list[SearchResultItem]) -> list[SearchResultItem]:
        raise NotImplementedError("Subclass must implement this method")

    def _arun(self, params: RerankerToolInputSchema) -> RerankerToolOutputSchema:
        # rerank results
        ranked_results = self._rerank_results(params.query, params.results)

        # deduplicate results
        if self.config.deduplication:
            ranked_results = self._deduplicate_results(ranked_results, deduplication_key=self.config.deduplication_key)

        return RerankerToolOutputSchema(query=params.query, results=ranked_results)


class CrossEncoderRerankerTool(RerankerTool):
    """
    Tool for performing reranking of search results using a cross-encoder model.
    """

    def __init__(self, config: RerankerToolConfig | None = None, debug: bool = False):
        super().__init__(config=config, debug=debug)
        self.reranker_model = CrossEncoder(self.config.model_name)
        self.debug = debug

    def _rerank_results(self, query: str, results: list[SearchResultItem]) -> list[SearchResultItem]:
        # create pairs of query and results
        pairs = [(query, result.content) for result in results]

        # get similarity scores from CrossEncoder
        scores = self.reranker_model.predict(pairs)
        scores = 1 / (1 + np.exp(-scores))

        # attach scores
        for score, result in zip(scores, results):
            result.extra["score"] = score

        # sort results
        return self._sort_results(results, sort_key=self.config.sort_key)
