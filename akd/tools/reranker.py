from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, Literal

import numpy as np
from loguru import logger
from pydantic import AnyUrl, BaseModel
from pydantic.fields import Field
from sentence_transformers import CrossEncoder

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig
from akd.tools.search.utils import deduplicate_results, sort_results

# Reranker type options for factory function
RerankerType = Literal["cross_encoder", "identity", "no_op", "nope", "none", "llm"]


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


# LLM-based Reranker Components


class ScoringCategory(BaseModel):
    """Scoring category with name, description, and numeric value."""

    name: str = Field(..., description="Category name (e.g., 'Perfect Match', 'Acceptable', 'Unusable')")
    description: str = Field(..., description="Detailed description of what this category represents")
    value: float = Field(..., description="Numeric score for this category")


class ScoringCriterion(BaseModel):
    """Individual criterion for evaluating results."""

    name: str = Field(..., description="Criterion name (e.g., 'Relevancy', 'Processing Level', 'Ease of Use')")
    description: str = Field(..., description="Detailed description of what this criterion evaluates")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Weight for this criterion (0.0 to 1.0)")
    scoring_categories: list[ScoringCategory] = Field(
        default_factory=lambda: [
            ScoringCategory(
                name="Perfect Match",
                description="Result fully satisfies the criterion requirements",
                value=3.0,
            ),
            ScoringCategory(
                name="Acceptable",
                description="Result partially satisfies the criterion requirements",
                value=2.0,
            ),
            ScoringCategory(
                name="Unusable",
                description="Result does not satisfy the criterion requirements",
                value=0.0,
            ),
        ],
        description="Scoring categories specific to this criterion with their numeric values",
    )


class IndividualScore(OutputSchema):
    """Score output from LLM for a single result against a single criterion."""

    criterion: str = Field(..., description="The criterion being evaluated")
    category: str = Field(..., description="The selected category (e.g., 'Perfect Match', 'Acceptable', 'Unusable')")
    reasoning: str = Field(..., description="Brief explanation for the score")


class IndividualScoreInput(InputSchema):
    """Input schema for scoring a single result."""

    query: str = Field(..., description="The search query")
    result_content: dict[str, Any] = Field(..., description="The result fields to evaluate")
    criterion: ScoringCriterion = Field(..., description="The criterion to evaluate against")


class LLMRerankerToolConfig(RerankerToolConfig):
    """Configuration for LLM-based reranker."""

    # LLM Agent configuration
    base_url: AnyUrl | None = Field(default=None, description="Base URL for LLM API")
    api_key: str | None = Field(default=None, description="API key for LLM")
    model_name: str = Field(default="gpt-4o-mini", description="LLM model name")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")
    agent_system_prompt: str = Field(
        default=(
            "You are an expert at evaluating search results. "
            "Analyze the provided result for the query against the given criterion and "
            "select the most appropriate category. Provide clear reasoning."
        ),
        description="System prompt for the internal scoring agent",
    )

    # Reranking configuration
    fields_to_evaluate: dict[str, str] = Field(
        ...,
        description=(
            "Dictionary mapping field names to their descriptions. "
            "Example: {'title': 'The dataset title', 'spatial_resolution': 'Ground sampling distance - lower is higher resolution'}"
        ),
    )
    scoring_criteria: list[ScoringCriterion] = Field(
        default_factory=lambda: [
            ScoringCriterion(
                name="Relevancy",
                description="How relevant is this result to the query?",
                weight=1.0,
            ),
            ScoringCriterion(
                name="Processing Level",
                description="How well does this result match the required processing level?",
                weight=0.5,
            ),
            ScoringCriterion(
                name="Ease of Use",
                description="How easy is it for the user to utilize this result?",
                weight=0.5,
            ),
        ],
        description="List of criteria to evaluate each result against",
    )
    prompt_template: str = Field(
        default=(
            "Evaluate the following result against the criterion.\n\n"
            "Query: {query}\n\n"
            "Result:\n{result_content}\n\n"
            "Criterion: {criterion_name}\n"
            "Description: {criterion_description}\n\n"
            "Available categories: {categories}\n\n"
            "Choose the most appropriate category and provide brief reasoning."
        ),
        description="Template for the evaluation prompt",
    )
    log_scores: bool = Field(
        default=True,
        description="Whether to log individual scores for post-hoc analysis",
    )


class LLMRerankerTool(RerankerTool):
    """
    LLM-based reranker that scores results individually using a language model.

    This reranker evaluates each result independently against configurable criteria,
    using an LLM to select categorical scores. The categorical scores are then
    mapped to numeric values for ranking and can be weighted for final scoring.

    Key features:
    - Individual result evaluation (not comparative)
    - Configurable evaluation criteria and scoring categories
    - Structured output using instructor
    - Score logging for post-hoc tuning
    - Weighted scoring across multiple criteria
    """

    config_schema = LLMRerankerToolConfig

    def __init__(
        self,
        config: LLMRerankerToolConfig | None = None,
        debug: bool = False,
    ):
        """
        Initialize the LLM reranker.

        Args:
            config: Reranker configuration
            debug: Enable debug logging
        """
        super().__init__(config=config, debug=debug)
        self.config: LLMRerankerToolConfig = self.config  # type hint

        # Create internal scoring agent
        agent_config = BaseAgentConfig(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            system_prompt=self.config.agent_system_prompt,
        )

        # Create a specialized scoring agent class with proper schema attributes
        class ScoringAgent(LiteLLMInstructorBaseAgent):
            input_schema = IndividualScoreInput
            output_schema = IndividualScore

        self.scoring_agent = ScoringAgent(
            config=agent_config,
            debug=debug,
        )

        # Normalize weights
        total_weight = sum(c.weight for c in self.config.scoring_criteria)
        if total_weight > 0:
            for criterion in self.config.scoring_criteria:
                criterion.weight = criterion.weight / total_weight

        # Pre-build field descriptions string
        self._field_desc_str = self._build_field_descriptions_str()

    def _build_field_descriptions_str(self) -> str:
        """Build field descriptions string once during init from fields_to_evaluate."""
        if not self.config.fields_to_evaluate:
            return ""

        desc = "\n\nFIELD DESCRIPTIONS:\n"
        desc += "The following fields are included in the result. Use these descriptions to understand what each field represents:\n"
        for field_name, field_desc in self.config.fields_to_evaluate.items():
            desc += f"- {field_name}: {field_desc}\n"
        return desc

    def _format_prompt_template(
        self,
        query: str,
        result_content: str,
        criterion_name: str,
        criterion_description: str,
        categories: str,
    ) -> str:
        """Format the prompt template with all variables."""
        prompt = self.config.prompt_template.format(
            query=query,
            result_content=result_content,
            criterion_name=criterion_name,
            criterion_description=criterion_description,
            categories=categories,
        )
        # Append field descriptions if present
        if self._field_desc_str:
            prompt += self._field_desc_str
        return prompt

    def _extract_result_fields(self, result: SearchResultItem) -> dict[str, Any]:
        """
        Extract relevant fields from a result for evaluation.

        Args:
            result: The result to extract fields from

        Returns:
            Dictionary mapping field names to their values
        """
        result_content = {}
        for field_name in self.config.fields_to_evaluate.keys():
            # Try direct attribute access first
            if hasattr(result, field_name):
                value = getattr(result, field_name)
                result_content[field_name] = value
            # Then try exact key match in extra
            elif field_name in result.extra:
                result_content[field_name] = result.extra[field_name]
            # Finally try case-insensitive search in extra
            else:
                field_lower = field_name.lower().replace(" ", "_").replace("-", "_")
                for key in result.extra.keys():
                    key_normalized = key.lower().replace(" ", "_").replace("-", "_")
                    if key_normalized == field_lower:
                        result_content[field_name] = result.extra[key]
                        break
        return result_content

    async def _score_single_result(
        self,
        query: str,
        result_content: dict[str, Any],
        criterion: ScoringCriterion,
    ) -> tuple[float, str, str]:
        """
        Score a single result against a single criterion.

        Args:
            query: The search query
            result_content: Pre-extracted content fields from the result
            criterion: The criterion to evaluate

        Returns:
            Tuple of (numeric_score, category, reasoning)
        """
        # Get categories specific to this criterion with descriptions
        categories_str = "\n".join(f"- {cat.name}: {cat.description}" for cat in criterion.scoring_categories)

        # Format the prompt with criterion-specific categories
        formatted_prompt = self._format_prompt_template(
            query=query,
            result_content="\n".join(f"{k}: {v}" for k, v in result_content.items()),
            criterion_name=criterion.name,
            criterion_description=criterion.description,
            categories=categories_str,
        )

        print(formatted_prompt)

        # Get LLM score using formatted prompt
        try:
            messages = [
                self.scoring_agent._default_system_message(),
                {
                    "role": "user",
                    "content": formatted_prompt,
                },
            ]

            score_output = await self.scoring_agent.get_response_async(
                messages=messages,
                response_model=self.scoring_agent.output_schema,
            )

            category = score_output.category
            reasoning = score_output.reasoning

            category_map = {cat.name: cat.value for cat in criterion.scoring_categories}
            numeric_score = category_map.get(category, 0.0)

            if self.debug:
                logger.debug(
                    f"Scored result for criterion '{criterion.name}': "
                    f"category={category}, score={numeric_score}, reasoning={reasoning[:100]}",
                )

            return numeric_score, category, reasoning

        except Exception as e:
            logger.error(f"Error scoring result for criterion '{criterion.name}': {e}")
            return 0.0, "Error", str(e)

    async def _rerank_results(self, query: str, results: list[SearchResultItem]) -> list[SearchResultItem]:
        """
        Rerank results by scoring each one individually across all criteria.

        Args:
            query: The search query
            results: List of results to rerank

        Returns:
            List of results sorted by weighted score
        """
        scored_results = []

        for idx, result in enumerate(results):
            result_content = self._extract_result_fields(result)

            # Score against all criteria in parallel using asyncio.gather
            scoring_tasks = [
                self._score_single_result(
                    query=query,
                    result_content=result_content,
                    criterion=criterion,
                )
                for criterion in self.config.scoring_criteria
            ]

            criterion_results = await asyncio.gather(*scoring_tasks)

            criterion_scores = {}
            total_score = 0.0

            for criterion, (numeric_score, category, reasoning) in zip(self.config.scoring_criteria, criterion_results):
                criterion_scores[criterion.name] = {
                    "category": category,
                    "score": numeric_score,
                    "reasoning": reasoning,
                    "weight": criterion.weight,
                }

                total_score += numeric_score * criterion.weight

            result.score = total_score
            result.extra["llm_reranker"] = {
                "total_score": total_score,
                "criterion_scores": criterion_scores,
            }

            if self.config.log_scores:
                logger.info(
                    f"Result {idx + 1}/{len(results)} | Total Score: {total_score:.3f} | "
                    f"Scores: {', '.join(f'{k}={v["score"]:.1f}' for k, v in criterion_scores.items())}",
                )

            scored_results.append(result)

        # Sort by total score (descending)
        return await self._sort_results(scored_results, sort_key="score")

    def __str__(self) -> str:
        criteria_names = [c.name for c in self.config.scoring_criteria]
        return f"{self.__class__.__name__} | criteria={criteria_names}"


def create_reranker(
    reranker_type: RerankerType,
    config: RerankerToolConfig | LLMRerankerToolConfig | None = None,
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
            - "llm": LLMRerankerTool using LLM for individual result scoring
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
        >>> # Create LLM reranker with custom config
        >>> config = LLMRerankerToolConfig(
        ...     model_name="gpt-4o",
        ...     temperature=0.0,
        ...     scoring_criteria=[...],
        ... )
        >>> reranker = create_reranker("llm", config=config)
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

    # LLM-based reranking
    if reranker_type == "llm":
        if not isinstance(config, LLMRerankerToolConfig):
            if config is not None:
                raise ValueError("config must be LLMRerankerToolConfig for 'llm' reranker type")
            config = LLMRerankerToolConfig()
        return LLMRerankerTool(config=config, debug=debug)

    # No-op/identity reranking - pass-through that returns original results
    if reranker_type in ("identity", "no_op", "nope", "none"):
        return NoOpRerankerTool(config=config, debug=debug)

    # Unknown type
    raise ValueError(
        f"Unknown reranker type: '{reranker_type}'. Supported types: cross_encoder, llm, identity, no_op, none, nope",
    )


# Export public API
__all__ = [
    # Type definitions
    "RerankerType",
    # Config and schemas
    "RerankerToolConfig",
    "RerankerToolInputSchema",
    "RerankerToolOutputSchema",
    "LLMRerankerToolConfig",
    "IndividualScoreInput",
    "IndividualScore",
    # Scoring components
    "ScoringCategory",
    "ScoringCriterion",
    # Base and implementations
    "RerankerTool",
    "CrossEncoderRerankerTool",
    "NoOpRerankerTool",
    "LLMRerankerTool",
    # Factory
    "create_reranker",
]
