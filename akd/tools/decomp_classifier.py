"""
Decomposition classification tool for categorizing decomposed queries.

This module provides tools to classify decomposed research queries into categories
(EXACT, CALCULATOR, PROXY, TANGENTIAL) based on their relationship to the original topic.
Uses an LLM-based approach with structured outputs for reliable classification.
"""

from __future__ import annotations

from typing import Any, List

from loguru import logger
from pydantic import AnyUrl, BaseModel, Field
from pydantic import create_model

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent
from akd.structures import ClassifiedQuery, DecompositionClassification
from akd.tools._base import BaseTool, BaseToolConfig


class DecompClassifierInputSchema(InputSchema):
    """
    Input schema for decomposition classification tool.

    Attributes:
        original_topic: The original research question/topic that was decomposed
        queries: List of decomposed queries to classify
    """

    original_topic: str = Field(
        ...,
        description="The original research question or topic that was decomposed",
    )
    queries: List[str] = Field(..., description="List of decomposed queries to classify")


class DecompClassifierOutputSchema(OutputSchema):
    """
    Output schema for decomposition classification tool.

    Attributes:
        classified_queries: List of queries with their classifications and reasoning
    """

    classified_queries: List[ClassifiedQuery] = Field(
        ...,
        description="Queries with their classifications and reasoning",
    )


class DecompClassifierConfig(BaseToolConfig):
    """
    Configuration for the decomposition classifier tool.

    This config includes LLM settings and the detailed prompt template that
    guides the classification process.
    """

    base_url: AnyUrl | None = Field(default=None, description="Base URL for LLM API")
    api_key: str | None = Field(default=None, description="API key for LLM")
    model_name: str = Field(default="gpt-5-mini", description="LLM model name")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature for consistency")

    agent_system_prompt: str = Field(
        default=(
            "You are an expert at analyzing scientific research queries and classifying them "
            "based on their relationship to the original research topic.\n\n"
            "You will classify each decomposed query into one of four categories:\n\n"
            "1. EXACT - The query is essentially the same as the topic. A domain expert would say "
            "'yes, that is exactly what you asked for.' Example: Fire risk → Fire Weather Index\n\n"
            "2. CALCULATOR - The query is a mechanistic input or driver that physically affects "
            "the topic. Changing this variable would physically change the phenomenon. "
            "Example: Fire risk → soil moisture, wind speed\n\n"
            "3. PROXY - The query is not the phenomenon itself, but is used as a surrogate because "
            "it correlates with the topic or is easier to measure. Example: Phytoplankton biomass → "
            "chlorophyll-a concentration\n\n"
            "4. TANGENTIAL - The query is only indirectly or weakly related, potentially useful as "
            "context but not standard practice as a core input or proxy. Example: Fire risk → "
            "regional humidity (if not part of the risk index)\n\n"
            "Provide clear, concise reasoning for each classification."
        ),
        description="System prompt for the internal classification agent",
    )

    classification_prompt_template: str = Field(
        default=(
            "Original Research Topic: {original_topic}\n\n"
            "Decomposed Queries to Classify:\n{queries}\n\n"
            "CLASSIFICATION DECISION TREE:\n"
            "For each query, follow this decision process:\n\n"
            "1. Is the query conceptually the same quantity as the topic?\n"
            "   → If YES: Classify as EXACT\n\n"
            "2. Does the query enter the physical/statistical mechanism of the topic?\n"
            "   (i.e., would changing this variable physically change the topic?)\n"
            "   → If YES: Classify as CALCULATOR\n\n"
            "3. Is the query used as a surrogate because it tracks the topic?\n"
            "   (i.e., we use this because we can't easily measure the topic directly)\n"
            "   → If YES: Classify as PROXY\n\n"
            "4. Otherwise: Classify as TANGENTIAL\n\n"
            "EXAMPLES:\n"
            "- Fire risk → Fire Weather Index: EXACT (direct fire risk metric)\n"
            "- Fire risk → soil moisture: CALCULATOR (mechanistic input to fire risk)\n"
            "- Fire risk → wind speed: CALCULATOR (physical driver of fire spread)\n"
            "- Phytoplankton biomass → chlorophyll-a: PROXY (surrogate for biomass)\n"
            "- Ocean health → sea surface temperature: CALCULATOR (mechanistic input)\n"
            "- Flood risk → precipitation: CALCULATOR (physical driver)\n"
            "- Fire risk → regional humidity: TANGENTIAL (weakly related, not core)\n"
            "- Fire risk → ENSO SST: TANGENTIAL (indirect teleconnection)\n\n"
            "For EACH query above, classify it relative to the original topic and provide "
            "brief reasoning (1-2 sentences) explaining your classification."
        ),
        description="Template for the classification prompt with decision tree and examples",
    )


class DecompClassifierTool(BaseTool[DecompClassifierInputSchema, DecompClassifierOutputSchema]):
    """
    Tool for classifying decomposed queries using LLM-based structured output.

    This tool takes an original research topic and a list of decomposed queries,
    then classifies each query into one of four categories (EXACT, CALCULATOR,
    PROXY, TANGENTIAL) using a single LLM call that evaluates all queries together.

    The tool uses instructor with dynamic Pydantic models to ensure structured,
    reliable output from the LLM.

    Key features:
    - Single LLM call for all queries (efficient and context-aware)
    - Structured output with reasoning for each classification
    - Follows established pattern from LLMRerankerTool
    - Returns ClassifiedQuery objects with classification and reasoning

    Example:
        >>> config = DecompClassifierConfig(model_name="gpt-4o-mini")
        >>> classifier = DecompClassifierTool(config=config)
        >>> result = await classifier.arun(
        ...     classifier.input_schema(
        ...         original_topic="What is fire risk?",
        ...         queries=["soil moisture data", "Fire Weather Index"],
        ...     )
        ... )
        >>> for cq in result.classified_queries:
        ...     print(f"{cq.query} → {cq.classification}: {cq.reasoning}")
    """

    input_schema = DecompClassifierInputSchema
    output_schema = DecompClassifierOutputSchema
    config_schema = DecompClassifierConfig

    def __init__(
        self,
        config: DecompClassifierConfig | None = None,
        debug: bool = False,
    ):
        """
        Initialize the decomposition classifier tool.

        Args:
            config: Configuration for the classifier (LLM settings, prompts)
            debug: Enable debug logging
        """
        super().__init__(config=config, debug=debug)
        self.config: DecompClassifierConfig = self.config  # type hint

        # Create internal agent config
        agent_config = BaseAgentConfig(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            system_prompt=self.config.agent_system_prompt,
        )

        # Create dummy input schema for the internal agent
        class DummyInput(InputSchema):
            """Dummy input schema for classification agent."""

            pass

        # We'll create the dynamic output model per request since it depends
        # on the number of queries. For now, initialize the agent wrapper.
        self.agent_config = agent_config
        self.DummyInput = DummyInput

    def _create_dynamic_classification_model(self, queries: List[str]) -> type[BaseModel]:
        """
        Create a dynamic Pydantic model with one field per query.

        This allows the LLM to see each query as a separate field in the JSON schema,
        making it easier for structured output generation.

        Args:
            queries: List of query strings to create fields for

        Returns:
            Dynamically created Pydantic model class with one ClassifiedQuery field per query
        """

        def sanitize_field_name(name: str) -> str:
            """Convert query text to a valid Python identifier."""
            # Take first few words, replace special chars with underscores
            sanitized = name[:50].replace(" ", "_").replace("-", "_")
            # Remove non-alphanumeric chars except underscore
            sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
            # Ensure it starts with a letter
            if sanitized and not sanitized[0].isalpha():
                sanitized = "q_" + sanitized
            return sanitized or "query"

        # Build fields dict: {field_name: (type, Field(...))}
        query_fields = {}
        for idx, query in enumerate(queries):
            field_name = f"query_{idx}_{sanitize_field_name(query)}"
            field_description = (
                f"Classification for query: '{query}'. "
                "Select category (EXACT, CALCULATOR, PROXY, TANGENTIAL) and provide brief reasoning."
            )
            query_fields[field_name] = (
                ClassifiedQuery,
                Field(..., description=field_description),
            )

        # Create the dynamic model
        DynamicClassificationModel = create_model(
            "AllQueryClassifications",
            **query_fields,
        )

        return DynamicClassificationModel

    def _format_prompt(self, original_topic: str, queries: List[str]) -> str:
        """
        Format the classification prompt with the original topic and queries.

        Args:
            original_topic: The original research question
            queries: List of decomposed queries

        Returns:
            Formatted prompt string ready for LLM
        """
        # Format queries as numbered list
        queries_formatted = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

        # Fill in the template
        prompt = self.config.classification_prompt_template.format(
            original_topic=original_topic,
            queries=queries_formatted,
        )

        return prompt

    async def _classify_all_queries(
        self,
        original_topic: str,
        queries: List[str],
    ) -> List[ClassifiedQuery]:
        """
        Classify all queries in a single LLM call.

        Args:
            original_topic: The original research question
            queries: List of decomposed queries to classify

        Returns:
            List of ClassifiedQuery objects with classifications and reasoning
        """
        # Create dynamic output model for this specific set of queries
        DynamicOutputModel = self._create_dynamic_classification_model(queries)

        # Create classification agent with dynamic output schema
        class ClassificationAgent(LiteLLMInstructorBaseAgent):
            input_schema = self.DummyInput
            output_schema = DynamicOutputModel

        classification_agent = ClassificationAgent(
            config=self.agent_config,
            debug=self.debug,
        )

        # Format the prompt
        formatted_prompt = self._format_prompt(original_topic, queries)

        if self.debug:
            logger.debug(f"Classification prompt:\n{formatted_prompt}")

        try:
            # Call the LLM with structured output
            messages = [
                classification_agent._default_system_message(),
                {
                    "role": "user",
                    "content": formatted_prompt,
                },
            ]

            response = await classification_agent.get_response_async(messages=messages)

            # Extract classified queries from response
            response_dict = response.model_dump()

            classified_queries = []
            for idx, query in enumerate(queries):
                # Find the corresponding field in the response
                # The field names follow the pattern query_{idx}_...
                matching_key = None
                for key in response_dict.keys():
                    if key.startswith(f"query_{idx}_"):
                        matching_key = key
                        break

                if matching_key and response_dict[matching_key]:
                    # The value should already be a dict with 'query', 'classification', 'reasoning'
                    # But since we defined the field type as ClassifiedQuery, we need to reconstruct it
                    classification_data = response_dict[matching_key]

                    classified_query = ClassifiedQuery(
                        query=query,  # Use original query text
                        classification=DecompositionClassification(classification_data["classification"]),
                        reasoning=classification_data["reasoning"],
                    )
                    classified_queries.append(classified_query)

                    if self.debug:
                        logger.debug(
                            f"Classified '{query}' as {classified_query.classification}: "
                            f"{classified_query.reasoning[:100]}"
                        )
                else:
                    logger.warning(f"No classification found for query '{query}', defaulting to TANGENTIAL")
                    classified_queries.append(
                        ClassifiedQuery(
                            query=query,
                            classification=DecompositionClassification.TANGENTIAL,
                            reasoning="Classification not returned by LLM",
                        )
                    )

            return classified_queries

        except Exception as e:
            logger.error(f"Error classifying queries: {e}")
            # Return tangential for all queries on error
            return [
                ClassifiedQuery(
                    query=query,
                    classification=DecompositionClassification.TANGENTIAL,
                    reasoning=f"Error during classification: {str(e)}",
                )
                for query in queries
            ]

    async def _arun(self, params: DecompClassifierInputSchema) -> DecompClassifierOutputSchema:
        """
        Main execution method for the classifier tool.

        Args:
            params: Input parameters with original_topic and queries

        Returns:
            Output with classified queries including classifications and reasoning
        """
        if not params.queries:
            return DecompClassifierOutputSchema(classified_queries=[])

        # Classify all queries in one LLM call
        classified_queries = await self._classify_all_queries(
            original_topic=params.original_topic,
            queries=params.queries,
        )

        return DecompClassifierOutputSchema(classified_queries=classified_queries)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} | model={self.config.model_name}"

    def __repr__(self) -> str:
        return str(self)


# Export public API
__all__ = [
    "DecompClassifierInputSchema",
    "DecompClassifierOutputSchema",
    "DecompClassifierConfig",
    "DecompClassifierTool",
]
