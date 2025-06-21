from typing import List, Optional, Union

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import create_model, BaseModel, ConfigDict

from akd.agents.relevancy import MultiRubricRelevancyAgent
from akd.configs.project import CONFIG
from akd.configs.prompts import INTENT_SYSTEM_PROMPT, EXTRACTION_SYSTEM_PROMPT, QUERY_SYSTEM_PROMPT
from akd.structures import ExtractionSchema, SingleEstimation

from .extraction import ExtractionInputSchema
from .intents import IntentAgent
from .query import QueryAgent


def create_intent_agent(config: Optional[ConfigDict] = None) -> IntentAgent:
    config = config or ConfigDict(
        client=ChatOpenAI(
                    api_key=CONFIG.model_config_settings.api_keys.openai,
                    model=CONFIG.model_config_settings.model_name,
                    temperature=0.0,
            ),
        system_prompt=ChatPromptTemplate.from_messages(
                    [
                        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                        MessagesPlaceholder(variable_name="memory"),
                    ]
                ),
    )
    return IntentAgent(config)


def create_extraction_agent(
    extraction_output_schema: Union[ExtractionSchema, List[SingleEstimation]],
    config: Optional[ConfigDict] = None,
) -> BaseModel:
    """
    Dynamically build the agent based on the type
    of extraction schema provided.
    """

    _ExtractionOutputSchema = create_model(
        "_ExtractionOutputSchema",
        answer=(extraction_output_schema, "Answer from the extraction"),
        __base__=BaseModel,
        __doc__="Extracted information from the research",
    )
    return BaseModel(
        ConfigDict(
            client=ChatOpenAI(
                    api_key=CONFIG.model_config_settings.api_keys.openai,
                    model=CONFIG.model_config_settings.model_name,
            ),
            system_prompt=ChatPromptTemplate.from_messages(
                    [
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        MessagesPlaceholder(variable_name="memory"),
                    ]
                ),
            input_schema=ExtractionInputSchema,
            output_schema=_ExtractionOutputSchema,
        ),
    )


def create_query_agent(config: Optional[ConfigDict] = None) -> QueryAgent:
    config = config or ConfigDict(
        client=ChatOpenAI(
                    api_key=CONFIG.model_config_settings.api_keys.openai,
                    model=CONFIG.model_config_settings.model_name,
            ),
        system_prompt=ChatPromptTemplate.from_messages(
                    [
                        {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                        MessagesPlaceholder(variable_name="memory"),
                    ]
                )
    )
    return QueryAgent(config)


def create_multi_rubric_relevancy_agent() -> MultiRubricRelevancyAgent:
    """Factory function to create a MultiRubricRelevancyAgent with proper configuration"""

    return MultiRubricRelevancyAgent(
        config=BaseAgentConfig(
            client=instructor.from_openai(
                openai.AsyncOpenAI(
                    api_key=CONFIG.model_config_settings.api_keys.openai,
                ),
            ),
            model=CONFIG.model_config_settings.model_name,
            system_prompt_generator=SystemPromptGenerator(
                background=[
                    (
                        "You are an expert literature relevance assessor with deep expertise in academic research, "
                        "scientific methodology, and content quality evaluation. Your task is to evaluate content "
                        "against queries using multiple specific rubrics to ensure high-quality literature search results."
                    ),
                ],
                steps=[
                    "Carefully read and understand the query to identify the main topic, scope, and research requirements",
                    "Analyze the content systematically across all six relevancy dimensions",
                    "For Topic Alignment: Determine if content directly addresses the main concepts in the query",
                    "For Content Depth: Assess whether the treatment is comprehensive or just surface-level",
                    "For Recency Relevance: Evaluate if the content is current enough for the field (consider domain norms)",
                    "For Methodological Relevance: Check if methods/approaches used are sound and appropriate",
                    "For Evidence Quality: Assess the credibility and strength of evidence presented",
                    "For Scope Relevance: Determine if the content scope matches what the query is seeking",
                    "Synthesize individual assessments into an overall relevance judgment",
                    "Provide clear, specific reasoning for each dimension assessment",
                ],
                output_instructions=[
                    "Be strict in your assessments - content should meet high standards across multiple dimensions",
                    "For literature search, prioritize methodological soundness and evidence quality",
                    "Mark content as ALIGNED only if it directly addresses the main topic, not just tangentially related",
                    "Consider COMPREHENSIVE only if the content provides substantial, detailed coverage",
                    "Use METHODOLOGICALLY_SOUND only for rigorous, appropriate research approaches",
                    "Apply HIGH_QUALITY_EVIDENCE only to credible, well-supported claims from reliable sources",
                    "Provide specific, actionable reasoning steps that explain your assessment for each rubric",
                    "Be conservative in relevance judgments to maintain high-quality literature search results",
                ],
            ),
        ),
    )
