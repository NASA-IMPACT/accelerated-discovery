from typing import List, Optional, Union

import instructor
import openai
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from pydantic import create_model

from akd.agents.relevancy import MultiRubricRelevancyAgent
from akd.configs.project import CONFIG
from akd.structures import ExtractionSchema, SingleEstimation

from .extraction import ExtractionInputSchema
from .intents import IntentAgent
from .query import QueryAgent


def create_intent_agent(config: Optional[BaseAgentConfig] = None) -> IntentAgent:
    config = config or BaseAgentConfig(
        client=instructor.from_openai(
            openai.AsyncOpenAI(api_key=CONFIG.model_config_settings.api_keys.openai),
        ),
        model=CONFIG.model_config_settings.model_name,
        system_prompt_generator=SystemPromptGenerator(
            background=[
                ("You are an expert intent detector."),
            ],
            output_instructions=[
                "Estimation is when the data is used in a process to estimate numerically",
                "Data discovery deals with queries asking for data explicitly",
                "Example for Data Disoverery includes: Where do I find HLS data?",
            ],
        ),
    )
    return IntentAgent(config)


def create_extraction_agent(
    extraction_output_schema: Union[ExtractionSchema, List[SingleEstimation]],
    config: Optional[BaseAgentConfig] = None,
) -> BaseAgent:
    """
    Dynamically build the agent based on the type
    of extraction schema provided.
    """

    _ExtractionOutputSchema = create_model(
        "_ExtractionOutputSchema",
        answer=(extraction_output_schema, "Answer from the extraction"),
        __base__=BaseIOSchema,
        __doc__="Extracted information from the research",
    )
    return BaseAgent(
        BaseAgentConfig(
            client=instructor.from_openai(
                openai.OpenAI(api_key=CONFIG.model_config_settings.api_keys.openai),
            ),
            model=CONFIG.model_config_settings.model_name,
            system_prompt_generator=SystemPromptGenerator(
                background=[
                    (
                        "You are an expert in scientific information extraction.",
                        "Your goal is to accurately extract and summarize relevant "
                        "information from academic literature while maintaining fidelity "
                        "to the original sources.",
                    ),
                ],
                steps=[
                    "Identify patterns, contradictions, and "
                    "gaps in the literature across different sources.",
                    "Extract relevant information based on the given schema, "
                    "ensuring clarity and completeness.",
                    "Maintain structured and systematic extraction, "
                    "focusing on key arguments, methodologies, findings, and supporting data.",
                ],
                output_instructions=[
                    "Don't give anything that's not present in the content.",
                    "Ensure extracted content remains faithful to original sources, "
                    "avoiding extrapolation or misinterpretation.",
                    "Provide structured summaries including key arguments, "
                    "methodologies, findings, and limitations.",
                    "Use a scientific tone, ensuring clarity, coherence, "
                    "and proper citation handling.",
                    "Avoid speculation, personal opinions, or unverifiable claims.",
                ],
            ),
            input_schema=ExtractionInputSchema,
            output_schema=_ExtractionOutputSchema,
        ),
    )


def create_query_agent(config: Optional[BaseAgentConfig] = None) -> QueryAgent:
    config = config or BaseAgentConfig(
        client=instructor.from_openai(
            openai.AsyncOpenAI(api_key=CONFIG.model_config_settings.api_keys.openai),
        ),
        model=CONFIG.model_config_settings.model_name,
        system_prompt_generator=SystemPromptGenerator(
            background=[
                (
                    "You are an expert scientific search engine query generator with a deep understanding of which"
                    "queries will maximize the number of relevant results for science."
                ),
            ],
            steps=[
                "Analyze the given instruction to identify key concepts and aspects that need to be researched",
                "For each aspect, craft a search query using appropriate search operators and syntax",
                "Ensure queries cover different angles of the topic (technical, practical, comparative, etc.)",
            ],
            output_instructions=[
                "Return exactly the requested number of queries",
                "Format each query like a search engine query, not a natural language question",
                "Each query should be a concise string of keywords and operators",
            ],
        ),
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
