from typing import List, Optional, Union

import instructor
import openai
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from pydantic import create_model

from ..config import CONFIG
from ..structures import ExtractionSchema, SingleEstimation
from .extraction import ExtractionInputSchema
from .intents import IntentAgent


def create_intent_agent(config: Optional[BaseAgentConfig] = None) -> IntentAgent:
    config = config or BaseAgentConfig(
        client=instructor.from_openai(
            openai.OpenAI(api_key=CONFIG.model_config_.api_keys.openai),
        ),
        model=CONFIG.model_config_.model_name,
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


def build_extraction_agent(
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
                openai.OpenAI(api_key=CONFIG.model_config_.api_keys.openai),
            ),
            model=CONFIG.model_config_.model_name,
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
