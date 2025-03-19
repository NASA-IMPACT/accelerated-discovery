from typing import Optional

import instructor
import openai
from atomic_agents.agents.base_agent import BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

from ..config import CONFIG
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
