from typing import Optional

import instructor
import openai
from atomic_agents.agents.base_agent import BaseAgent as AtomicBaseAgent
from atomic_agents.agents.base_agent import BaseAgentConfig, BaseIOSchema

from ..config import CONFIG


class BaseAgent(AtomicBaseAgent):
    def __init__(
        self,
        config: Optional[BaseAgentConfig] = None,
        debug: bool = False,
    ) -> None:
        config = config or BaseAgentConfig(
            client=instructor.from_openai(
                openai.OpenAI(api_key=CONFIG.model_config_.api_keys.openai),
            ),
            model=CONFIG.model_config_.model_name,
            temperature=0.0,
        )
        self.debug = debug
        super().__init__(config)
