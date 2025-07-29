"""
Embedded instruction builder component for literature search agents.
"""

from typing import List, Optional

from loguru import logger

from akd.agents._base import BaseAgentConfig, InstructorBaseAgent
from akd.agents.schemas import (
    InstructionBuilderInputSchema,
    InstructionBuilderOutputSchema,
)
from akd.configs.prompts import RESEARCH_INSTRUCTION_AGENT_PROMPT


class InstructionBuilderComponentConfig(BaseAgentConfig):
    """Configuration for the embedded instruction builder component."""

    system_prompt: str = RESEARCH_INSTRUCTION_AGENT_PROMPT
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3


class InstructionBuilderComponent:
    """
    Embedded instruction builder component that creates detailed research instructions.

    This component is embedded within literature search agents to provide
    instruction building functionality without requiring separate agent instantiation.
    """

    def __init__(
        self,
        config: Optional[InstructionBuilderComponentConfig] = None,
        debug: bool = False,
    ):
        self.config = config or InstructionBuilderComponentConfig()
        self.debug = debug

        # Create internal instructor agent for instruction building
        self._agent = InstructorBaseAgent[
            InstructionBuilderInputSchema, InstructionBuilderOutputSchema
        ](config=self.config, debug=debug)
        self._agent.input_schema = InstructionBuilderInputSchema
        self._agent.output_schema = InstructionBuilderOutputSchema

    async def process(
        self, query: str, clarifications: Optional[List[str]] = None
    ) -> str:
        """
        Build detailed research instructions from query and clarifications.

        Args:
            query: The research query (possibly enriched)
            clarifications: Optional list of clarification responses

        Returns:
            Detailed research instructions string
        """
        if self.debug:
            logger.debug(f"Building research instructions for query: {query[:100]}...")

        instruction_input = InstructionBuilderInputSchema(
            query=query,
            clarifications=clarifications,
        )

        instruction_output = await self._agent.arun(instruction_input)

        if self.debug:
            logger.debug(
                f"Generated research instructions ({len(instruction_output.research_instructions)} chars)"
            )
            logger.debug(f"Focus areas: {instruction_output.focus_areas}")

        return instruction_output.research_instructions
