"""
Embedded instruction builder component for literature search agents.
"""

from typing import List, Optional

from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent
from akd.configs.prompts import RESEARCH_INSTRUCTION_AGENT_PROMPT


class InstructionBuilderInputSchema(InputSchema):
    """Input schema for instruction builder agent."""

    query: str = Field(..., description="Research query to build instructions for")
    context: Optional[str] = Field(
        default=None, description="Additional context for instruction building"
    )
    clarifications: Optional[List[str]] = Field(
        default=None, description="Clarifications gathered from the user/loop"
    )


class InstructionBuilderOutputSchema(OutputSchema):
    """Output schema for instruction builder agent."""

    research_instructions: str = Field(
        ..., description="Detailed research instructions"
    )
    search_strategy: str = Field(..., description="Recommended search strategy")
    key_concepts: List[str] = Field(
        default_factory=list, description="Key concepts to focus on"
    )


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

        # Debug preview of input (200 chars cap)
        if self.debug:
            preview_context = (instruction_input.context or "")[:200]
            preview_clar = ("; ".join(clarifications or []))[:200]
            logger.debug(
                f"InstructionBuilder input preview | query: {query[:200]} | context: {preview_context} | clarifications: {preview_clar}"
            )

        instruction_output = await self._agent.arun(instruction_input)

        if self.debug:
            logger.debug(
                f"Generated research instructions ({len(instruction_output.research_instructions)} chars)"
            )
            logger.debug(f"Key concepts: {instruction_output.key_concepts}")
            logger.debug(
                f"InstructionBuilder output preview | research_instructions: {instruction_output.research_instructions[:200]} | search_strategy: {instruction_output.search_strategy[:200]}"
            )

        return instruction_output.research_instructions
