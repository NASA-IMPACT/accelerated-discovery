"""
Embedded triage component for literature search agents.
"""

from typing import Optional

from loguru import logger
from pydantic import Field

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgentConfig, InstructorBaseAgent
from akd.configs.prompts import TRIAGE_AGENT_PROMPT


class TriageAgentInputSchema(InputSchema):
    """Input schema for triage agent."""

    query: str = Field(..., description="Research query to triage")


class TriageAgentOutputSchema(OutputSchema):
    """Output schema for triage agent."""

    routing_decision: str = Field(..., description="Routing decision for the query")
    needs_clarification: bool = Field(
        default=False, description="Whether query needs clarification"
    )
    reasoning: str = Field(..., description="Reasoning for the routing decision")


class TriageComponentConfig(BaseAgentConfig):
    """Configuration for the embedded triage component."""

    system_prompt: str = TRIAGE_AGENT_PROMPT
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1


class TriageComponent:
    """
    Embedded triage component that determines optimal query processing path.

    This component is embedded within literature search agents to provide
    triage functionality without requiring separate agent instantiation.
    """

    def __init__(
        self,
        config: Optional[TriageComponentConfig] = None,
        debug: bool = False,
    ):
        self.config = config or TriageComponentConfig()
        self.debug = debug

        # Create internal instructor agent for triage processing
        self._agent = InstructorBaseAgent[
            TriageAgentInputSchema, TriageAgentOutputSchema
        ](config=self.config, debug=debug)
        self._agent.input_schema = TriageAgentInputSchema
        self._agent.output_schema = TriageAgentOutputSchema

    async def process(self, query: str) -> TriageAgentOutputSchema:
        """
        Process query triage to determine optimal processing path.

        Args:
            query: The research query to triage

        Returns:
            Triage output with routing decision and reasoning
        """
        if self.debug:
            logger.debug(f"Triaging query: {query}")

        triage_input = TriageAgentInputSchema(query=query)

        # Debug preview of input (200 chars cap)
        if self.debug:
            logger.debug(f"Triage input preview | query: {query[:200]}")

        triage_output = await self._agent.arun(triage_input)

        if self.debug:
            logger.debug(f"Triage decision: {triage_output.routing_decision}")
            logger.debug(f"Needs clarification: {triage_output.needs_clarification}")
            logger.debug(
                f"Triage output preview | reasoning: {triage_output.reasoning[:200]}"
            )

        return triage_output
