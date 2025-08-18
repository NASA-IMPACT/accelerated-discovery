"""
Embedded clarification component for literature search agents.
"""

from typing import Dict, List, Optional, Tuple

from loguru import logger
from pydantic import Field

from akd.agents._base import BaseAgentConfig, InstructorBaseAgent
from akd._base import InputSchema, OutputSchema
from akd.configs.prompts import CLARIFYING_AGENT_PROMPT


class ClarifyingAgentInputSchema(InputSchema):
    """Input schema for clarifying agent."""
    
    query: str = Field(..., description="Query that needs clarification")
    search_results: Optional[List[Dict]] = Field(default=None, description="Existing search results for context")


class ClarifyingAgentOutputSchema(OutputSchema):
    """Output schema for clarifying agent."""
    
    clarifying_questions: List[str] = Field(..., description="List of clarifying questions")
    needs_clarification: bool = Field(..., description="Whether clarification is needed")
    reasoning: str = Field(..., description="Reasoning for clarification needs")


class ClarificationComponentConfig(BaseAgentConfig):
    """Configuration for the embedded clarification component."""

    system_prompt: str = CLARIFYING_AGENT_PROMPT
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3


class ClarificationComponent:
    """
    Embedded clarification component that generates clarifying questions.

    This component is embedded within literature search agents to provide
    clarification functionality without requiring separate agent instantiation.
    """

    def __init__(
        self,
        config: Optional[ClarificationComponentConfig] = None,
        debug: bool = False,
    ):
        self.config = config or ClarificationComponentConfig()
        self.debug = debug

        # Create internal instructor agent for clarification processing
        self._agent = InstructorBaseAgent[
            ClarifyingAgentInputSchema, ClarifyingAgentOutputSchema
        ](config=self.config, debug=debug)
        self._agent.input_schema = ClarifyingAgentInputSchema
        self._agent.output_schema = ClarifyingAgentOutputSchema

    async def process(
        self, query: str, mock_answers: Optional[Dict[str, str]] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate clarifying questions and create enriched query.

        Args:
            query: The original research query
            mock_answers: Optional mock answers for testing

        Returns:
            Tuple of (enriched_query, clarifications)
        """
        if self.debug:
            logger.debug(f"Generating clarifying questions for: {query}")

        clarifying_input = ClarifyingAgentInputSchema(query=query)
        clarifying_output = await self._agent.arun(clarifying_input)

        if self.debug:
            logger.debug(f"Generated {len(clarifying_output.clarifying_questions)} questions")

        #TODO: In live AKD workflow, this would be an interrupt / interaction with the user
        # For now, we'll use mock answers or default responses
        clarifications = []
        for question in clarifying_output.clarifying_questions:
            answer = (mock_answers or {}).get(question, "No specific preference")
            clarifications.append(f"{question}: {answer}")

        # Create enriched query with clarifications
        enriched_query = f"{query}\n\nAdditional context:\n" + "\n".join(clarifications)

        if self.debug:
            logger.debug(
                f"Created enriched query with {len(clarifications)} clarifications"
            )

        return enriched_query, clarifications
