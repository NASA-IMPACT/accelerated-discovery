"""
Deep Research Agents for multi-agent research workflows.

This module implements agents for the Deep Research pattern inspired by OpenAI's
Deep Research API, adapted to work within the akd framework.
"""

from akd.configs.prompts import (
    CLARIFYING_AGENT_PROMPT,
    DEEP_RESEARCH_AGENT_PROMPT,
    RESEARCH_INSTRUCTION_AGENT_PROMPT,
    TRIAGE_AGENT_PROMPT,
)

from ._base import BaseAgentConfig, InstructorBaseAgent
from .schemas import (
    ClarifyingAgentInputSchema,
    ClarifyingAgentOutputSchema,
    DeepResearchInputSchema,
    DeepResearchOutputSchema,
    InstructionBuilderInputSchema,
    InstructionBuilderOutputSchema,
    TriageAgentInputSchema,
    TriageAgentOutputSchema,
)


class ClarifyingAgentConfig(BaseAgentConfig):
    """Configuration for the ClarifyingAgent."""

    system_prompt: str = CLARIFYING_AGENT_PROMPT
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3


class ClarifyingAgent(
    InstructorBaseAgent[ClarifyingAgentInputSchema, ClarifyingAgentOutputSchema]
):
    """
    Agent that asks clarifying questions to enrich user research queries.
    
    This agent analyzes research queries and generates 2-3 targeted questions
    to gather additional context that will improve research quality.
    """

    input_schema = ClarifyingAgentInputSchema
    output_schema = ClarifyingAgentOutputSchema
    config_schema = ClarifyingAgentConfig

    def __init__(
        self,
        config: ClarifyingAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the ClarifyingAgent with configuration."""
        config = config or ClarifyingAgentConfig()
        super().__init__(config=config, debug=debug)


class InstructionBuilderConfig(BaseAgentConfig):
    """Configuration for the InstructionBuilderAgent."""

    system_prompt: str = RESEARCH_INSTRUCTION_AGENT_PROMPT
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3


class InstructionBuilderAgent(
    InstructorBaseAgent[InstructionBuilderInputSchema, InstructionBuilderOutputSchema]
):
    """
    Agent that transforms user queries into detailed research instructions.
    
    This agent takes the original query and any clarifications, then creates
    comprehensive research instructions optimized for deep research execution.
    """

    input_schema = InstructionBuilderInputSchema
    output_schema = InstructionBuilderOutputSchema
    config_schema = InstructionBuilderConfig

    def __init__(
        self,
        config: InstructionBuilderConfig | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the InstructionBuilderAgent with configuration."""
        config = config or InstructionBuilderConfig()
        super().__init__(config=config, debug=debug)


class TriageAgentConfig(BaseAgentConfig):
    """Configuration for the TriageAgent."""

    system_prompt: str = TRIAGE_AGENT_PROMPT
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1  # Lower temperature for more consistent routing


class TriageAgent(InstructorBaseAgent[TriageAgentInputSchema, TriageAgentOutputSchema]):
    """
    Agent that triages research queries to determine the optimal processing path.
    
    This agent quickly assesses whether a query needs clarification, instruction
    building, or can proceed directly to research.
    """

    input_schema = TriageAgentInputSchema
    output_schema = TriageAgentOutputSchema
    config_schema = TriageAgentConfig

    def __init__(
        self,
        config: TriageAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the TriageAgent with configuration."""
        config = config or TriageAgentConfig()
        super().__init__(config=config, debug=debug)


class DeepResearchAgentConfig(BaseAgentConfig):
    """Configuration for the DeepResearchAgent."""

    system_prompt: str = DEEP_RESEARCH_AGENT_PROMPT
    model_name: str = "gpt-4o"  # Use more powerful model for research
    temperature: float = 0.2
    max_tokens: int = 4000  # Allow longer outputs for comprehensive reports


class DeepResearchAgent(
    InstructorBaseAgent[DeepResearchInputSchema, DeepResearchOutputSchema]
):
    """
    Agent that performs deep, iterative research based on detailed instructions.
    
    This agent executes comprehensive research using multiple search iterations,
    source evaluation, and synthesis to produce high-quality research reports.
    It's designed to work with search tools and relevancy agents to ensure
    thorough coverage of research topics.
    """

    input_schema = DeepResearchInputSchema
    output_schema = DeepResearchOutputSchema
    config_schema = DeepResearchAgentConfig

    def __init__(
        self,
        config: DeepResearchAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the DeepResearchAgent with configuration."""
        config = config or DeepResearchAgentConfig()
        super().__init__(config=config, debug=debug)

    async def _arun(
        self,
        params: DeepResearchInputSchema,
        **kwargs,
    ) -> DeepResearchOutputSchema:
        """
        Override to add research-specific logic if needed.
        
        This base implementation uses the standard InstructorBaseAgent flow,
        but can be extended to add features like:
        - Progress tracking
        - Intermediate result caching
        - Dynamic iteration control
        """
        # For now, use the standard implementation
        # Future enhancement: Add research-specific orchestration here
        return await super()._arun(params, **kwargs)