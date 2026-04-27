"""
Data structures for AKD planner.

This module contains workflow planning data models.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from akd._base import OutputSchema
from akd.configs.project import CONFIG


class AgentSuggestion(OutputSchema):
    """Agent suggestion for workflow planning."""

    agent_id: str = Field(..., description="Agent identifier")
    agent_name: str = Field(..., description="Human-readable agent name")
    reason: str = Field(..., description="Why this agent is suggested")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this suggestion (0.0-1.0)")
    required_inputs: list[str] = Field(default_factory=list, description="Required input fields")
    expected_outputs: list[str] = Field(default_factory=list, description="Expected output fields")
    depends_on: list[str] | None = Field(default=None, description="Agent IDs this agent depends on for input data")


class WorkflowPlan(OutputSchema):
    """Workflow plan for execution by WorkflowBuilder."""

    workflow_description: str = Field(..., description="High-level description of the workflow")
    research_goal: str = Field(..., description="The research goal this workflow addresses")
    suggested_agents: list[AgentSuggestion] = Field(default_factory=list, description="Agents to include")
    workflow_steps: list[str] = Field(default_factory=list, description="High-level workflow steps")
    potential_issues: list[str] = Field(default_factory=list, description="Potential issues or limitations")

    @field_validator("workflow_steps")
    @classmethod
    def validate_steps_reference_agents(cls, steps: list[str], info) -> list[str]:
        """
        Validate that workflow steps reference actual agents.

        Checks that each step mentions at least one agent from suggested_agents.
        Note: Workflow steps are meant to be user-friendly descriptions that may use
        plain language without explicitly mentioning agent IDs. This validation logs
        at DEBUG level to avoid cluttering output for valid UX-optimized descriptions.
        """
        if not steps:
            return steps  # Empty list is okay

        # Get agent info from suggested_agents field
        suggested_agents = info.data.get("suggested_agents", [])
        if not suggested_agents:
            return steps  # Can't validate if no agents yet

        # Build set of agent identifiers to check against
        agent_identifiers = set()
        for agent in suggested_agents:
            # Add both agent_id and agent_name (normalized)
            agent_identifiers.add(agent.agent_id.lower())
            agent_identifiers.add(agent.agent_name.lower())
            # Add normalized versions (e.g., "my_agent" -> "my agent") I was able to visualize rough changes to            agent_identifiers.add(agent.agent_id.replace("_", " ").lower())

        # Check each step mentions at least one agent (soft validation)
        # Plain language steps without explicit agent names are valid UX choice
        for i, step in enumerate(steps, 1):
            step_lower = step.lower()
            if not any(agent_id in step_lower for agent_id in agent_identifiers):
                logger.debug(
                    f"Workflow step {i} ('{step[:50]}...') uses plain language description. "
                    f"Agents in workflow: {[a.agent_id for a in suggested_agents]}",
                )

        return steps


class FieldMappingEntry(BaseModel):
    """Single field mapping with confidence and reasoning."""

    target_field: str = Field(..., description="Target agent input field name")
    source_field: str = Field(..., description="Source agent output field name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Explanation for this mapping")


class FieldMappingResult(OutputSchema):
    """Complete field mapping result from LLM."""

    mappings: list[FieldMappingEntry] = Field(..., description="List of field mappings with confidence scores")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in the entire mapping set")
    notes: str | None = Field(None, description="Additional notes or warnings about the mapping")


class PlannerConfig(BaseModel):
    """Configuration for workflow planners."""

    model_name: str = Field(
        default_factory=lambda: CONFIG.model_config_settings.planner_model_name,
        description="LLM model to use for planning",
    )
    temperature: float = Field(default=0.3, description="Temperature for LLM generation (deterministic planning)")
    max_conversation_turns: int = Field(default=25, description="Maximum conversation turns before forcing completion")
    field_mapping_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for auto-approving field mappings",
    )
    input_extraction_temperature: float = Field(
        default=0.1,
        description="Temperature for input extraction LLM calls (very deterministic)",
    )
