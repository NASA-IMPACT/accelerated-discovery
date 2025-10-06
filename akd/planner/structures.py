"""
Data structures for AKD planner.

This module contains workflow planning data models.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from akd._base import OutputSchema


class AgentSuggestion(OutputSchema):
    """Agent suggestion for workflow planning."""

    agent_id: str = Field(..., description="Agent identifier")
    agent_name: str = Field(..., description="Human-readable agent name")
    reason: str = Field(..., description="Why this agent is suggested")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this suggestion (0.0-1.0)")
    required_inputs: list[str] = Field(default_factory=list, description="Required input fields")
    expected_outputs: list[str] = Field(default_factory=list, description="Expected output fields")
    depends_on: Optional[list[str]] = Field(default=None, description="Agent IDs this agent depends on for input data")


class WorkflowPlan(OutputSchema):
    """Workflow plan for execution by WorkflowBuilder."""

    workflow_description: str = Field(..., description="High-level description of the workflow")
    research_goal: str = Field(..., description="The research goal this workflow addresses")
    suggested_agents: list[AgentSuggestion] = Field(default_factory=list, description="Agents to include")
    workflow_steps: list[str] = Field(default_factory=list, description="High-level workflow steps")
    estimated_complexity: Literal["low", "medium", "high"] = Field(..., description="Estimated complexity")
    potential_issues: list[str] = Field(default_factory=list, description="Potential issues or limitations")


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
    notes: Optional[str] = Field(None, description="Additional notes or warnings about the mapping")


class PlannerConfig(BaseModel):
    """Configuration for workflow planners."""

    model_name: str = Field(default="gpt-4", description="LLM model to use for planning")
    temperature: float = Field(default=0.7, description="Temperature for LLM generation")
    max_conversation_turns: int = Field(
        default=10, description="Maximum conversation turns before forcing completion"
    )
    auto_approve_high_confidence: bool = Field(
        default=False, description="Auto-approve plans with confidence > 0.9"
    )
