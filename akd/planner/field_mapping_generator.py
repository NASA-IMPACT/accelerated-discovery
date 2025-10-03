"""
LLM-based field mapping generator using Instructor for structured output.

Generates semantic field mappings when explicit mappings don't exist,
with confidence scoring and detailed reasoning.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from loguru import logger

from akd.agents._base import LiteLLMInstructorBaseAgent, BaseAgentConfig
from akd._base import InputSchema, OutputSchema
from akd.configs.project import CONFIG
from akd.planner.registry import AgentEntry


class FieldMappingEntry(BaseModel):
    """Single field mapping with confidence and reasoning."""

    target_field: str = Field(..., description="Target agent input field name")
    source_field: str = Field(..., description="Source agent output field name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    reasoning: str = Field(..., description="Explanation for this mapping")


class FieldMappingResult(OutputSchema):
    """Complete field mapping result from LLM."""

    mappings: List[FieldMappingEntry] = Field(
        ...,
        description="List of field mappings with confidence scores"
    )
    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the entire mapping set"
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes or warnings about the mapping"
    )


class FieldMappingInput(InputSchema):
    """Input schema for field mapping agent (not used, but required by agent framework)."""
    pass


class _FieldMappingAgent(LiteLLMInstructorBaseAgent[FieldMappingInput, FieldMappingResult]):
    """Internal agent for field mapping using LiteLLM Instructor."""

    input_schema = FieldMappingInput
    output_schema = FieldMappingResult


class FieldMappingGenerator:
    """
    LLM-based generator for semantic field mappings.

    Uses LiteLLMInstructorBaseAgent to generate intelligent mappings when
    field names don't match exactly between agents.
    """

    CONFIDENCE_EXACT = 0.95  # Exact semantic match
    CONFIDENCE_STRONG = 0.80  # Strong semantic similarity
    CONFIDENCE_WEAK = 0.60   # Uncertain mapping

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize field mapping generator.

        Args:
            model: Model name (defaults to CONFIG setting)
            temperature: Generation temperature (0.0 for deterministic)
            api_key: API key (defaults to CONFIG setting)
        """
        self.model = model or CONFIG.model_config_settings.model_name
        self.temperature = temperature
        self.api_key = api_key or CONFIG.model_config_settings.api_keys.openai

        # Create agent config
        agent_config = BaseAgentConfig(
            model_name=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            system_prompt=self._build_system_prompt(),
            stateless=True,
            enable_trimming=True,
        )

        # Initialize agent
        self.agent = _FieldMappingAgent(config=agent_config, debug=False)

        logger.info(
            f"Initialized FieldMappingGenerator with model={self.model}, "
            f"temperature={self.temperature}"
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt for field mapping task."""
        return """You are a semantic field mapping expert for multi-agent workflows.

Your task is to map output fields from a source agent to input fields of a target agent.

Guidelines:
1. **Exact Match** (confidence 0.95-1.0): Same semantic meaning, compatible types
2. **Strong Match** (confidence 0.80-0.94): Similar meaning, may need transformation
3. **Weak Match** (confidence < 0.80): Uncertain mapping, likely requires user input

Consider:
- Semantic similarity of field names and descriptions
- Type compatibility (string→string exact, list→list with item type check)
- Purpose and context of the fields
- Whether data transformation is needed

Provide detailed reasoning for each mapping explaining:
- Why you chose this source field
- Semantic relationship between source and target
- Any concerns about type compatibility
- Whether transformation is needed"""

    def _build_user_prompt(
        self,
        source_agent: AgentEntry,
        target_agent: AgentEntry,
        required_target_fields: List[str]
    ) -> str:
        """Build user prompt with agent schemas and required fields."""
        # Format source outputs
        source_outputs = "\n".join([
            f"  - {field.name} ({field.type}): {field.description}"
            for field in source_agent.output_schema.fields
        ])

        # Format target inputs
        target_inputs = "\n".join([
            f"  - {field.name} ({field.type}): {field.description}"
            for field in target_agent.input_schema.fields
            if field.name in required_target_fields
        ])

        return f"""Map fields from source agent to target agent.

**Source Agent**: {source_agent.name} (ID: {source_agent.agent_id})
Output fields:
{source_outputs}

**Target Agent**: {target_agent.name} (ID: {target_agent.agent_id})
Required input fields:
{target_inputs}

For each required target field, identify the best matching source output field.
Provide confidence scores and detailed reasoning."""

    async def generate_mapping(
        self,
        source_agent: AgentEntry,
        target_agent: AgentEntry,
        required_target_fields: List[str]
    ) -> FieldMappingResult:
        """
        Generate semantic field mapping using LLM.

        Args:
            source_agent: Source agent with output schema
            target_agent: Target agent with input schema
            required_target_fields: List of target fields that need mapping

        Returns:
            FieldMappingResult with mappings, confidence, and reasoning
        """
        logger.info(
            f"Generating field mapping: {source_agent.agent_id} -> {target_agent.agent_id}"
        )
        logger.debug(f"Required target fields: {required_target_fields}")

        user_prompt = self._build_user_prompt(
            source_agent,
            target_agent,
            required_target_fields
        )

        try:
            # Use the agent's get_response_async method
            result = await self.agent.get_response_async(
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                response_model=FieldMappingResult
            )

            logger.info(
                f"Generated mapping with overall confidence: {result.overall_confidence:.2f}"
            )

            # Log individual mappings
            for entry in result.mappings:
                logger.debug(
                    f"  {entry.target_field} <- {entry.source_field} "
                    f"(confidence: {entry.confidence:.2f})"
                )

            return result

        except Exception as e:
            logger.error(f"Error generating field mapping: {e}")
            raise

    def convert_to_mapping_dict(
        self,
        result: FieldMappingResult
    ) -> Dict[str, str]:
        """
        Convert FieldMappingResult to simple mapping dict.

        Args:
            result: FieldMappingResult from LLM

        Returns:
            Dict mapping target_field -> source_field
        """
        return {
            entry.target_field: entry.source_field
            for entry in result.mappings
        }

    def extract_reasoning(
        self,
        result: FieldMappingResult
    ) -> Dict[str, str]:
        """
        Extract per-field reasoning from result.

        Args:
            result: FieldMappingResult from LLM

        Returns:
            Dict mapping target_field -> reasoning
        """
        return {
            entry.target_field: entry.reasoning
            for entry in result.mappings
        }

    def should_request_approval(
        self,
        result: FieldMappingResult,
        threshold: float = 0.8
    ) -> bool:
        """
        Check if mapping requires user approval.

        Args:
            result: FieldMappingResult from LLM
            threshold: Confidence threshold for auto-approval

        Returns:
            True if any mapping is below threshold
        """
        # Check overall confidence
        if result.overall_confidence < threshold:
            logger.info(
                f"Mapping requires approval: overall confidence "
                f"{result.overall_confidence:.2f} < {threshold}"
            )
            return True

        # Check individual field confidences
        low_confidence = [
            entry for entry in result.mappings
            if entry.confidence < threshold
        ]

        if low_confidence:
            logger.info(
                f"Mapping requires approval: {len(low_confidence)} fields "
                f"below threshold {threshold}"
            )
            for entry in low_confidence:
                logger.debug(
                    f"  {entry.target_field}: confidence {entry.confidence:.2f}"
                )
            return True

        logger.info(
            f"Mapping auto-approved: all confidences >= {threshold}"
        )
        return False

    def format_approval_message(
        self,
        source_agent: AgentEntry,
        target_agent: AgentEntry,
        result: FieldMappingResult
    ) -> str:
        """
        Format human-readable approval message.

        Args:
            source_agent: Source agent
            target_agent: Target agent
            result: FieldMappingResult to present

        Returns:
            Formatted message for user review
        """
        lines = [
            "**Field Mapping Request**",
            "",
            f"**Source**: {source_agent.name} ({source_agent.agent_id})",
            f"**Target**: {target_agent.name} ({target_agent.agent_id})",
            f"**Overall Confidence**: {result.overall_confidence:.1%}",
            "",
            "**Proposed Mappings**:",
            ""
        ]

        for entry in result.mappings:
            confidence_emoji = "✅" if entry.confidence >= 0.8 else "⚠️"
            lines.extend([
                f"{confidence_emoji} **{entry.target_field}** ← {entry.source_field}",
                f"   Confidence: {entry.confidence:.1%}",
                f"   Reasoning: {entry.reasoning}",
                ""
            ])

        if result.notes:
            lines.extend([
                f"**Notes**: {result.notes}",
                ""
            ])

        lines.append("Do you approve this mapping?")

        return "\n".join(lines)
