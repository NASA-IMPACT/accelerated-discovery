"""Guardrails decorator and utilities for agent validation."""

from typing import List, Optional

from loguru import logger

from akd.configs.guardrails_config import GuardrailsConfig
from akd.tools.granite_guardian_tool import (
    GraniteGuardianInputSchema,
    GraniteGuardianTool,
    GraniteGuardianToolConfig,
    RiskDefinition,
)


def add_guardrails(
    input_guardrails: Optional[List[RiskDefinition]] = None,
    output_guardrails: Optional[List[RiskDefinition]] = None,
    config: Optional[GuardrailsConfig] = None,
):
    """
    Decorator to add Granite Guardian guardrails validation to any agent or tool class.

    This decorator enhances agent and tool classes with input/output risk validation using
    the Granite Guardian model. It follows the framework's decorator pattern for
    adding cross-cutting concerns to agents and tools.

    Args:
        input_guardrails: Risk types for input validation
        output_guardrails: Risk types for output validation
        config: Complete guardrails configuration (overrides individual parameters)

    Returns:
        Decorator function that wraps agent/tool classes with guardian validation

    Usage:
        @add_guardrails(
            input_guardrails=[RiskDefinition.JAILBREAK, RiskDefinition.HARM],
            output_guardrails=[RiskDefinition.ANSWER_RELEVANCE]
        )
        class MyAgent(InstructorBaseAgent):
            pass

        @add_guardrails(
            input_guardrails=[RiskDefinition.JAILBREAK],
            output_guardrails=[RiskDefinition.GROUNDEDNESS]
        )
        class MyTool(BaseTool):
            pass
    """

    def decorator(cls):
        class GuardedClass(cls):
            """Agent/Tool class enhanced with Granite Guardian guardrails validation."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._setup_guardrails_validation(
                    config, input_guardrails, output_guardrails
                )

            def _setup_guardrails_validation(
                self,
                config: Optional[GuardrailsConfig],
                input_guardrails: Optional[List[RiskDefinition]],
                output_guardrails: Optional[List[RiskDefinition]],
            ) -> None:
                """Initialize guardrails configuration and tool."""
                self.guardrails_config = config or GuardrailsConfig(
                    input_risk_types=input_guardrails
                    if input_guardrails is not None
                    else [
                        RiskDefinition.JAILBREAK,
                        RiskDefinition.HARM,
                        RiskDefinition.UNETHICAL_BEHAVIOR,
                    ],
                    output_risk_types=output_guardrails
                    if output_guardrails is not None
                    else [RiskDefinition.ANSWER_RELEVANCE, RiskDefinition.GROUNDEDNESS],
                )

                self.guardrails_tool = None
                if self.guardrails_config.enabled:
                    self.guardrails_tool = GraniteGuardianTool(
                        config=GraniteGuardianToolConfig(
                            model=self.guardrails_config.guardian_model,
                            ollama_type=self.guardrails_config.ollama_type,
                            snippet_n_chars=self.guardrails_config.snippet_n_chars,
                        ),
                        debug=getattr(self, "debug", False),
                    )

            async def _validate_with_guardrails(
                self, text: str, risk_types: List[RiskDefinition], is_input: bool = True
            ) -> bool:
                """Validate text with Granite Guardian model."""
                if (
                    not self.guardrails_config.enabled
                    or not self.guardrails_tool
                    or not text
                ):
                    return True

                try:
                    for risk_type in risk_types:
                        guardian_input = GraniteGuardianInputSchema(
                            query=text if is_input else "",
                            response="" if is_input else text,
                            risk_type=risk_type.value,
                        )

                        result = await self.guardrails_tool.arun(guardian_input)

                        for risk_result in result.risk_results:
                            if risk_result.get("is_risky", False):
                                return self._handle_risk_detection(
                                    text, risk_type, is_input
                                )

                    return True

                except Exception as e:
                    return self._handle_validation_error(e)

            def _handle_risk_detection(
                self, text: str, risk_type: RiskDefinition, is_input: bool
            ) -> bool:
                """Handle detected risk based on configuration."""
                text_type = "input" if is_input else "response"
                snippet = text[: self.guardrails_config.snippet_n_chars]
                message = f"Guardrails detected {risk_type.value} risk in {text_type}. Snippet: '{snippet}...'"

                if self.guardrails_config.fail_on_risk:
                    raise ValueError(f"Guardrails validation failed: {message}")
                else:
                    logger.warning(f"[Guardrails] {message}")
                    return False

            def _handle_validation_error(self, error: Exception) -> bool:
                """Handle validation errors based on configuration."""
                if self.guardrails_config.fail_on_risk:
                    raise error
                else:
                    logger.error(f"[Guardrails] Validation error: {error}")
                    return True  # Default to allowing on error

            def _extract_text_content(self, obj, preferred_fields: List[str]) -> str:
                """Extract text content from Pydantic object using preferred field order."""
                text_parts = []

                # Try preferred fields first
                for field_name in preferred_fields:
                    value = getattr(obj, field_name, "")
                    if value:
                        text_parts.append(str(value))

                # Fallback to all string fields if no preferred fields found
                if not text_parts:
                    for field_name, field_info in obj.model_fields.items():
                        if field_info.annotation is str:
                            value = getattr(obj, field_name, "")
                            if value:
                                text_parts.append(str(value))

                return " | ".join(text_parts) if text_parts else ""

            async def _arun(self, params, **kwargs):
                """Enhanced _arun with guardrails validation."""
                # Input validation
                if self.guardrails_config.enabled:
                    input_text = self._extract_text_content(
                        params, ["query", "content", "text", "user_input"]
                    )
                    input_passed = await self._validate_with_guardrails(
                        input_text,
                        self.guardrails_config.input_risk_types,
                        is_input=True,
                    )
                else:
                    input_passed = True

                # Run parent _arun
                response = await super()._arun(params, **kwargs)

                # Output validation
                if self.guardrails_config.enabled:
                    output_text = self._extract_text_content(
                        response, ["response", "answer", "content", "text"]
                    )
                    output_passed = await self._validate_with_guardrails(
                        output_text,
                        self.guardrails_config.output_risk_types,
                        is_input=False,
                    )
                else:
                    output_passed = True

                # Add guardrails status as computed field
                self._add_guardrails_status(response, input_passed and output_passed)
                return response

            def _add_guardrails_status(self, response, guardrails_passed: bool) -> None:
                """Add guardrails validation status to response object."""
                # Store the guardrails status in the object's __dict__ to bypass Pydantic validation
                object.__setattr__(response, "_guardrails_passed", guardrails_passed)

                # Add a method to check guardrails validation status
                def guardrails_validated():
                    return getattr(response, "_guardrails_passed", True)

                object.__setattr__(
                    response, "guardrails_validated", guardrails_validated
                )

        # Preserve original class metadata
        GuardedClass.__name__ = f"Guardrailed{cls.__name__}"
        GuardedClass.__qualname__ = f"Guardrailed{cls.__qualname__}"
        GuardedClass.__module__ = cls.__module__
        GuardedClass.__doc__ = cls.__doc__ or f"Guardrailed version of {cls.__name__}"

        # Preserve class attributes needed by the framework
        if hasattr(cls, "input_schema"):
            GuardedClass.input_schema = cls.input_schema
        if hasattr(cls, "output_schema"):
            GuardedClass.output_schema = cls.output_schema
        if hasattr(cls, "config_schema"):
            GuardedClass.config_schema = cls.config_schema

        return GuardedClass

    return decorator


# Note: Convenience aliases moved to avoid circular imports
# Users should create guardrailed agents/tools by decorating their classes:
# @add_guardrails(...)
# class MyAgent(InstructorBaseAgent):
#     pass
#
# @add_guardrails(...)
# class MyTool(BaseTool):
#     pass
