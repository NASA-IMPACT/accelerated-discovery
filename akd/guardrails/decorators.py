"""Decorators for applying guardrails to agents and tools.

Example:
    from akd.guardrails import guardrail
    from akd.guardrails.providers import GraniteGuardianTool

    @guardrail(
        input_guardrail=GraniteGuardianTool(),
        fail_on_input_risk=True,
    )
    class MyAgent(BaseAgent):
        ...

    # Or apply to an existing instance:
    from akd.guardrails import apply_guardrails

    agent = MyAgent()
    guarded = apply_guardrails(agent, input_guardrail=GraniteGuardianTool())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, computed_field, create_model

from akd.configs.project import CONFIG
from akd.errors import GuardrailError, InputGuardrailTriggered, OutputGuardrailTriggered
from akd.guardrails._base import GuardrailInput, GuardrailOutput, GuardrailProtocol
from akd.guardrails.utils import extract_text_content

if TYPE_CHECKING:
    from akd.agents._base import BaseAgent
    from akd.tools._base import BaseTool


def _get_source_context(cls_or_obj: type | object, explicit_override: str | None = None) -> str | None:
    """Extract source context from class or instance for guardrail evaluation.

    Priority: explicit_override > description (if string) > __doc__

    Note: When cls_or_obj is a class and description is a @property,
    getattr returns the property object, not a string. We detect this
    and fall back to __doc__.
    """
    if explicit_override:
        return explicit_override.strip()
    desc = getattr(cls_or_obj, "description", None)
    # Only use description if it's actually a string (not a property object)
    if not isinstance(desc, str):
        cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
        desc = cls.__doc__
    return desc.strip() if desc else None


class GuardrailResultMixin(BaseModel):
    """Mixin that adds guardrail result fields to response models."""

    input_guardrail_result: GuardrailOutput | None = None
    output_guardrail_result: GuardrailOutput | None = None

    @computed_field
    @property
    def guardrails_passed(self) -> bool:
        """Returns True if no guardrails were triggered."""
        input_passed = self.input_guardrail_result is None or self.input_guardrail_result.passed
        output_passed = self.output_guardrail_result is None or self.output_guardrail_result.passed
        return input_passed and output_passed


def guardrail(
    input_guardrail: GuardrailProtocol | None = None,
    output_guardrail: GuardrailProtocol | None = None,
    fail_on_input_risk: bool | None = False,
    fail_on_output_risk: bool | None = False,
    input_fields: list[str] | None = None,
    output_fields: list[str] | None = None,
    source_context: str | None = None,
    debug: bool = False,
    **kwargs: Any,
):
    """Decorator to add guardrail checks to an agent or tool class.

    Args:
        input_guardrail: Guardrail to check inputs (can be composed with &, |, >>).
        output_guardrail: Guardrail to check outputs (can be composed with &, |, >>).
        fail_on_input_risk: Raise InputGuardrailTriggered if input risk detected.
            If None, uses CONFIG.guardrails.fail_on_input_risk.
        fail_on_output_risk: Raise OutputGuardrailTriggered if output risk detected.
            If None, uses CONFIG.guardrails.fail_on_output_risk.
        input_fields: Fields to extract text from in input params.
            If None, uses CONFIG.guardrails.input_fields.
        output_fields: Fields to extract text from in output.
            If None, uses CONFIG.guardrails.output_fields.
        debug: Enable debug logging for guardrail checks.

    Example:
        from akd.guardrails.providers import GraniteGuardianTool, RiskAgent

        granite = GraniteGuardianTool()
        risk = RiskAgent()

        @guardrail(
            input_guardrail=granite >> risk,  # fail-fast: granite first, then risk
            output_guardrail=granite,
            fail_on_input_risk=True,
        )
        class MyAgent(BaseAgent):
            ...
    """
    # Validate guardrail types at decoration time for early error detection
    if input_guardrail is not None and not isinstance(input_guardrail, GuardrailProtocol):
        raise GuardrailError(
            f"input_guardrail must implement GuardrailProtocol, got {type(input_guardrail).__name__}",
        )
    if output_guardrail is not None and not isinstance(output_guardrail, GuardrailProtocol):
        raise GuardrailError(
            f"output_guardrail must implement GuardrailProtocol, got {type(output_guardrail).__name__}",
        )

    def decorator[T](cls: type[T]) -> type[T]:
        class GuardedClass(cls):  # type: ignore[valid-type,misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._guardrail_input = input_guardrail
                self._guardrail_output = output_guardrail
                self._guardrail_fail_on_input = (
                    CONFIG.guardrails.fail_on_input_risk if fail_on_input_risk is None else fail_on_input_risk
                )
                self._guardrail_fail_on_output = (
                    CONFIG.guardrails.fail_on_output_risk if fail_on_output_risk is None else fail_on_output_risk
                )
                self._guardrail_input_fields = input_fields or CONFIG.guardrails.input_fields
                self._guardrail_output_fields = output_fields or CONFIG.guardrails.output_fields
                self._guardrail_log_warnings = CONFIG.guardrails.log_warnings
                self._guardrail_debug = debug
                self._guardrail_source_context = _get_source_context(cls, source_context)

            async def _check_input_guardrail(self, params: Any) -> GuardrailOutput | None:
                """Check input against guardrail.

                Returns:
                    GuardrailOutput if guardrail was run, None if no guardrail configured.

                Raises:
                    InputGuardrailTriggered: If risk detected and fail_on_input_risk=True.
                """
                if not self._guardrail_input:
                    if self._guardrail_debug:
                        logger.debug(f"[{cls.__name__}] No input guardrail configured, skipping")
                    return None

                text = extract_text_content(params, self._guardrail_input_fields)
                if self._guardrail_debug:
                    logger.debug(f"[{cls.__name__}] Checking input guardrail with text: {text[:200]}...")

                result = await self._guardrail_input.acheck(GuardrailInput(content=text))

                if self._guardrail_debug:
                    logger.debug(
                        f"[{cls.__name__}] Input guardrail result: passed={result.passed}, risks={result.detected_risks}",
                    )

                if not result.passed:
                    if self._guardrail_fail_on_input:
                        raise InputGuardrailTriggered(
                            f"Input guardrail triggered: {result.summary}",
                            output=result,
                        )
                    if self._guardrail_log_warnings:
                        logger.warning(f"Input guardrail warning: {result.summary}")

                return result

            async def _check_output_guardrail(self, output: Any, params: Any) -> GuardrailOutput | None:
                """Check output against guardrail.

                Returns:
                    GuardrailOutput if guardrail was run, None if no guardrail configured.

                Raises:
                    OutputGuardrailTriggered: If risk detected and fail_on_output_risk=True.
                """
                if not self._guardrail_output:
                    if self._guardrail_debug:
                        logger.debug(f"[{cls.__name__}] No output guardrail configured, skipping")
                    return None

                text = extract_text_content(output, self._guardrail_output_fields)
                context = extract_text_content(params, self._guardrail_input_fields)
                if self._guardrail_debug:
                    logger.debug(f"[{cls.__name__}] Checking output guardrail with text: {text[:200]}...")

                result = await self._guardrail_output.acheck(
                    GuardrailInput(
                        content=text,
                        context=context,
                        source_context=self._guardrail_source_context,
                    ),
                )

                if self._guardrail_debug:
                    logger.debug(
                        f"[{cls.__name__}] Output guardrail result: passed={result.passed}, risks={result.detected_risks}",
                    )

                if not result.passed:
                    if self._guardrail_fail_on_output:
                        raise OutputGuardrailTriggered(
                            f"Output guardrail triggered: {result.summary}",
                            output=result,
                        )
                    if self._guardrail_log_warnings:
                        logger.warning(f"Output guardrail warning: {result.summary}")

                return result

            def _wrap_response_with_guardrails(
                self,
                response: Any,
                input_result: GuardrailOutput | None,
                output_result: GuardrailOutput | None,
            ) -> Any:
                """Wrap response in a dynamic model with guardrail result fields."""
                if not hasattr(response, "model_dump"):
                    # Not a Pydantic model, can't wrap
                    return response

                OriginalClass = response.__class__

                # Create dynamic model: inherits from Original + GuardrailResultMixin
                GuardrailedClass = create_model(
                    OriginalClass.__name__,
                    __base__=(GuardrailResultMixin, OriginalClass),
                    __doc__=OriginalClass.__doc__,
                )

                return GuardrailedClass(
                    **response.model_dump(),
                    input_guardrail_result=input_result,
                    output_guardrail_result=output_result,
                )

            async def _arun(self, params: Any, **kwargs: Any) -> Any:
                input_result = await self._check_input_guardrail(params)
                output = await super()._arun(params, **kwargs)
                output_result = await self._check_output_guardrail(output, params)
                return self._wrap_response_with_guardrails(output, input_result, output_result)

        # Preserve class identity
        GuardedClass.__name__ = cls.__name__
        GuardedClass.__qualname__ = cls.__qualname__
        GuardedClass.__module__ = cls.__module__
        GuardedClass.__doc__ = cls.__doc__

        return GuardedClass  # type: ignore[return-value]

    return decorator


def apply_guardrails(
    component: "BaseAgent | BaseTool",
    input_guardrail: GuardrailProtocol | None = None,
    output_guardrail: GuardrailProtocol | None = None,
    fail_on_input_risk: bool | None = False,
    fail_on_output_risk: bool | None = False,
    input_fields: list[str] | None = None,
    output_fields: list[str] | None = None,
    source_context: str | None = None,
    debug: bool = False,
    **kwargs: Any,
) -> "BaseAgent | BaseTool":
    """Apply guardrails to an existing agent or tool instance.

    This is useful when you want to add guardrails to an instance that was
    created without the @guardrail decorator.

    Args:
        component: The agent or tool instance to wrap.
        input_guardrail: Guardrail to check inputs (can be composed with &, |, >>).
        output_guardrail: Guardrail to check outputs (can be composed with &, |, >>).
        fail_on_input_risk: Raise exception if input risk detected.
        fail_on_output_risk: Raise exception if output risk detected.
        input_fields: Fields to extract text from in input params.
        output_fields: Fields to extract text from in output.
        debug: Enable debug logging for guardrail checks.

    Returns:
        A new instance with guardrails applied.

    Example:
        from akd.guardrails import apply_guardrails
        from akd.guardrails.providers import GraniteGuardianTool

        agent = MyAgent()
        guarded = apply_guardrails(
            agent,
            input_guardrail=GraniteGuardianTool(),
            fail_on_input_risk=True,
        )
    """
    source_context = _get_source_context(component, source_context)

    # Create decorated class
    GuardedClass = guardrail(
        input_guardrail=input_guardrail,
        output_guardrail=output_guardrail,
        fail_on_input_risk=fail_on_input_risk,
        fail_on_output_risk=fail_on_output_risk,
        input_fields=input_fields,
        output_fields=output_fields,
        source_context=source_context,
        debug=debug,
    )(component.__class__)

    # Create new instance and copy state
    guarded = GuardedClass.__new__(GuardedClass)
    guarded.__dict__.update(component.__dict__)

    # Set guardrail attributes (in case __init__ wasn't called)
    guarded._guardrail_input = input_guardrail  # type: ignore[attr-defined]
    guarded._guardrail_output = output_guardrail  # type: ignore[attr-defined]
    guarded._guardrail_fail_on_input = (  # type: ignore[attr-defined]
        CONFIG.guardrails.fail_on_input_risk if fail_on_input_risk is None else fail_on_input_risk
    )
    guarded._guardrail_fail_on_output = (  # type: ignore[attr-defined]
        CONFIG.guardrails.fail_on_output_risk if fail_on_output_risk is None else fail_on_output_risk
    )
    guarded._guardrail_input_fields = input_fields or CONFIG.guardrails.input_fields  # type: ignore[attr-defined]
    guarded._guardrail_output_fields = output_fields or CONFIG.guardrails.output_fields  # type: ignore[attr-defined]
    guarded._guardrail_log_warnings = CONFIG.guardrails.log_warnings  # type: ignore[attr-defined]
    guarded._guardrail_debug = debug  # type: ignore[attr-defined]
    guarded._guardrail_source_context = source_context  # type: ignore[attr-defined]

    return guarded


__all__ = ["guardrail", "apply_guardrails"]
