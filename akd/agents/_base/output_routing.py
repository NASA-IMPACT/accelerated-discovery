"""OutputRoutingMixin — union output handling for multi_tool and unified_schema modes."""

from __future__ import annotations

from functools import cached_property
from typing import Any

from akd._base import OutputSchema, TextOutput
from akd._base.streaming import CompletedEvent, CompletedEventData, ToolResultEvent
from akd.tools.output import OutputTool

from .utils import UnifiedOutput


class OutputRoutingMixin:
    """Union output handling for multi_tool and unified_schema modes.

    Public API (override in your agent):
        check_output(output) -> str | None
            Check output quality before accepting. Return None to accept,
            or an error string to reject (sent back to model as retry).

        build_output_tools() -> list[OutputTool]
            Customize output tool construction. Default: one tool per schema branch.

        effective_output_schema -> type[OutputSchema]  (cached_property)
            The resolved output schema for structured output requests.

    Internal (don't override):
        _resolve_output_tool_result(result) -> OutputSchema | None
        _tool_result_to_completed_event(event) -> CompletedEvent | None
        _unwrap_unified_output(output) -> OutputSchema | None

    Reads self.output_mode and self.output_schema_resolved from BaseAgent (via MRO).
    """

    def _post_init(self):
        super()._post_init()
        self.output_tools: list[OutputTool] = self.build_output_tools()

    # ── Public: properties ──────────────────────────────────────────

    @cached_property
    def effective_output_schema(self) -> type[OutputSchema]:
        """The resolved output schema for structured output requests."""
        schemas = self.output_schema_resolved
        if not schemas:
            raise TypeError("output_schema must declare at least one OutputSchema type")
        if len(schemas) <= 1:
            return schemas[0]
        if self.output_mode == "unified_schema":
            return UnifiedOutput(*schemas)
        return schemas[0]

    # ── Public: override points ─────────────────────────────────────

    def build_output_tools(self) -> list[OutputTool]:
        """Build output tools for multi_tool mode. Override to customize."""
        schemas = self.output_schema_resolved
        if len(schemas) <= 1:
            return [OutputTool(schemas[0])]
        if self.output_mode != "multi_tool":
            return [OutputTool(schemas[0])]
        return [OutputTool(schema, name=f"final_{schema.__name__}") for schema in schemas]

    def check_output(self, output: OutputSchema) -> str | None:
        """Check output quality before accepting. Override for custom checks.

        Called when an output tool produces a result, before it becomes
        the agent's final answer. Like pydantic-ai's @output_validator.

        Returns:
            None — output is valid, accept it
            str  — rejection reason, sent back to model as retry feedback
        """
        if isinstance(output, TextOutput) and not output.content.strip():
            return "Empty response. Provide a substantive answer."
        if isinstance(output, str) and not output.strip():
            return "Empty response. Provide a substantive answer."
        return None

    # ── Private: plumbing ───────────────────────────────────────────

    def _resolve_output_tool_result(self, result: Any) -> OutputSchema | None:
        """Resolve completed output from a tool-result payload if it is an output tool."""
        tool_name = getattr(result, "tool_name", None)
        if not isinstance(tool_name, str) or getattr(result, "error", None):
            return None
        schemas = self.output_schema_resolved
        schema: type[OutputSchema] | None = (
            schemas[0]
            if len(schemas) == 1 and tool_name == "final_answer"
            else next((s for s in schemas if f"final_{s.__name__}" == tool_name), None)
        )
        if schema is None:
            return None
        content = getattr(result, "content", None)
        if isinstance(content, schema):
            return content
        if isinstance(content, dict):
            try:
                return schema.model_validate(content)
            except Exception:
                return None
        return None

    def _tool_result_to_completed_event(
        self,
        event: ToolResultEvent,
    ) -> CompletedEvent | None:
        """If this tool result is from an output tool AND passes validation, return CompletedEvent.

        Use in _run_engine_stream inline as results are yielded:
            yield tool_event
            if completed := self._tool_result_to_completed_event(tool_event):
                yield completed
                return
        """
        resolved = self._resolve_output_tool_result(event.data.result)
        if resolved is None:
            return None
        rejection = self.check_output(resolved)
        if rejection is not None:
            return None
        return CompletedEvent(
            source=event.source,
            message=f"Completed {event.source}",
            data=CompletedEventData(output=resolved),
            run_context=event.run_context,
        )

    def _unwrap_unified_output(self, output: Any) -> OutputSchema | None:
        """Unwrap single-envelope union output to a concrete schema branch."""
        if self.output_mode != "unified_schema":
            return None
        schemas = self.output_schema_resolved
        if len(schemas) <= 1:
            return None
        envelope_model = self.effective_output_schema
        if not isinstance(output, envelope_model):
            return None
        kind = getattr(output, "kind", None)
        selected_schema: type[OutputSchema] | None = None

        if isinstance(kind, str):
            selected_schema = next((schema for schema in schemas if schema.__name__ == kind), None)
        else:
            populated = [schema for schema in schemas if getattr(output, schema.__name__, None) is not None]
            if len(populated) == 1:
                selected_schema = populated[0]

        if selected_schema is None:
            return None

        branch = getattr(output, selected_schema.__name__, None)
        if isinstance(branch, OutputSchema):
            return branch
        if isinstance(branch, dict):
            return selected_schema.model_validate(branch)
        return None
