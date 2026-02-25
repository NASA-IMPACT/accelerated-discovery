"""OutputRoutingMixin — opt-in union output handling for providers that can't do unions natively."""

from __future__ import annotations

from typing import Any

from akd._base import OutputSchema
from akd._base.streaming import CompletedEvent, CompletedEventData, ToolResultEvent
from akd.tools.output import OutputTool

from .utils import UnifiedOutput


class OutputRoutingMixin:
    """Opt-in mixin for providers that need output tool routing for union types.

    Provides:
    - output_tools: list of OutputTool for multi_tool mode
    - _get_effective_output_schema(): returns the schema to use for structured output
    - _resolve_output_from_tool_result(): detects output from a single tool result
    - _tool_result_to_completed_event(): converts a ToolResultEvent to CompletedEvent if output tool
    - _unwrap_unified_output(): unwraps unified schema envelope

    Reads self.output_mode and self.output_schema_resolved from BaseAgent (via MRO).
    """

    def _post_init(self):
        super()._post_init()
        self.output_tools: list[OutputTool] = self._build_output_tools()

    def _build_output_tools(self) -> list[OutputTool]:
        """Build output tools according to union output mode."""
        schemas = self.output_schema_resolved
        if len(schemas) <= 1:
            return [OutputTool(schemas[0])]
        if self.output_mode != "multi_tool":
            return [OutputTool(schemas[0])]
        return [OutputTool(schema, name=f"final_{schema.__name__}") for schema in schemas]

    def _get_effective_output_schema(self) -> type[OutputSchema]:
        """Return schema used for provider structured-output requests."""
        schemas = self.output_schema_resolved
        if not schemas:
            raise TypeError("output_schema must declare at least one OutputSchema type")
        if len(schemas) <= 1:
            return schemas[0]
        if self.output_mode == "unified_schema":
            return UnifiedOutput(*schemas)
        return schemas[0]

    def _resolve_output_from_tool_result(self, result: Any) -> OutputSchema | None:
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
        """If this tool result event is from an output tool, return a CompletedEvent.

        Use in _run_engine_stream inline as results are yielded:
            yield tool_event
            if completed := self._tool_result_to_completed_event(tool_event):
                yield completed
                return
        """
        if resolved := self._resolve_output_from_tool_result(event.data.result):
            return CompletedEvent(
                source=event.source,
                message=f"Completed {event.source}",
                data=CompletedEventData(output=resolved),
                run_context=event.run_context,
            )
        return None

    def _unwrap_unified_output(self, output: Any) -> OutputSchema | None:
        """Unwrap single-envelope union output to a concrete schema branch."""
        if self.output_mode != "unified_schema":
            return None
        schemas = self.output_schema_resolved
        if len(schemas) <= 1:
            return None
        envelope_model = self._get_effective_output_schema()
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
