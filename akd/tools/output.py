"""Output tool for structured final answers (Pydantic AI pattern)."""

from typing import Any

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool


class _OutputToolPlaceholder(InputSchema, OutputSchema):
    """Placeholder schema for OutputTool class definition.

    Inherits from both InputSchema and OutputSchema to satisfy BaseTool's
    type bounds. The AbstractBaseMeta validates input_schema/output_schema
    at class definition time - actual schemas are set dynamically in __init__.
    """

    pass


class OutputTool(BaseTool[_OutputToolPlaceholder, _OutputToolPlaceholder]):
    """Submit your FINAL answer. Only call this AFTER you have gathered enough information using other tools. Do NOT call multiple final answer tools at the same time — pick exactly one."""

    input_schema = _OutputToolPlaceholder
    output_schema = _OutputToolPlaceholder

    def __init__(
        self,
        schema: type[OutputSchema],
        name: str = "final_answer",
        debug: bool = False,
    ) -> None:
        """Initialize OutputTool with dynamic schema.

        Args:
            schema: The agent's output schema to use for final_answer tool.
            debug: Enable debug mode.
        """
        # Override placeholders with actual schema BEFORE super().__init__()
        self._schema = schema
        self.input_schema = schema
        self.output_schema = schema
        super().__init__(debug=debug)
        # Keep default name for backwards compatibility.
        self.name = name
        # Schema-specific description for LLM tool routing
        schema_desc = (schema.__doc__ or schema.__name__).strip()
        self.description = (
            f"Submit your FINAL answer as {schema.__name__}. "
            f"Only call this AFTER you have gathered enough information using other tools. "
            f"Do NOT call multiple final answer tools at the same time — pick exactly one. "
            f"{schema_desc}"
        )

    def as_tool_definition(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        Overrides base to flatten schema (remove $defs) for better LLM compatibility.
        """
        schema = self._schema.model_json_schema()
        schema.pop("$defs", None)  # Remove definitions to flatten schema

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    f"Provide the final answer to the user's query. "
                    f"Call this tool when you have gathered enough information to respond. "
                    f"Output schema: {self._schema.__doc__ or self._schema.__name__}"
                ),
                "parameters": schema,
            },
        }

    async def _arun(
        self,
        params: _OutputToolPlaceholder,
        **kwargs,
    ) -> _OutputToolPlaceholder:
        """Return validated input as final output.

        The input is already validated against the schema by _execute_tool().
        This method just returns it as the final answer.
        """
        return params
