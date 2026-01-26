"""Output tool for structured final answers (Pydantic AI pattern)."""

from typing import Any

from akd._base import OutputSchema
from akd.tools._base import BaseTool


class _OutputToolPlaceholder(OutputSchema):
    """Placeholder schema for OutputTool class definition.

    The AbstractBaseMeta validates input_schema/output_schema at class definition time.
    This placeholder satisfies that requirement - actual schemas are set in __init__.
    """

    pass


class OutputTool(BaseTool[_OutputToolPlaceholder, _OutputToolPlaceholder]):
    """Tool for structured final output in agent tool calling (Pydantic AI pattern).

    This special tool is registered alongside regular tools. When the model
    is ready to give its final answer, it calls this tool with structured
    output matching the agent's output schema.

    Unlike regular tools, OutputTool's schemas are dynamic - they're set at
    construction time based on the agent's output_schema.

    Example:
        output_tool = OutputTool(MyAgentOutputSchema)
        tool_def = output_tool.as_tool_definition()
        # {"type": "function", "function": {"name": "final_answer", ...}}
    """

    input_schema = _OutputToolPlaceholder
    output_schema = _OutputToolPlaceholder

    def __init__(
        self,
        schema: type[OutputSchema],
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
        # Set tool name to "final_answer"
        self.name = "final_answer"

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
