from typing import Any

from akd._base import AbstractBase, BaseConfig, InputSchema, OutputSchema


class BaseToolConfig(BaseConfig):
    """
    Configuration class for BaseTool.
    This class can be extended to add tool-specific configurations.
    """

    title: str | None = None


class BaseTool[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](AbstractBase):
    config_schema = BaseToolConfig

    def as_tool_definition(self) -> dict[str, Any]:
        """Convert tool to function calling format for LLM tool use.

        Returns OpenAI-compatible format with input parameters only.
        LiteLLM uses this format internally for all providers.

        Works with:
            - Class-based: class MyTool(BaseTool): ...
            - Function-based: @tool_wrapper def my_func(): ...

        Returns:
            dict: Tool definition in OpenAI function calling format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.__class__.__name__,
                "description": self.description or self.__class__.__doc__ or "",
                "parameters": self.input_schema.model_json_schema(),
            },
        }
