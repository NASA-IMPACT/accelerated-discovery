from collections.abc import Awaitable, Callable
from inspect import Parameter, Signature
from typing import Any

from pydantic_core import PydanticUndefined

from akd._base import AbstractBase, BaseConfig, InputSchema, OutputSchema
from akd.utils import build_annotated_type


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

    def as_function(self) -> Callable[..., Awaitable[OutputSchema]]:
        """Return an async callable with a typed Python signature.

        Input parameters come from ``input_schema`` fields.
        Returns the raw Pydantic ``OutputSchema`` from ``arun()`` —
        consumer decides serialization (e.g. ``.model_dump()`` for FastMCP,
        ``.model_dump_json()`` for OpenAI SDK, or direct attribute access).
        """
        InputModel = self.input_schema
        parameters: list[Parameter] = []
        annotations: dict[str, Any] = {}

        for field_name, field in InputModel.model_fields.items():
            field_type = build_annotated_type(field)
            annotations[field_name] = field_type
            default = field.default if field.default is not PydanticUndefined else Parameter.empty
            parameters.append(
                Parameter(
                    field_name,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=field_type,
                ),
            )

        sig = Signature(parameters, return_annotation=self.output_schema)
        annotations["return"] = self.output_schema
        tool = self

        async def wrapper(*args: Any, **kwargs: Any) -> OutputSchema:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return await tool.arun(InputModel(**bound.arguments))

        wrapper.__name__ = self.name
        wrapper.__doc__ = self.description or self.__class__.__doc__ or ""
        wrapper.__signature__ = sig
        wrapper.__annotations__ = annotations
        return wrapper

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
                "name": self.name,
                "description": self.description or self.__class__.__doc__ or "",
                "parameters": self.input_schema.model_json_schema(),
            },
        }
