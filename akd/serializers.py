"""Custom serializers for AKD project.

This module provides custom serializers that extend LangGraph's JsonPlusSerializer
to handle Pydantic models and other complex objects properly.
"""

from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel
from pydantic.networks import HttpUrl


class AKDSerializer(JsonPlusSerializer):
    """Custom serializer that extends JsonPlusSerializer with better Pydantic handling.

    This serializer converts Pydantic models to dictionaries before serialization
    to ensure msgpack compatibility while maintaining all existing functionality.
    """

    def _convert_pydantic_to_dict(self, obj: Any) -> Any:
        """Recursively convert Pydantic models and special types to serializable formats."""
        if isinstance(obj, BaseModel):
            # Use model_dump with mode='json' to properly serialize all Pydantic types
            return obj.model_dump(mode="json")
        elif isinstance(obj, HttpUrl):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_pydantic_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_pydantic_to_dict(item) for item in obj]
        # elif isinstance(obj, NodeState):
        #     # Convert NodeState to a dictionary
        #     return {
        #         "messages": obj.messages,
        #         "inputs": obj.inputs,
        #         "outputs": obj.outputs,
        #         "input_guardrails": obj.input_guardrails,
        #         "output_guardrails": obj.output_guardrails,
        #         "steps": obj.steps,
        #         "tool_calls": [tool_call.model_dump(mode="json") for tool_call in obj.tool_calls],
        #     }
        else:
            return obj

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Convert object to typed binary format, handling Pydantic models."""
        # Convert Pydantic models to dictionaries before serialization
        converted_obj = self._convert_pydantic_to_dict(obj)
        return super().dumps_typed(converted_obj)

    def dumps(self, obj: Any) -> bytes:
        """Convert object to binary format, handling Pydantic models."""
        # Convert Pydantic models to dictionaries before serialization
        converted_obj = self._convert_pydantic_to_dict(obj)
        return super().dumps(converted_obj)
