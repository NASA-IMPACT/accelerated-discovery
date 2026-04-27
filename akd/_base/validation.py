"""Standalone input/output validation functions.

Used by both AbstractBase (internally) and framework adapters that
don't inherit AbstractBase (e.g. PydanticAIAgent).
"""

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from .errors import SchemaValidationError


def validate_schema(schema: type, value: Any, *, coerce_dict: bool = False) -> Any:
    """Validate value against a schema. Supports union types (A | B).

    If coerce_dict=True, attempts schema(**value) when value is a dict.
    """
    origin = get_origin(schema)
    if origin in (types.UnionType, Union):
        args = [arg for arg in get_args(schema) if isinstance(arg, type) and issubclass(arg, BaseModel)]
        if isinstance(value, tuple(args)):
            return value
        if coerce_dict and isinstance(value, dict):
            for arg in args:
                try:
                    return arg(**value)
                except Exception:
                    continue
            raise SchemaValidationError(f"Dict doesn't match any of: {[a.__name__ for a in args]}")
        raise TypeError("Must be an instance of one of: " + ", ".join(a.__name__ for a in args))

    if isinstance(value, schema):
        return value
    if coerce_dict and isinstance(value, dict):
        try:
            return schema(**value)
        except ValidationError as e:
            raise SchemaValidationError(f"Invalid parameters: {e}") from e
    raise TypeError(f"Must be an instance of {schema.__name__}")


def validate_input(input_schema: type, params: Any) -> Any:
    """Validate input params — coerces dicts, supports unions."""
    return validate_schema(input_schema, params, coerce_dict=True)


def validate_output(output_schema: type, output: Any) -> Any:
    """Validate output — supports unions, no dict coercion."""
    return validate_schema(output_schema, output)


__all__ = ["validate_input", "validate_output", "validate_schema"]
