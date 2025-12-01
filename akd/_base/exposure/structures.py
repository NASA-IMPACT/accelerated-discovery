"""Core data structures for parameter exposure system."""

from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import StrEnum
from typing import Any

# Constant for metadata attribute name used across exposure system
_EXPOSED_META_VAR_NAME = "_exposed_meta"  # noqa


class ExposedParamTypeSource(StrEnum):
    """Source of type information for exposed parameters."""

    FGET = "fget"
    FSET = "fset"
    ANNOTATED = "annotated"
    RUNTIME = "runtime"
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass
class ExposedParam:
    """Metadata for exposed parameters.

    This model supports both decorator-based (@exposed_param) and annotation-based
    (Annotated[T, Exposed()]) parameter exposure patterns.

    Attributes:
        name: Parameter name
        description: Human readable description
        extra: Additional metadata for the parameter
        expose: Whether to create a property for external control
        persistent: Whether this is persistent state (True) or transient data (False)
        type_hint: Type hint captured from annotations (e.g., str, int, list[str])
    """

    name: str = ""
    description: str = ""
    expose: bool = True
    persistent: bool = True
    type_hint: Any | None = None
    type_source: ExposedParamTypeSource | None = None
    extra: dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class ExposedParamRuntimeInfo(ExposedParam):
    """Complete runtime information about an exposed parameter.

    Inherits name, description, and extra from ExposedParam.
    Adds runtime-specific fields like type, editability, and current value.

    Attributes:
        type_: Parameter type as string (e.g., 'str', 'int', 'float')
        type_source: Source of type inference: 'fget' (getter annotation), 'fset' (setter annotation), 'runtime' (inferred from value), or 'none' (unknown)
        editable: Whether the parameter has a setter and can be modified
        current_value: Current runtime value of the parameter (only populated when include_values=True)
    """

    type_: str = "unknown"
    editable: bool = False
    current_value: Any | None = None


# Helper functions for creating ExposedParam metadata


def Exposed(
    name: str | None = None,
    description: str | None = None,
    expose: bool = True,
    **extra,
) -> ExposedParam:
    """Create metadata for persistent, exposed config parameters.

    Use this in Annotated type hints to mark config fields that should be:
    - Exposed as properties on the class
    - Externally controllable (via UI, API, etc.)

    Args:
        name: Explicit parameter name (defaults to field name if not provided)
        description: Human readable description of the parameter
        **extra: Additional metadata stored in the 'extra' field

    Returns:
        ExposedParam instance configured for persistent, exposed parameters

    Example:
        ```python
        class MyConfig(BaseModel):
            temperature: Annotated[float, Exposed(
                description="LLM temperature",
            )] = 0.7
        ```
    """
    return ExposedParam(
        name=name or "",
        description=description or "",
        expose=expose,
        persistent=True,
        extra=extra,
    )
