"""Core data structures for parameter exposure system."""

import functools
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import StrEnum
from typing import Any

# Constant for metadata attribute name used across exposure system
_EXPOSED_META_VAR_NAME = "_exposed_meta"  # noqa
_BUILTIN_TYPES = (str, int, float, bytes, tuple, frozenset, list, dict, set)


@functools.cache
def get_annotated_type(base_type: type, prefix: str = "Annotated") -> type:
    """Create annotated subclass for any type."""

    class_name = f"{prefix}{base_type.__name__.capitalize()}"

    # Immutable built-in types
    if base_type in (str, int, float, bytes, tuple, frozenset):

        def __new__(cls, value, metadata=None):
            instance = base_type.__new__(cls, value)
            return instance

        def __init__(self, value, metadata=None):
            if metadata is not None:
                object.__setattr__(self, _EXPOSED_META_VAR_NAME, metadata)

        return type(
            class_name,
            (base_type,),
            {
                "__new__": __new__,
                "__init__": __init__,
            },
        )

    # Mutable built-in types
    elif base_type in (list, dict, set):

        def __init__(self, value, metadata=None):
            base_type.__init__(self, value)
            if metadata is not None:
                setattr(self, _EXPOSED_META_VAR_NAME, metadata)

        return type(
            class_name,
            (base_type,),
            {
                "__init__": __init__,
            },
        )

    # Custom types - empty subclass (will use __class__ swap)
    return type(
        class_name,
        (base_type,),
        {"__module__": base_type.__module__},
    )


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

    def __ror__[T](self, value: T) -> T:
        base_type = type(value)
        annotated_cls = get_annotated_type(base_type, prefix="Annotated")
        if base_type in _BUILTIN_TYPES:
            # built-in types - create new instance
            return annotated_cls(value, metadata=self)  # type: ignore
        value.__class__ = annotated_cls  # type: ignore
        setattr(value, _EXPOSED_META_VAR_NAME, self)
        return value


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
