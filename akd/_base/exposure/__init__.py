"""Parameter exposure system for agents and tools.

This module provides a complete system for exposing and managing parameters
in agents and tools through both decorator-based and annotation-based approaches.
"""

# Public API: decorator and mixin
from .core import ParamExposureMixin, exposed_param

# Public API: metadata helpers and data structures
from .structures import (
    Exposed,
    ExposedParam,
    ExposedParamRuntimeInfo,
    ReadOnly,
    Validated,
)

__all__ = [
    # Core data structures
    "ExposedParam",
    "ExposedParamRuntimeInfo",
    # Metadata helpers (main public API)
    "Exposed",
    "Validated",
    "ReadOnly",
    # Decorator
    "exposed_param",
    # Mixin (main class to inherit from)
    "ParamExposureMixin",
]
