"""Internal helpers for BaseAgent output routing."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, cast

from pydantic import Field, create_model

from akd._base import OutputSchema


@lru_cache(maxsize=64)
def build_unified_output_model(schemas: tuple[type[OutputSchema], ...]) -> type[OutputSchema]:
    """Build one envelope model that can represent a union of output schemas."""
    schema_names = tuple(schema.__name__ for schema in schemas)
    literal_kind = Literal.__getitem__(schema_names)
    fields: dict[str, tuple[Any, Any]] = {
        "kind": (literal_kind | None, Field(default=None, description="Selected output branch")),
    }
    for schema in schemas:
        key = schema.__name__
        fields[key] = (schema | None, None)

    model = create_model(
        "UnifiedOutputSchema",
        __base__=OutputSchema,
        __doc__="Unified output envelope for union output mode.",
        **cast(dict[str, Any], fields),
    )
    return cast(type[OutputSchema], model)


class UnifiedOutput:
    """Helper that returns an envelope model for output unions."""

    def __new__(cls, *schemas: type[OutputSchema]) -> type[OutputSchema]:
        return build_unified_output_model(tuple(schemas))
