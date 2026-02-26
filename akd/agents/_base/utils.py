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
    branch_names = ", ".join(f'"{name}"' for name in schema_names)

    fields: dict[str, tuple[Any, Any]] = {
        "kind": (
            literal_kind,
            Field(
                ...,
                description=f"REQUIRED. Must be exactly one of: {branch_names}. "
                "This selects which output branch to populate. "
                "Only fill the field matching this value; leave all others null.",
            ),
        ),
    }
    for schema in schemas:
        key = schema.__name__
        branch_desc = (schema.__doc__ or key).strip()
        fields[key] = (
            schema | None,
            Field(default=None, description=f"Populate ONLY when kind={key!r}. {branch_desc}"),
        )

    doc = (
        "Unified output envelope. You MUST:\n"
        f"1. Set 'kind' to exactly one of: {branch_names}\n"
        "2. Populate ONLY the field matching 'kind'\n"
        "3. Leave all other branch fields null"
    )
    model = create_model(
        "UnifiedOutputSchema",
        __base__=OutputSchema,
        __doc__=doc,
        **cast(dict[str, Any], fields),
    )
    return cast(type[OutputSchema], model)


class UnifiedOutput:
    """Helper that returns an envelope model for output unions."""

    def __new__(cls, *schemas: type[OutputSchema]) -> type[OutputSchema]:
        return build_unified_output_model(tuple(schemas))
