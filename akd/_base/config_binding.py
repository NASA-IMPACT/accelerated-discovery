"""Opt-in config and metadata binding mixin.

Provides config property binding (``agent.x`` → ``agent.config.x``) and
metadata binding (name, description, IO hints) without requiring
``AbstractBaseMeta`` or inheriting from ``AbstractBase``.

Framework adapters that inherit a third-party agent class directly
(e.g. ``class PydanticAIAgent(pydantic_ai.Agent)``) can mix this in
to get akd-style config access and description building.

Config properties are collision-aware: fields already defined on the class
(e.g. set by the framework's ``__init__``) are skipped to avoid shadowing.
"""

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from akd.utils import get_model_fields, to_snake_case


def _make_config_property(field_name: str) -> property:
    """Create a property that delegates to ``self.config.<field_name>``."""

    def getter(self: Any) -> Any:
        if not hasattr(self, "config") or self.config is None:
            return self.__dict__.get(field_name)
        return getattr(self.config, field_name)

    def setter(self: Any, value: Any) -> None:
        if not hasattr(self, "config") or self.config is None:
            self.__dict__[field_name] = value
        else:
            setattr(self.config, field_name, value)

    return property(getter, setter)


def _make_computed_property(field_name: str) -> property:
    """Create a read-only property for a computed config field."""

    def getter(self: Any) -> Any:
        if not hasattr(self, "config") or self.config is None:
            return self.__dict__.get(field_name)
        return getattr(self.config, field_name)

    return property(getter)


def _format_single_schema(schema: type) -> str:
    """Format one schema's fields as bullet lines."""
    fields = get_model_fields(schema, skip_no_description=False)
    if not fields:
        return ""
    return "\n".join(
        f"- **{field['name']}**: {field.get('description', field['name'].replace('_', ' '))}" for field in fields
    )


def _format_schema_fields(schema: Any) -> str:
    """Format schema field names + descriptions as a string.

    Supports single types and union types (e.g. ``A | B``). For unions,
    each branch is labeled with its docstring and fields are listed under it.
    """
    if schema is None:
        return ""

    origin = get_origin(schema)
    if origin in (types.UnionType, Union):
        branches = [arg for arg in get_args(schema) if isinstance(arg, type) and issubclass(arg, BaseModel)]
        if not branches:
            return ""
        parts = []
        for branch in branches:
            doc = (branch.__doc__ or branch.__name__).strip().split("\n")[0]
            lines = _format_single_schema(branch)
            if lines:
                indented = "\n".join(f"  {line}" for line in lines.splitlines())
                parts.append(f"**{branch.__name__}**: {doc}\n{indented}")
            else:
                parts.append(f"**{branch.__name__}**: {doc}")
        return "\n".join(parts)

    if isinstance(schema, type):
        return _format_single_schema(schema)

    return ""


class ConfigBindingMixin:
    """Opt-in config property binding + metadata (name, description, IO hints).

    Config properties are created at class definition time via
    ``__init_subclass__``. Properties that conflict with existing class
    attributes (e.g. framework-owned fields like ``retries``, ``name``)
    are skipped automatically.

    Call ``_bind_metadata()`` in your ``__init__`` after ``self.config``
    is set to populate ``self.name`` and ``self.description``.

    Example::

        class PydanticAIAgent(ConfigBindingMixin, pydantic_ai.Agent):
            config_schema = MyConfig

            def __init__(self, config=None, **kw):
                self.config = config or self.config_schema()
                self._bind_metadata()
                pydantic_ai.Agent.__init__(self, system_prompt=self.description, ...)
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        config_cls = getattr(cls, "config_schema", None)
        if config_cls is None or not isinstance(config_cls, type) or not issubclass(config_cls, BaseModel):
            return

        # Regular model fields
        for field_name in getattr(config_cls, "model_fields", {}):
            if field_name in cls.__dict__:
                continue
            existing = getattr(cls, field_name, None)
            if existing is not None and not isinstance(existing, property):
                continue
            setattr(cls, field_name, _make_config_property(field_name))

        # Computed fields (read-only)
        for field_name in getattr(config_cls, "model_computed_fields", {}):
            if field_name in cls.__dict__:
                continue
            existing = getattr(cls, field_name, None)
            if existing is not None and not isinstance(existing, property):
                continue
            setattr(cls, field_name, _make_computed_property(field_name))

    def _bind_metadata(self) -> None:
        """Populate ``self.name`` and ``self.description`` from config + class info.

        Call this in ``__init__`` after ``self.config`` is set.
        """
        if not getattr(self, "name", None):
            self.name = to_snake_case(type(self).__name__)

        config = getattr(self, "config", None)
        desc = (getattr(config, "description", None) or (type(self).__doc__ or "")).strip()

        io_hints = getattr(config, "io_hints", True) if config else True
        if io_hints:
            in_info = _format_schema_fields(getattr(self, "input_schema", None))
            out_info = _format_schema_fields(getattr(self, "output_schema", None))
            if in_info:
                desc += f"\n\nINPUT FIELD DESCRIPTIONS:\n{in_info}"
            if out_info:
                desc += f"\n\nOUTPUT FIELD DESCRIPTIONS:\n{out_info}"

        self.description = desc


__all__ = ["ConfigBindingMixin"]
