"""Utility functions for guardrails."""

from __future__ import annotations

from typing import Any


def extract_text_content(obj: Any, preferred_fields: list[str]) -> str:
    """Extract text content with field prioritization and fallback to auto-extraction.

    This matches the legacy guardrails text extraction semantics exactly.

    Args:
        obj: Object to extract text from (Pydantic model, dict, or other).
        preferred_fields: Field names to prioritize when extracting text.

    Returns:
        Extracted text content, separated by " | ".
    """
    if not obj:
        return ""

    # Step 1: Try preferred fields first
    preferred_content = _extract_preferred_fields(obj, preferred_fields)
    if preferred_content:
        return preferred_content

    # Step 2: Fallback - recursively collect all text content
    strings: list[str] = []
    visited: set[int] = set()
    _collect_all_content(obj, strings, 0, 3, visited)
    return " | ".join(strings) if strings else ""


def _extract_preferred_fields(obj: Any, preferred_fields: list[str]) -> str:
    """Extract content from preferred fields with priority ordering."""
    found_values: list[str] = []

    for field in preferred_fields:
        value = None

        # Try to get the field value
        if isinstance(obj, dict) and field in obj:
            value = obj[field]
        elif hasattr(obj, "model_fields") and field in obj.model_fields:
            value = getattr(obj, field, None)
        elif hasattr(obj, field):
            value = getattr(obj, field, None)

        # If we found the field, stringify it
        if value is not None:
            stringified = str(value).strip()
            if stringified:
                found_values.append(stringified)

    return " | ".join(found_values) if found_values else ""


def _iterate_object_fields(obj: Any):
    """Yield (key, value) pairs from any object type with password filtering."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if "password" not in str(key).lower():
                yield key, value
    elif hasattr(obj, "model_fields"):  # Pydantic
        for field_name in obj.model_fields.keys():
            if "password" not in field_name.lower():
                try:
                    value = getattr(obj, field_name, None)
                    yield field_name, value
                except Exception:
                    continue
    elif hasattr(obj, "__dict__"):  # Regular object
        for key, value in obj.__dict__.items():
            if not key.startswith("_") and "password" not in key.lower():
                yield key, value


def _process_field_value(value: Any, strings: list[str]) -> None:
    """Helper method to process field values with consistent stringification logic."""
    if isinstance(value, str) and value.strip():
        strings.append(value.strip())
    elif value is not None:
        stringified = str(value).strip()
        if stringified and stringified not in ["None", "[]", "{}", "0", "False"]:
            strings.append(stringified)


def _collect_all_content(
    obj: Any,
    strings: list[str],
    depth: int,
    max_depth: int,
    visited: set[int],
) -> None:
    """Recursively collect all content (strings + stringify non-strings) with safety guards."""
    if depth >= max_depth or obj is None or id(obj) in visited:
        return

    visited.add(id(obj))
    try:
        if isinstance(obj, str) and obj.strip():
            strings.append(obj.strip())
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _collect_all_content(item, strings, depth + 1, max_depth, visited)
        elif isinstance(obj, dict) or hasattr(obj, "model_fields") or hasattr(obj, "__dict__"):
            for key, value in _iterate_object_fields(obj):
                if isinstance(obj, dict):
                    # For dictionaries, recursively process values
                    _collect_all_content(value, strings, depth + 1, max_depth, visited)
                else:
                    # For Pydantic and regular objects, directly process values
                    _process_field_value(value, strings)
    except Exception:
        pass
    finally:
        visited.discard(id(obj))


__all__ = ["extract_text_content"]
