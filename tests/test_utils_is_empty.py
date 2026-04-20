"""Tests for akd.utils.is_empty structural emptiness check."""

from datetime import datetime

import pytest
from pydantic import BaseModel, Field

from akd._base import OutputSchema
from akd.utils import is_empty


class _Inner(OutputSchema):
    """Inner schema for nested tests."""

    text: str = ""
    items: list[str] = Field(default_factory=list)


class _Outer(OutputSchema):
    """Outer schema with nested OutputSchema."""

    title: str = ""
    tags: list[str] = Field(default_factory=list)
    inner: _Inner | None = None
    inners: list[_Inner] = Field(default_factory=list)


class _Plain(BaseModel):
    """Plain pydantic model (not an OutputSchema)."""

    a: str = ""
    b: int | None = None


# ---------- scalar cases ----------


@pytest.mark.parametrize("value", [None, "", "   ", "\t\n", b""])
def test_scalar_empty(value):
    assert is_empty(value) is True


@pytest.mark.parametrize("value", ["x", " a ", b"x", 0, 1, False, True, 0.0, 3.14])
def test_scalar_non_empty(value):
    assert is_empty(value) is False


def test_datetime_not_empty():
    assert is_empty(datetime(2026, 4, 20)) is False


# ---------- container cases ----------


@pytest.mark.parametrize(
    "value",
    [
        [],
        (),
        set(),
        frozenset(),
        {},
        [None, "", []],
        [None, [None, {}]],
        {"a": None, "b": "", "c": []},
        {"a": {"b": {"c": None}}},
        ({}, [], ""),
    ],
)
def test_container_empty(value):
    assert is_empty(value) is True


@pytest.mark.parametrize(
    "value",
    [
        [0],
        [False],
        {"a": 0},
        {"a": "x"},
        [None, "", "x"],
        ({}, [], 1),
    ],
)
def test_container_non_empty(value):
    assert is_empty(value) is False


# ---------- pydantic BaseModel cases ----------


def test_plain_basemodel_empty():
    assert is_empty(_Plain()) is True


def test_plain_basemodel_non_empty_string():
    assert is_empty(_Plain(a="hi")) is False


def test_plain_basemodel_non_empty_int():
    assert is_empty(_Plain(b=0)) is False


# ---------- OutputSchema.is_empty() ----------


def test_output_schema_all_defaults_empty():
    assert _Inner().is_empty() is True


def test_output_schema_string_populated():
    assert _Inner(text="hello").is_empty() is False


def test_output_schema_list_populated():
    assert _Inner(items=["a"]).is_empty() is False


def test_output_schema_list_of_empty_strings_empty():
    assert _Inner(items=["", "   "]).is_empty() is True


def test_nested_output_schema_empty():
    assert _Outer(inner=_Inner()).is_empty() is True


def test_nested_output_schema_non_empty():
    assert _Outer(inner=_Inner(text="x")).is_empty() is False


def test_list_of_empty_nested_schemas_empty():
    assert _Outer(inners=[_Inner(), _Inner()]).is_empty() is True


def test_list_of_nested_schemas_one_populated():
    assert _Outer(inners=[_Inner(), _Inner(text="y")]).is_empty() is False


def test_top_level_field_populated():
    assert _Outer(title="t").is_empty() is False
