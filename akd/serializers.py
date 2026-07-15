"""Custom serializers for AKD project.

``AKDSerializer`` plays two roles:

* Lightweight Pydantic-aware converter — ``_convert_pydantic_to_dict`` is
  used by the mapping system to render models (and nested structures with
  special types like numpy arrays, enums, and ``HttpUrl``) into
  JSON-serializable dicts. This works with no extra dependencies.
* Optional langgraph checkpoint serde — when ``langgraph`` is installed,
  ``AKDSerializer`` additionally extends ``JsonPlusSerializer`` so it can be
  dropped into checkpointers like ``AsyncPostgresSaver(serde=AKDSerializer())``.

The ``langgraph`` import is lazy so ``akd`` core installs without it.
Consumers that need the checkpoint serde role should install
``akd[serializer]`` (or pin ``langgraph`` themselves).
"""

import sys
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic.networks import HttpUrl

try:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer as _SerdeBase

    _HAS_LANGGRAPH = True
except ImportError:  # langgraph is an optional extra (`akd[serializer]`)
    _SerdeBase = object
    _HAS_LANGGRAPH = False


class AKDSerializer(_SerdeBase):
    """Pydantic-aware serializer with optional langgraph checkpoint serde support.

    ``_convert_pydantic_to_dict`` always works. ``dumps`` / ``dumps_typed``
    require ``langgraph`` to be installed (they delegate to
    ``JsonPlusSerializer`` after converting Pydantic models to plain dicts).
    ``loads`` / ``loads_typed`` are inherited from ``JsonPlusSerializer``
    when available.
    """

    def _convert_pydantic_to_dict(self, obj: Any) -> Any:
        """Recursively convert Pydantic models and special types to serializable formats."""
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        elif isinstance(obj, HttpUrl):
            return str(obj)
        # numpy is an optional dep (akd[search]/akd[ml]); detect arrays without
        # importing it. If numpy was never imported, obj cannot be an ndarray.
        elif (_np := sys.modules.get("numpy")) is not None and isinstance(obj, _np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {(k.value if isinstance(k, Enum) else k): self._convert_pydantic_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_pydantic_to_dict(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize to typed binary, converting Pydantic models first (requires langgraph)."""
        if not _HAS_LANGGRAPH:
            raise RuntimeError(
                "AKDSerializer.dumps_typed requires `langgraph`. Install with `pip install akd[serializer]`.",
            )
        return super().dumps_typed(self._convert_pydantic_to_dict(obj))

    def dumps(self, obj: Any) -> bytes:
        """Serialize to binary, converting Pydantic models first (requires langgraph)."""
        if not _HAS_LANGGRAPH:
            raise RuntimeError(
                "AKDSerializer.dumps requires `langgraph`. Install with `pip install akd[serializer]`.",
            )
        return super().dumps(self._convert_pydantic_to_dict(obj))
