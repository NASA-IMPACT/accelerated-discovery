"""Enrich workflow nodes with full input schema metadata.

Transforms flat field values into rich schema objects so the backend/frontend
can render proper input forms without a separate enrichment step.

Prefers the live in-memory :class:`AgentRegistry` (which includes runtime-
registered agents like those from ``akd-ext``) and falls back to reading
``agent_registry.json`` on disk when no registry is provided.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from akd.utils import get_akd_root

if TYPE_CHECKING:
    from .registry import AgentRegistry

_REGISTRY_PATH = get_akd_root() / "akd" / "mapping" / "agent_registry.json"


def _build_input_schemas_from_registry(registry: AgentRegistry) -> dict[str, dict[str, Any]]:
    """Build {agent_id: {field_name: field_schema}} lookup from live registry object.

    Uses the in-memory registry which includes both auto-discovered agents
    and runtime-registered agents (e.g., CMRCareAgent from akd-ext).
    """
    schemas: dict[str, dict[str, Any]] = {}

    for agent in registry.get_all_agents():
        field_map: dict[str, Any] = {}
        for field_def in agent.input_schema.fields:
            schema: dict[str, Any] = {
                "required": field_def.required,
                "description": field_def.description or "",
                "type": field_def.type,
            }
            if field_def.default is not None:
                schema["default"] = field_def.default
            if field_def.items_type:
                schema["items_type"] = field_def.items_type
            if field_def.allowed_values:
                schema["allowed_values"] = field_def.allowed_values
            if field_def.value is not None:
                schema["value"] = field_def.value
            field_map[field_def.name] = schema
        schemas[agent.agent_id] = field_map

    return schemas


def _load_input_schemas_from_file() -> dict[str, dict[str, Any]]:
    """Fallback: Build {agent_id: {field_name: field_schema}} lookup from agent_registry.json on disk."""
    registry_path = Path(_REGISTRY_PATH)
    if not registry_path.exists():
        logger.warning(f"Agent registry not found at {registry_path}, skipping schema enrichment")
        return {}

    with open(registry_path) as f:
        registry_data = json.load(f)

    schemas: dict[str, dict[str, Any]] = {}
    for agent_id, agent_entry in registry_data.get("agents", {}).items():
        fields_list = agent_entry.get("input_schema", {}).get("fields", [])
        field_map: dict[str, Any] = {}
        for field_def in fields_list:
            name = field_def.get("name")
            if not name:
                continue
            schema: dict[str, Any] = {
                "required": field_def.get("required", False),
                "description": field_def.get("description", ""),
                "type": field_def.get("type", "string"),
            }
            if field_def.get("default") is not None:
                schema["default"] = field_def["default"]
            if field_def.get("items_type"):
                schema["items_type"] = field_def["items_type"]
            if field_def.get("allowed_values"):
                schema["allowed_values"] = field_def["allowed_values"]
            if field_def.get("value") is not None:
                schema["value"] = field_def["value"]
            field_map[name] = schema
        schemas[agent_id] = field_map

    return schemas


def enrich_workflow_schema(
    workflow_config: dict[str, Any],
    registry: AgentRegistry | None = None,
) -> dict[str, Any]:
    """
    Enrich workflow nodes with full schema metadata.

    For every node, each input field is expanded from a plain value:
        {"query": "volcanoes"}
    into a rich schema object:
        {"query": {"required": true, "type": "string", "value": "volcanoes", ...}}

    Fields that appear as keys in any node's io_map are skipped because
    they are filled at runtime by the orchestrator.

    Args:
        workflow_config: Raw workflow dict (e.g. from WorkflowFormat.model_dump())
        registry: Optional AgentRegistry instance.  When provided, schemas are
                  read from the live in-memory registry (includes runtime-
                  registered agents).  Falls back to agent_registry.json on disk.
    """
    if registry is not None:
        input_schemas = _build_input_schemas_from_registry(registry)
    else:
        input_schemas = _load_input_schemas_from_file()
    enriched_config = copy.deepcopy(workflow_config)

    io_mapped_fields: set[str] = set()
    for node in enriched_config.get("nodes", []):
        io_map = node.get("io_map") or {}
        io_mapped_fields.update(io_map.keys())

    for node in enriched_config.get("nodes", []):
        node_type = node.get("type")
        schema_fields = input_schemas.get(node_type, {})

        if not schema_fields:
            logger.debug(f"No input schema found for node type '{node_type}', skipping enrichment")
            continue

        existing_values: dict[str, Any] = {}
        for field_dict in node.get("input", {}).get("fields", []):
            if isinstance(field_dict, dict):
                for k, v in field_dict.items():
                    existing_values[k] = v

        merged_fields: list[dict[str, Any]] = []
        for field_name, field_schema in schema_fields.items():
            if field_name in io_mapped_fields:
                continue

            merged_field = copy.deepcopy(field_schema)

            # Only set value if planner actually provided one
            if field_name in existing_values:
                merged_field["value"] = existing_values[field_name]

                # Safety net: validate enum values
                if merged_field.get("allowed_values") and merged_field["value"] not in merged_field["allowed_values"]:
                    logger.warning(
                        f"Invalid enum value '{merged_field['value']}' for field '{field_name}'. "
                        f"Allowed: {merged_field['allowed_values']}. Removing invalid value."
                    )
                    del merged_field["value"]

            merged_fields.append({field_name: merged_field})

        node["input"]["fields"] = merged_fields

    return enriched_config
