"""
Clean workflow builder for AKD planner.

Builds executable WorkflowFormat from WorkflowPlan with runtime data flow using io_map.
Uses three-tier field mapping strategy:
1. Explicit mappings (human-defined)
2. Exact name match (automatic)
3. LLM-generated mappings (intelligent fallback)
"""

import re
from typing import TypedDict

import jsonpath_ng
from loguru import logger

from .field_mapping_registry import FieldMappingRegistry
from .format_builder import (
    WORKFLOW_FORMAT_VERSION,
    WORKFLOW_TYPE,
    FieldValue,
    WorkflowEdge,
    WorkflowFormat,
    WorkflowNode,
    WorkflowNodeIO,
)
from .registry import AgentEntry, AgentRegistry
from .structures import WorkflowPlan


class UnmappedFieldInfo(TypedDict):
    """Information about unmapped fields between agents."""

    source_agent_id: str
    target_agent_id: str
    unmapped_fields: list[str]


# Type alias for agent inputs
AgentInputs = dict[str, FieldValue]


class WorkflowBuilder:
    """Builds WorkflowFormat with io_map for runtime data flow using intelligent field mapping."""

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        mapping_registry: FieldMappingRegistry | None = None,
        debug: bool = False,
    ):
        """
        Initialize workflow builder.

        Args:
            registry: Agent registry for schema access
            mapping_registry: Field mapping registry (creates default if None)
        """
        self.registry = registry or AgentRegistry()
        self.mapping_registry = mapping_registry or FieldMappingRegistry()
        self.debug = debug

    def _build_and_validate_jsonpath(self, agent_id: str, field_name: str) -> str:
        """
        Build and validate JSONPath expression for field mapping.

        Args:
            agent_id: Agent identifier to reference in JSONPath
            field_name: Field name to reference in JSONPath

        Returns:
            Validated JSONPath expression (e.g., "$.agent_id.outputs.field_name")

        Raises:
            ValueError: If agent_id or field_name contains invalid characters
                       or if the resulting JSONPath expression is malformed
        """
        # Sanitize identifiers - only allow alphanumeric and underscore
        # This prevents JSONPath injection and ensures valid syntax
        if not re.match(r"^[a-zA-Z0-9_]+$", agent_id):
            raise ValueError(
                f"Invalid agent_id for JSONPath: '{agent_id}'. "
                f"Only alphanumeric characters and underscores are allowed.",
            )

        if not re.match(r"^[a-zA-Z0-9_]+$", field_name):
            raise ValueError(
                f"Invalid field_name for JSONPath: '{field_name}'. "
                f"Only alphanumeric characters and underscores are allowed.",
            )

        # Build JSONPath expression
        path_str = f"$.{agent_id}.outputs.{field_name}"

        # Validate JSONPath syntax using jsonpath_ng
        try:
            jsonpath_ng.parse(path_str)
        except Exception as e:
            raise ValueError(f"Invalid JSONPath expression '{path_str}': {e}") from e

        return path_str

    def _build_field_mappings(
        self,
        agent: AgentEntry,
        all_prev_agents: list[AgentEntry],
        filled_inputs: dict[str, AgentInputs],
        agent_id: str,
    ) -> dict[str, str]:
        """
        Build io_map for an agent using all upstream agents.

        Rules:
        - If ANY explicit mapping exists in registry → use io_map, remove input.
        - Else if exact name match exists in ANY upstream agent → use io_map, remove input.
        - Otherwise → keep in inputs (LLM fallback).
        """

        io_map = {}
        inputs = filled_inputs.get(agent_id, {})

        # Iterate through each field of the current agent
        for field in agent.input_schema.fields:
            mapped = False
            source_field = None
            source_agent_id = None

            # 1️⃣ FIRST: Check explicit mapping across ALL upstream agents
            for prev_agent in reversed(all_prev_agents):  # nearest first
                prev_id = prev_agent.agent_id
                mapping = self.mapping_registry.get_mapping(prev_id, agent_id)
                prev_outputs = {f.name for f in prev_agent.output_schema.fields}

                if mapping and field.name in mapping:
                    src = mapping[field.name]
                    if src in prev_outputs:
                        mapped = True
                        source_field = src
                        source_agent_id = prev_id
                        break

            # 2️⃣ SECOND: Exact name matching across ALL upstream agents
            if not mapped:
                for prev_agent in reversed(all_prev_agents):
                    prev_id = prev_agent.agent_id
                    prev_outputs = {f.name for f in prev_agent.output_schema.fields}

                    if field.name in prev_outputs:
                        mapped = True
                        source_field = field.name
                        source_agent_id = prev_id
                        break

            # 3️⃣ Build io_map if found
            if mapped:
                # Remove LLM prefilled value
                if field.name in inputs:
                    del inputs[field.name]

                io_map[field.name] = self._build_and_validate_jsonpath(
                    source_agent_id,
                    source_field,
                )
                continue

            # 4️⃣ Fallback to inputs (LLM) — do nothing
            # Called only if not mapped
            logger.debug(f"{agent_id}.{field.name} remains in inputs")

        return io_map

    def build(self, plan: WorkflowPlan, filled_inputs: dict[str, AgentInputs]) -> WorkflowFormat:
        """
        Build WorkflowFormat from plan with io_map for runtime data flow.

        Args:
            plan: WorkflowPlan from LLM
            filled_inputs: Pre-filled inputs per agent {agent_id: {field: value}}
                          (may be modified in-place to remove auto-mapped fields)

        Returns:
            WorkflowFormat ready for execution
        """
        nodes = []
        edges = []
        agents = plan.suggested_agents

        if not agents:
            logger.warning("No agents in workflow plan")
            return WorkflowFormat(
                workflow_type=WORKFLOW_TYPE,
                version=WORKFLOW_FORMAT_VERSION,
                nodes=[],
                edges=[],
            )

        # Build nodes with io_map for sequential data flow
        for i, agent_suggestion in enumerate(agents):
            agent_id = agent_suggestion.agent_id
            agent = self.registry.get_agent(agent_id)

            if not agent:
                logger.warning(f"Agent {agent_id} not in registry, skipping")
                continue

            # Build io_map for runtime data flow from previous agent
            # Build list of all previous agents for inheritance
            all_prev_agents = [
                self.registry.get_agent(a.agent_id) for a in agents[:i] if self.registry.get_agent(a.agent_id)
            ]

            io_map = self._build_field_mappings(
                agent,
                all_prev_agents,
                filled_inputs,
                agent_id,
            )

            # Get filled inputs for this agent AFTER mapping (auto-mapped fields removed)
            inputs = filled_inputs.get(agent_id, {})

            # Create node
            node = WorkflowNode(
                type=agent_id,
                input=WorkflowNodeIO(fields=[{k: v} for k, v in inputs.items()]),
                output=WorkflowNodeIO(fields=[]),  # Runtime fills this
                io_map=io_map if io_map else None,
            )
            nodes.append(node)

        # Build sequential edges
        if nodes:
            edges.append(WorkflowEdge(from_node="START", to_node=nodes[0].type))
            for i in range(len(nodes) - 1):
                edges.append(WorkflowEdge(from_node=nodes[i].type, to_node=nodes[i + 1].type))
            edges.append(WorkflowEdge(from_node=nodes[-1].type, to_node="END"))

        return WorkflowFormat(
            workflow_type=WORKFLOW_TYPE,
            version=WORKFLOW_FORMAT_VERSION,
            nodes=nodes,
            edges=edges,
            output=nodes[-1].output if nodes else None,
        )

    def check_missing_agents(self, plan: WorkflowPlan) -> list[str]:
        """
        Simple validation: check if all agents exist in registry.

        Returns:
            List of missing agent IDs (empty if all exist)
        """
        missing = []
        for agent_suggestion in plan.suggested_agents:
            if not self.registry.get_agent(agent_suggestion.agent_id):
                missing.append(agent_suggestion.agent_id)
        return missing

    def identify_unmapped_fields(
        self,
        plan: WorkflowPlan,
        filled_inputs: dict[str, AgentInputs],
    ) -> list[UnmappedFieldInfo]:
        """
        Identify fields that need LLM-based mapping.

        Returns list of unmapped field info for LLM generation:
        [
            {
                "source_agent_id": str,
                "target_agent_id": str,
                "unmapped_fields": List[str]  # Target field names
            }
        ]
        """
        unmapped = []
        agents = plan.suggested_agents

        for i in range(1, len(agents)):
            prev_agent_id = agents[i - 1].agent_id
            agent_id = agents[i].agent_id

            prev_agent = self.registry.get_agent(prev_agent_id)
            agent = self.registry.get_agent(agent_id)

            if not prev_agent or not agent:
                continue

            inputs = filled_inputs.get(agent_id, {})
            mapping = self.mapping_registry.get_mapping(prev_agent_id, agent_id)
            source_fields = {f.name for f in prev_agent.output_schema.fields}

            fields_needing_mapping = []

            for field in agent.input_schema.fields:
                if field.required and field.name not in inputs:
                    # Check if mapping exists
                    has_mapping = (mapping and field.name in mapping) or (field.name in source_fields)

                    if not has_mapping:
                        fields_needing_mapping.append(field.name)

            if fields_needing_mapping:
                unmapped.append(
                    {
                        "source_agent_id": prev_agent_id,
                        "target_agent_id": agent_id,
                        "unmapped_fields": fields_needing_mapping,
                    },
                )

        return unmapped
