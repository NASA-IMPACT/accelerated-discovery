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

from .enrich_schema import enrich_workflow_schema
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

    def _build_and_validate_jsonpath(self, node_id: str, field_name: str) -> str:
        """
        Build and validate JSONPath expression for field mapping.

        Args:
            node_id: Unique node identifier to reference in JSONPath (e.g., my_agent_0)
            field_name: Field name to reference in JSONPath

        Returns:
            Validated JSONPath expression (e.g., "$.node_id.outputs.field_name")

        Raises:
            ValueError: If node_id or field_name contains invalid characters
                       or if the resulting JSONPath expression is malformed
        """
        # Sanitize identifiers - only allow alphanumeric and underscore
        # This prevents JSONPath injection and ensures valid syntax
        if not re.match(r"^[a-zA-Z0-9_]+$", node_id):
            raise ValueError(
                f"Invalid node_id for JSONPath: '{node_id}'. Only alphanumeric characters and underscores are allowed.",
            )

        if not re.match(r"^[a-zA-Z0-9_]+$", field_name):
            raise ValueError(
                f"Invalid field_name for JSONPath: '{field_name}'. "
                f"Only alphanumeric characters and underscores are allowed.",
            )

        # Build JSONPath expression
        path_str = f"$.{node_id}.outputs.{field_name}"

        # Validate JSONPath syntax using jsonpath_ng
        try:
            jsonpath_ng.parse(path_str)
        except Exception as e:
            raise ValueError(f"Invalid JSONPath expression '{path_str}': {e}") from e

        return path_str

    def _build_field_mappings(
        self,
        agent: AgentEntry,
        prev_agent: AgentEntry,
        filled_inputs: dict[str, AgentInputs],
        agent_id: str,
        prev_agent_id: str,
        node_id: str,
        prev_node_id: str,
    ) -> dict[str, str]:
        """
        Build io_map for runtime data flow from previous agent to current agent.

        Uses three-tier field mapping strategy:
        1. Explicit/LLM-approved mapping from registry (Priority 1 & 2)
        2. Exact name match fallback (Priority 3)
        3. No mapping - logs warning

        Args:
            agent: Current agent instance
            prev_agent: Previous agent instance
            filled_inputs: Pre-filled inputs for all agents
            agent_id: Current agent type ID (for registry lookup)
            prev_agent_id: Previous agent type ID (for registry lookup)
            node_id: Current node unique ID (for JSONPath)
            prev_node_id: Previous node unique ID (for JSONPath)

        Returns:
            Dictionary mapping current agent field names to JSONPath expressions
            (e.g., {"field_name": "$.prev_node_id.outputs.source_field"})
        """
        io_map = {}
        inputs = filled_inputs.get(agent_id, {})

        # Get available source fields for validation
        source_field_names = {f.name for f in prev_agent.output_schema.fields}

        # Map each required input using three-tier strategy
        for field in agent.input_schema.fields:
            if field.required and field.name not in inputs:
                # Get field mapping from registry (uses agent types, not node IDs)
                # Only use mappings with confidence >= 0.8 to avoid bad auto-approved mappings
                mapping = self.mapping_registry.get_mapping(
                    prev_agent_id,
                    agent_id,
                    min_confidence=0.8,
                )

                if mapping and field.name in mapping:
                    # Priority 1 or 2: Use explicit or LLM-approved mapping
                    source_field = mapping[field.name]

                    # VALIDATION: Check source field exists in prev_agent outputs
                    if source_field not in source_field_names:
                        logger.error(
                            f"Invalid mapping: {node_id}.{field.name} <- "
                            f"{prev_node_id}.{source_field} "
                            f"(source field '{source_field}' does not exist in {prev_agent_id} output schema)",
                        )
                        if self.debug:
                            raise ValueError(
                                f"Mapping references non-existent output field: '{source_field}' "
                                f"not in {prev_agent_id}.outputs. "
                                f"Available fields: {source_field_names}",
                            )
                        # Skip invalid mapping in production mode
                        logger.warning(f"Skipping invalid mapping for {node_id}.{field.name}")
                        continue

                    # Build and validate JSONPath expression using node ID
                    io_map[field.name] = self._build_and_validate_jsonpath(prev_node_id, source_field)
                    logger.debug(
                        f"Mapped {node_id}.{field.name} <- {prev_node_id}.{source_field} (from registry, validated)",
                    )
                else:
                    # Priority 3: Exact name match fallback
                    if field.name in source_field_names:
                        # Build and validate JSONPath expression using node ID
                        io_map[field.name] = self._build_and_validate_jsonpath(prev_node_id, field.name)
                        logger.debug(
                            f"Mapped {node_id}.{field.name} <- {prev_node_id}.{field.name} (exact match, validated)",
                        )
                    else:
                        # No mapping available - will need LLM generation
                        logger.warning(
                            f"No mapping found for {node_id}.{field.name} from "
                            f"{prev_node_id}. Field will be missing unless LLM mapping "
                            f"is generated.",
                        )

        return io_map

    def build(self, plan: WorkflowPlan, filled_inputs: dict[str, AgentInputs]) -> WorkflowFormat:
        """
        Build WorkflowFormat from plan with io_map for runtime data flow.

        Args:
            plan: WorkflowPlan from LLM
            filled_inputs: Pre-filled inputs per agent {agent_id: {field: value}}

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

        # Track agent type counts for generating unique node IDs
        agent_counts: dict[str, int] = {}
        # Map from index to node_id for edge creation
        node_ids: list[str] = []

        # Build nodes with io_map for sequential data flow
        for i, agent_suggestion in enumerate(agents):
            agent_id = agent_suggestion.agent_id
            agent = self.registry.get_agent(agent_id)

            if not agent:
                logger.warning(f"Agent {agent_id} not in registry, skipping")
                continue

            # Generate unique node ID: {agent_id}_{count} (starting from 1)
            count = agent_counts.get(agent_id, 1)
            node_id = f"{agent_id}_{count}"
            agent_counts[agent_id] = count + 1
            node_ids.append(node_id)

            # Get filled inputs for this agent (try node_id first, then agent_id for backward compat)
            inputs = filled_inputs.get(node_id, filled_inputs.get(agent_id, {}))

            # Build io_map for runtime data flow from previous agent
            io_map = {}
            if i > 0:
                prev_agent_id = agents[i - 1].agent_id
                prev_agent = self.registry.get_agent(prev_agent_id)
                prev_node_id = node_ids[i - 1]

                if not prev_agent:
                    logger.warning(f"Previous agent {prev_agent_id} not in registry")
                else:
                    io_map = self._build_field_mappings(
                        agent,
                        prev_agent,
                        filled_inputs,
                        agent_id,
                        prev_agent_id,
                        node_id,
                        prev_node_id,
                    )

            # Create node with unique id and type
            node = WorkflowNode(
                id=node_id,
                type_=agent_id,
                input=WorkflowNodeIO(fields=[{k: v} for k, v in inputs.items()]),
                output=WorkflowNodeIO(fields=[]),  # Runtime fills this
                io_map=io_map if io_map else None,
            )
            nodes.append(node)

        # Build sequential edges using node IDs
        if nodes:
            edges.append(WorkflowEdge(from_node="START", to_node=nodes[0].id))
            for i in range(len(nodes) - 1):
                edges.append(WorkflowEdge(from_node=nodes[i].id, to_node=nodes[i + 1].id))
            edges.append(WorkflowEdge(from_node=nodes[-1].id, to_node="END"))

        workflow = WorkflowFormat(
            workflow_type=WORKFLOW_TYPE,
            version=WORKFLOW_FORMAT_VERSION,
            nodes=nodes,
            edges=edges,
            output=nodes[-1].output if nodes else None,
        )

        # Enrich node inputs with full schema metadata
        enriched_dict = enrich_workflow_schema(workflow.model_dump(by_alias=True), registry=self.registry)
        return WorkflowFormat(**enriched_dict)

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
                "unmapped_fields": list[str]  # Target field names
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
