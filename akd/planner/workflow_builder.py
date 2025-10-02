"""
Clean workflow builder for AKD planner.

Builds executable WorkflowFormat from WorkflowPlan with runtime data flow using io_map.
Uses three-tier field mapping strategy:
1. Explicit mappings (human-defined)
2. Exact name match (automatic)
3. LLM-generated mappings (intelligent fallback)
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from .format_builder import WorkflowFormat, WorkflowNode, WorkflowEdge, WorkflowNodeIO
from .registry import AgentRegistry
from .field_mapping_registry import FieldMappingRegistry

if TYPE_CHECKING:
    from .llm_planner import WorkflowPlan


class WorkflowBuilder:
    """Builds WorkflowFormat with io_map for runtime data flow using intelligent field mapping."""

    def __init__(
        self,
        registry: AgentRegistry,
        mapping_registry: Optional[FieldMappingRegistry] = None
    ):
        """
        Initialize workflow builder.

        Args:
            registry: Agent registry for schema access
            mapping_registry: Field mapping registry (creates default if None)
        """
        self.registry = registry
        self.mapping_registry = mapping_registry or FieldMappingRegistry()

    def build(self, plan: "WorkflowPlan", filled_inputs: Dict[str, Dict[str, Any]]) -> WorkflowFormat:
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
                workflow_type="AKDResearchWorkflow",
                version="1.0.0",
                nodes=[],
                edges=[]
            )

        # Build nodes with io_map for sequential data flow
        for i, agent_suggestion in enumerate(agents):
            agent_id = agent_suggestion.agent_id
            agent = self.registry.get_agent(agent_id)

            if not agent:
                logger.warning(f"Agent {agent_id} not in registry, skipping")
                continue

            # Get filled inputs for this agent
            inputs = filled_inputs.get(agent_id, {})

            # Build io_map for runtime data flow from previous agent
            io_map = {}
            if i > 0:
                prev_agent_id = agents[i - 1].agent_id
                prev_agent = self.registry.get_agent(prev_agent_id)

                if not prev_agent:
                    logger.warning(f"Previous agent {prev_agent_id} not in registry")
                else:
                    # Map each required input using three-tier strategy
                    for field in agent.input_schema.fields:
                        if field.required and field.name not in inputs:
                            # Get field mapping from registry
                            mapping = self.mapping_registry.get_mapping(
                                prev_agent_id,
                                agent_id
                            )

                            if mapping and field.name in mapping:
                                # Priority 1 or 2: Use explicit or LLM-approved mapping
                                source_field = mapping[field.name]
                                io_map[field.name] = f"$.{prev_agent_id}.outputs.{source_field}"
                                logger.debug(
                                    f"Mapped {agent_id}.{field.name} <- "
                                    f"{prev_agent_id}.{source_field} (from registry)"
                                )
                            else:
                                # Priority 3: Exact name match fallback
                                source_fields = {f.name for f in prev_agent.output_schema.fields}
                                if field.name in source_fields:
                                    io_map[field.name] = f"$.{prev_agent_id}.outputs.{field.name}"
                                    logger.debug(
                                        f"Mapped {agent_id}.{field.name} <- "
                                        f"{prev_agent_id}.{field.name} (exact match)"
                                    )
                                else:
                                    # No mapping available - will need LLM generation
                                    logger.warning(
                                        f"No mapping found for {agent_id}.{field.name} from "
                                        f"{prev_agent_id}. Field will be missing unless LLM mapping "
                                        f"is generated."
                                    )

            # Create node
            node = WorkflowNode(
                type=agent_id,
                input=WorkflowNodeIO(fields=[{k: v} for k, v in inputs.items()]),
                output=WorkflowNodeIO(fields=[]),  # Runtime fills this
                io_map=io_map if io_map else None
            )
            nodes.append(node)

        # Build sequential edges
        if nodes:
            edges.append(WorkflowEdge(from_node="START", to_node=nodes[0].type))
            for i in range(len(nodes) - 1):
                edges.append(WorkflowEdge(from_node=nodes[i].type, to_node=nodes[i + 1].type))
            edges.append(WorkflowEdge(from_node=nodes[-1].type, to_node="END"))

        return WorkflowFormat(
            workflow_type="AKDResearchWorkflow",
            version="1.0.0",
            nodes=nodes,
            edges=edges,
            output=nodes[-1].output if nodes else None
        )

    def check_agents_exist(self, plan: "WorkflowPlan") -> List[str]:
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
        plan: "WorkflowPlan",
        filled_inputs: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
                    has_mapping = (
                        (mapping and field.name in mapping) or
                        (field.name in source_fields)
                    )

                    if not has_mapping:
                        fields_needing_mapping.append(field.name)

            if fields_needing_mapping:
                unmapped.append({
                    "source_agent_id": prev_agent_id,
                    "target_agent_id": agent_id,
                    "unmapped_fields": fields_needing_mapping
                })

        return unmapped
