"""
Refactored LLM-based Workflow Planner for AKD Research Tasks

This module provides intelligent workflow planning using reasoning LLMs with
improved architecture, caching, and error handling.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from akd._base import InputSchema, OutputSchema
from .core import BasePlannerComponent, PlannerServiceManager, CacheManager


class WorkflowPlanningInput(InputSchema):
    """Input schema for workflow planning"""

    research_query: str = Field(
        ..., description="The research question or task description"
    )
    requirements: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional requirements and constraints"
    )
    available_agents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of available agents with their capabilities",
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context for planning"
    )


class WorkflowNode(BaseModel):
    """A node in the workflow plan"""

    node_id: str = Field(..., description="Unique identifier for the node")
    agent_name: str = Field(..., description="Name of the agent to use")
    description: str = Field(..., description="Description of what this node does")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input parameters")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Agent configuration"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Node IDs this node depends on"
    )
    estimated_duration: Optional[int] = Field(
        None, description="Estimated duration in seconds"
    )
    confidence: float = Field(0.8, description="Confidence in this node selection")


class WorkflowEdge(BaseModel):
    """An edge in the workflow plan"""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    data_mapping: Optional[Dict[str, str]] = Field(
        None, description="Optional data mapping between nodes"
    )
    condition: Optional[str] = Field(
        None, description="Optional condition for this edge"
    )


class WorkflowPlanOutput(OutputSchema):
    """Output schema for workflow planning"""

    workflow_plan: Dict[str, Any] = Field(
        ..., description="The generated workflow plan"
    )
    reasoning_trace: str = Field(
        ..., description="LLM reasoning trace for transparency"
    )
    confidence_score: float = Field(..., description="Overall confidence in the plan")
    estimated_duration: int = Field(
        ..., description="Estimated total duration in seconds"
    )
    alternative_plans: List[Dict[str, Any]] = Field(
        default_factory=list, description="Alternative workflow plans"
    )
    planning_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the planning process"
    )


class WorkflowPlanner(BasePlannerComponent):
    """Refactored LLM-based workflow planner with improved architecture"""

    input_schema = WorkflowPlanningInput
    output_schema = WorkflowPlanOutput

    def __init__(self, service_manager: PlannerServiceManager):
        super().__init__(service_manager)

        # Planning prompt template
        self.planning_prompt = self._create_planning_prompt()

        # Cache for results
        self.cache = CacheManager(max_size=50, ttl_seconds=1800)  # 30 minutes

    def _create_planning_prompt(self) -> ChatPromptTemplate:
        """Create the planning prompt template"""
        system_message = """You are an expert AI research workflow planner. Your task is to create intelligent, efficient workflows for scientific research tasks using available agents.

CORE PRINCIPLES:
1. Design workflows that are scientifically rigorous and follow research best practices
2. Use agents efficiently - avoid unnecessary steps while ensuring completeness
3. Consider data flow between agents and potential mapping requirements
4. Provide clear reasoning for each planning decision
5. Estimate realistic durations and confidence levels

AVAILABLE AGENT TYPES:
- Query agents: Process and understand research queries
- Literature search agents: Find relevant papers and sources
- Extraction agents: Extract specific data from documents
- Analysis agents: Perform data analysis and synthesis
- Relevancy agents: Check relevance and quality of results
- Validation agents: Validate results and findings

WORKFLOW STRUCTURE:
- Create a directed graph of nodes (agents) and edges (data flow)
- Each node should have clear inputs, outputs, and dependencies
- Consider parallelization where possible
- Include data mapping between incompatible agent schemas
- Provide confidence scores for each decision

REASONING REQUIREMENTS:
- Explain your reasoning process step-by-step
- Consider multiple approaches and explain why you chose the final one
- Identify potential failure points and mitigation strategies
- Estimate durations based on typical research workflows"""

        human_message = """Research Query: {research_query}

Available Agents:
{available_agents}

Additional Requirements:
{requirements}

Context:
{context}

Please create a comprehensive workflow plan that:
1. Addresses the research query effectively
2. Uses available agents optimally
3. Handles data flow and mapping between agents
4. Provides clear reasoning for each decision
5. Includes confidence scores and duration estimates

Return your response as a JSON object with the following structure:
{{
    "workflow_plan": {{
        "nodes": [
            {{
                "node_id": "unique_id",
                "agent_name": "AgentName",
                "description": "What this node does",
                "inputs": {{}},
                "config": {{}},
                "dependencies": [],
                "estimated_duration": 60,
                "confidence": 0.9
            }}
        ],
        "edges": [
            {{
                "source": "node_id_1",
                "target": "node_id_2",
                "data_mapping": {{}},
                "condition": null
            }}
        ]
    }},
    "reasoning_trace": "Step-by-step reasoning process...",
    "confidence_score": 0.85,
    "estimated_duration": 300,
    "alternative_plans": [],
    "planning_metadata": {{
        "agents_considered": [],
        "complexity_level": "medium",
        "parallelization_opportunities": []
    }}
}}

Focus on creating an efficient, scientifically sound workflow that maximizes the use of available agents while ensuring research quality."""

        return ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", human_message)]
        )

    async def _execute(self, params: WorkflowPlanningInput) -> WorkflowPlanOutput:
        """Generate workflow plan using LLM reasoning"""
        
        # Check cache first
        cache_key = self._create_cache_key(params)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.services.metrics["cache_hits"] += 1
            return cached_result

        # Prepare agent information
        agent_info = self._format_agent_information(params.available_agents)

        # Format requirements and context
        requirements_str = json.dumps(params.requirements, indent=2)
        context_str = json.dumps(params.context, indent=2)

        # Create the planning prompt
        formatted_prompt = self.planning_prompt.format_prompt(
            research_query=params.research_query,
            available_agents=agent_info,
            requirements=requirements_str,
            context=context_str,
        )

        # Get LLM response with reasoning
        try:
            self.services.metrics["llm_calls"] += 1
            result = await self.services.llm_service.invoke_with_retry(
                formatted_prompt.to_messages()
            )

            # Validate and enhance the plan
            enhanced_plan = await self._enhance_workflow_plan(result, params)

            output = WorkflowPlanOutput(
                workflow_plan=enhanced_plan["workflow_plan"],
                reasoning_trace=enhanced_plan["reasoning_trace"],
                confidence_score=enhanced_plan["confidence_score"],
                estimated_duration=enhanced_plan["estimated_duration"],
                alternative_plans=enhanced_plan.get("alternative_plans", []),
                planning_metadata=enhanced_plan.get("planning_metadata", {}),
            )
            
            # Cache the result
            self.cache.set(cache_key, output)
            self.services.metrics["plans_generated"] += 1
            return output

        except Exception as e:
            # Fallback to rule-based planning if LLM fails
            fallback_plan = await self._create_fallback_plan(params)
            output = WorkflowPlanOutput(
                workflow_plan=fallback_plan["workflow_plan"],
                reasoning_trace=f"LLM planning failed ({str(e)}), using fallback rule-based planning",
                confidence_score=0.6,
                estimated_duration=fallback_plan["estimated_duration"],
                alternative_plans=[],
                planning_metadata={"fallback_used": True, "error": str(e)},
            )
            
            # Cache fallback result too
            self.cache.set(cache_key, output)
            return output

    async def _arun(self, params: WorkflowPlanningInput) -> WorkflowPlanOutput:
        """Legacy method for backward compatibility"""
        return await self.execute_with_error_handling(params)

    def _format_agent_information(self, available_agents: List[Dict[str, Any]]) -> str:
        """Format agent information for the LLM prompt"""

        agent_descriptions = []
        for agent in available_agents:
            description = f"""
Agent: {agent["agent_name"]}
- Description: {agent.get("description", "No description available")}
- Input Schema: {agent.get("input_schema", "Unknown")}
- Output Schema: {agent.get("output_schema", "Unknown")}
- Capabilities: {", ".join(agent.get("capabilities", []))}
- Domain: {agent.get("domain", "general")}
- Estimated Performance: {agent.get("performance_metrics", {}).get("estimated_runtime", "unknown")}
"""
            agent_descriptions.append(description.strip())

        return "\n\n".join(agent_descriptions)

    async def _enhance_workflow_plan(
        self, raw_plan: Dict[str, Any], params: WorkflowPlanningInput
    ) -> Dict[str, Any]:
        """Enhanced workflow plan validation and improvement"""

        # Validate node IDs are unique
        node_ids = [node["node_id"] for node in raw_plan["workflow_plan"]["nodes"]]
        if len(node_ids) != len(set(node_ids)):
            raw_plan["reasoning_trace"] += "\nNote: Fixed duplicate node IDs"
            for i, node in enumerate(raw_plan["workflow_plan"]["nodes"]):
                node["node_id"] = f"{node['node_id']}_{i}"

        # Validate agent names exist using registry
        available_agent_names = {
            agent["agent_name"] for agent in params.available_agents
        }
        for node in raw_plan["workflow_plan"]["nodes"]:
            if node["agent_name"] not in available_agent_names:
                # Find closest match or use fallback
                closest_agent = self._find_closest_agent(
                    node["agent_name"], available_agent_names
                )
                if closest_agent:
                    node["agent_name"] = closest_agent
                    raw_plan["reasoning_trace"] += f"\nNote: Replaced '{node['agent_name']}' with '{closest_agent}'"
                else:
                    # Remove invalid node
                    raw_plan["reasoning_trace"] += f"\nWarning: Removed invalid agent '{node['agent_name']}'"

        # Add automatic data mapping hints
        enhanced_plan = await self._add_data_mapping_hints(raw_plan, params)

        # Validate dependencies and fix cycles
        enhanced_plan = self._validate_dependencies(enhanced_plan)

        return enhanced_plan

    def _find_closest_agent(self, requested_agent: str, available_agents: set) -> Optional[str]:
        """Find the closest matching agent name"""

        # Use registry for better matching
        best_match = None
        best_score = 0

        for agent in available_agents:
            score = self._calculate_name_similarity(requested_agent, agent)
            if score > best_score:
                best_score = score
                best_match = agent

        return best_match if best_score > 0.5 else None

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two agent names"""

        # Simple similarity based on common substrings
        name1_lower = name1.lower()
        name2_lower = name2.lower()

        if name1_lower == name2_lower:
            return 1.0

        # Check for common words
        words1 = set(name1_lower.split())
        words2 = set(name2_lower.split())

        if words1 & words2:  # Common words
            return 0.8

        # Check for substring matches
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.6

        return 0.0

    async def _add_data_mapping_hints(
        self, plan: Dict[str, Any], params: WorkflowPlanningInput
    ) -> Dict[str, Any]:
        """Add data mapping hints based on agent schemas"""

        # Create agent schema lookup
        agent_schemas = {}
        for agent in params.available_agents:
            agent_schemas[agent["agent_name"]] = {
                "input_schema": agent.get("input_schema"),
                "output_schema": agent.get("output_schema"),
            }

        # Analyze edges for potential mapping needs
        for edge in plan["workflow_plan"]["edges"]:
            source_node = next(
                (
                    n
                    for n in plan["workflow_plan"]["nodes"]
                    if n["node_id"] == edge["source"]
                ),
                None,
            )
            target_node = next(
                (
                    n
                    for n in plan["workflow_plan"]["nodes"]
                    if n["node_id"] == edge["target"]
                ),
                None,
            )

            if source_node and target_node:
                source_schema = agent_schemas.get(source_node["agent_name"], {}).get(
                    "output_schema"
                )
                target_schema = agent_schemas.get(target_node["agent_name"], {}).get(
                    "input_schema"
                )

                # Add mapping hint if schemas are different
                if source_schema and target_schema and source_schema != target_schema:
                    edge["data_mapping"] = {
                        "source_schema": source_schema,
                        "target_schema": target_schema,
                        "mapping_required": True,
                    }

        return plan

    def _validate_dependencies(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix dependency cycles using improved algorithm"""

        nodes = plan["workflow_plan"]["nodes"]
        edges = plan["workflow_plan"]["edges"]

        # Create dependency graph
        dependencies = {}
        for node in nodes:
            dependencies[node["node_id"]] = node.get("dependencies", [])

        # Add edge-based dependencies
        for edge in edges:
            target_deps = dependencies.get(edge["target"], [])
            if edge["source"] not in target_deps:
                target_deps.append(edge["source"])
                dependencies[edge["target"]] = target_deps

        # Check and fix cycles using topological sort
        visited = set()
        rec_stack = set()

        def has_cycle(node_id):
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            rec_stack.add(node_id)

            for dep in dependencies.get(node_id, []):
                if has_cycle(dep):
                    return True

            rec_stack.remove(node_id)
            return False

        # Remove cycles if found
        for node_id in dependencies:
            if has_cycle(node_id):
                # Simple fix: remove the last dependency that creates cycle
                if dependencies[node_id]:
                    dependencies[node_id].pop()
                    plan["reasoning_trace"] += (
                        f"\nNote: Removed cyclic dependency in {node_id}"
                    )

        # Update node dependencies
        for node in nodes:
            node["dependencies"] = dependencies.get(node["node_id"], [])

        return plan

    async def _create_fallback_plan(
        self, params: WorkflowPlanningInput
    ) -> Dict[str, Any]:
        """Create an improved fallback plan when LLM fails"""

        nodes = []
        edges = []

        if params.available_agents:
            # Create a more intelligent linear workflow
            prev_node = None

            # Prioritize agents by type
            agent_priorities = {
                "query": 1,
                "search": 2,
                "extraction": 3,
                "analysis": 4,
                "validation": 5
            }

            # Sort agents by priority
            sorted_agents = sorted(
                params.available_agents[:3],  # Limit to 3 agents
                key=lambda x: min(
                    agent_priorities.get(keyword, 99) 
                    for keyword in agent_priorities.keys()
                    if keyword in x.get("agent_name", "").lower()
                )
            )

            for i, agent in enumerate(sorted_agents):
                node_id = f"node_{i}"
                node = {
                    "node_id": node_id,
                    "agent_name": agent["agent_name"],
                    "description": f"Process using {agent['agent_name']}",
                    "inputs": {"query": params.research_query},
                    "config": {},
                    "dependencies": [prev_node] if prev_node else [],
                    "estimated_duration": 60,
                    "confidence": 0.6,
                }
                nodes.append(node)

                if prev_node:
                    edges.append(
                        {
                            "source": prev_node,
                            "target": node_id,
                            "data_mapping": None,
                            "condition": None,
                        }
                    )

                prev_node = node_id

        return {
            "workflow_plan": {"nodes": nodes, "edges": edges},
            "estimated_duration": len(nodes) * 60,
        }

    async def validate_plan(self, workflow_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a workflow plan for consistency and feasibility"""
        return self.services.validation_service.validate_workflow_plan(workflow_plan)

    async def generate_alternative_plans(
        self, params: WorkflowPlanningInput, num_alternatives: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate alternative workflow plans using batch processing"""
        
        # Create batch requests for parallel processing
        batch_requests = []
        for i in range(num_alternatives):
            modified_params = WorkflowPlanningInput(
                research_query=params.research_query,
                requirements={
                    **params.requirements,
                    "alternative_approach": i + 1,
                    "prefer_different_strategy": True,
                },
                available_agents=params.available_agents,
                context=params.context,
            )
            
            agent_info = self._format_agent_information(modified_params.available_agents)
            requirements_str = json.dumps(modified_params.requirements, indent=2)
            context_str = json.dumps(modified_params.context, indent=2)
            
            formatted_prompt = self.planning_prompt.format_prompt(
                research_query=modified_params.research_query,
                available_agents=agent_info,
                requirements=requirements_str,
                context=context_str,
            )
            batch_requests.append(formatted_prompt.to_messages())
        
        # Process alternatives in parallel
        results = await self.services.llm_service.invoke_batch(batch_requests)
        
        alternatives = []
        for result in results:
            if not isinstance(result, Exception):
                try:
                    enhanced_plan = await self._enhance_workflow_plan(result, params)
                    alternatives.append(enhanced_plan["workflow_plan"])
                except Exception:
                    continue
        
        return alternatives

    def _create_cache_key(self, params: WorkflowPlanningInput) -> str:
        """Create cache key for workflow planning parameters"""
        import hashlib
        
        key_data = {
            "query": params.research_query,
            "requirements": params.requirements or {},
            "agents": [agent.get("agent_name", "") for agent in (params.available_agents or [])],
            "context": params.context or {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def clear_cache(self):
        """Clear the planning cache"""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache._cache),
            "max_size": self.cache.max_size,
            "ttl_seconds": self.cache.ttl_seconds
        }