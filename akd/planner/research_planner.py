"""
Unified Research Planner for AKD

This module provides a unified interface for intelligent research planning that combines
LLM-based workflow generation with the existing AKD agent ecosystem.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type
from datetime import datetime

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent
from ..configs.planner_config import PlannerConfig
from .workflow_planner import WorkflowPlanner, WorkflowPlanningInput, WorkflowPlanOutput
from .agent_analyzer import AgentAnalyzer, AgentCapability
from .langgraph_builder import LangGraphWorkflowBuilder
from .core import PlannerServiceManager, BasePlannerComponent


class ResearchPlanningInput(InputSchema):
    """Input schema for research planning"""
    research_query: str
    requirements: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    preferred_agents: Optional[List[str]] = None
    session_id: Optional[str] = None


class ResearchPlanningOutput(OutputSchema):
    """Output schema for research planning"""
    workflow_plan: Dict[str, Any]
    reasoning_trace: str
    confidence_score: float
    estimated_duration: int
    available_agents: List[Dict[str, Any]]
    agent_capabilities: List[Dict[str, Any]]
    planning_metadata: Dict[str, Any]
    execution_ready: bool


class ResearchPlanner(BasePlannerComponent):
    """Unified research planner combining LLM planning with AKD agent ecosystem"""
    
    input_schema = ResearchPlanningInput
    output_schema = ResearchPlanningOutput
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        config = config or PlannerConfig()
        service_manager = PlannerServiceManager(config)
        super().__init__(service_manager)
        
        # Initialize components with service manager
        self.workflow_planner = WorkflowPlanner(service_manager)
        self.agent_analyzer = AgentAnalyzer(config)  # Keep original for now
        self.workflow_builder = LangGraphWorkflowBuilder(config)
        
        # Agent discovery and caching
        self.discovered_agents: Dict[str, Type[BaseAgent]] = {}
        self.agent_capabilities: List[AgentCapability] = []
        self.capabilities_cache: Dict[str, AgentCapability] = {}
        
        # Initialize agent discovery
        if getattr(config, 'auto_discover_agents', True):
            self._discover_agents()
    
    def _discover_agents(self):
        """Discover available agents using existing discovery mechanism"""
        
        # Use the existing discovery from LangGraphWorkflowBuilder
        self.discovered_agents = self.workflow_builder.agent_registry
        
        # Convert to available agents format
        available_agents_info = self.workflow_builder.get_available_agents()
        
        # Store the discovered agents info
        self.available_agents_info = available_agents_info
    
    async def _execute(self, params: ResearchPlanningInput) -> ResearchPlanningOutput:
        """Execute research planning with LLM-based workflow generation"""
        
        # Analyze available agents if not already done
        if not self.agent_capabilities:
            await self._analyze_available_agents()
        
        # Prepare input for workflow planner
        workflow_input = WorkflowPlanningInput(
            research_query=params.research_query,
            requirements=params.requirements or {},
            available_agents=self._format_agents_for_planner(),
            context={
                "constraints": params.constraints or {},
                "preferred_agents": params.preferred_agents or [],
                "session_id": params.session_id
            }
        )
        
        # Generate workflow plan using LLM
        workflow_output = await self.workflow_planner.execute_with_error_handling(workflow_input)
        
        # Validate and enhance the plan
        enhanced_plan = await self._enhance_plan_with_agent_validation(workflow_output)
        
        # Check execution readiness
        execution_ready = await self._check_execution_readiness(enhanced_plan.workflow_plan)
        
        output = ResearchPlanningOutput(
            workflow_plan=enhanced_plan.workflow_plan,
            reasoning_trace=enhanced_plan.reasoning_trace,
            confidence_score=enhanced_plan.confidence_score,
            estimated_duration=enhanced_plan.estimated_duration,
            available_agents=self._format_agents_for_output(),
            agent_capabilities=[cap.model_dump() for cap in self.agent_capabilities],
            planning_metadata={
                **enhanced_plan.planning_metadata,
                "planning_timestamp": datetime.now().isoformat(),
                "total_agents_available": len(self.discovered_agents),
                "agents_used": len(enhanced_plan.workflow_plan.get("nodes", [])),
                "execution_ready": execution_ready
            },
            execution_ready=execution_ready
        )
        
        return output
    
    async def _arun(self, params: ResearchPlanningInput) -> ResearchPlanningOutput:
        """Legacy method for backward compatibility"""
        return await self.execute_with_error_handling(params)
    
    async def _analyze_available_agents(self):
        """Analyze all available agents to create capability profiles"""
        
        # Convert discovered agents to list of agent classes
        agent_classes = list(self.discovered_agents.values())
        
        # Analyze agents in parallel
        self.agent_capabilities = await self.agent_analyzer.analyze_agent_batch(agent_classes)
        
        # Create capabilities cache
        self.capabilities_cache = {
            cap.agent_name: cap for cap in self.agent_capabilities
        }
    
    def _format_agents_for_planner(self) -> List[Dict[str, Any]]:
        """Format agent capabilities for the workflow planner"""
        
        formatted_agents = []
        
        for capability in self.agent_capabilities:
            agent_info = {
                "agent_name": capability.agent_name,
                "description": capability.description,
                "domain": capability.domain,
                "capabilities": capability.capabilities,
                "input_schema": capability.input_schema,
                "output_schema": capability.output_schema,
                "input_fields": capability.input_fields,
                "output_fields": capability.output_fields,
                "performance_metrics": capability.performance_metrics,
                "suggested_upstream": capability.suggested_upstream,
                "suggested_downstream": capability.suggested_downstream
            }
            formatted_agents.append(agent_info)
        
        return formatted_agents
    
    def _format_agents_for_output(self) -> List[Dict[str, Any]]:
        """Format agents for output response"""
        
        formatted_agents = []
        
        for agent_name, agent_info in self.available_agents_info.items():
            formatted_agent = {
                "agent_name": agent_name,
                "class_name": agent_info.get("class_name", agent_name),
                "module": agent_info.get("module", "unknown"),
                "has_input_schema": agent_info.get("has_input_schema", False),
                "has_output_schema": agent_info.get("has_output_schema", False),
                "input_schema": agent_info.get("input_schema"),
                "output_schema": agent_info.get("output_schema")
            }
            
            # Add capability info if available
            if agent_name in self.capabilities_cache:
                capability = self.capabilities_cache[agent_name]
                formatted_agent.update({
                    "description": capability.description,
                    "domain": capability.domain,
                    "capabilities": capability.capabilities,
                    "performance_metrics": capability.performance_metrics
                })
            
            formatted_agents.append(formatted_agent)
        
        return formatted_agents
    
    async def _enhance_plan_with_agent_validation(self, workflow_output: WorkflowPlanOutput) -> WorkflowPlanOutput:
        """Enhance the workflow plan with agent validation and compatibility checks"""
        
        plan = workflow_output.workflow_plan
        nodes = plan.get("nodes", [])
        edges = plan.get("edges", [])
        
        # Validate agent availability
        enhanced_nodes = []
        for node in nodes:
            agent_name = node["agent_name"]
            
            # Check if agent exists
            if agent_name not in self.discovered_agents:
                # Find similar agent
                similar_agent = self._find_similar_agent(agent_name)
                if similar_agent:
                    node["agent_name"] = similar_agent
                    workflow_output.reasoning_trace += f"\nNote: Replaced unavailable agent '{agent_name}' with '{similar_agent}'"
                else:
                    # Remove node if no similar agent found
                    workflow_output.reasoning_trace += f"\nWarning: Removed unavailable agent '{agent_name}'"
                    continue
            
            # Add capability info to node
            if node["agent_name"] in self.capabilities_cache:
                capability = self.capabilities_cache[node["agent_name"]]
                node["capability_info"] = {
                    "domain": capability.domain,
                    "capabilities": capability.capabilities,
                    "performance_metrics": capability.performance_metrics
                }
            
            enhanced_nodes.append(node)
        
        # Update plan with enhanced nodes
        plan["nodes"] = enhanced_nodes
        
        # Validate and enhance edges
        enhanced_edges = await self._enhance_edges_with_compatibility(edges, enhanced_nodes)
        plan["edges"] = enhanced_edges
        
        # Update workflow output
        workflow_output.workflow_plan = plan
        
        return workflow_output
    
    def _find_similar_agent(self, requested_agent: str) -> Optional[str]:
        """Find similar agent when requested agent is not available"""
        
        best_match = None
        best_score = 0
        
        for agent_name in self.discovered_agents.keys():
            score = self._calculate_agent_similarity(requested_agent, agent_name)
            if score > best_score:
                best_score = score
                best_match = agent_name
        
        return best_match if best_score > 0.5 else None
    
    def _calculate_agent_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between agent names"""
        
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        if name1_lower == name2_lower:
            return 1.0
        
        # Check for common words
        words1 = set(name1_lower.split())
        words2 = set(name2_lower.split())
        
        common_words = words1 & words2
        if common_words:
            return len(common_words) / max(len(words1), len(words2))
        
        # Check for substring matches
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.6
        
        return 0.0
    
    async def _enhance_edges_with_compatibility(self, edges: List[Dict], nodes: List[Dict]) -> List[Dict]:
        """Enhance edges with compatibility analysis"""
        
        enhanced_edges = []
        
        for edge in edges:
            source_node = next((n for n in nodes if n["node_id"] == edge["source"]), None)
            target_node = next((n for n in nodes if n["node_id"] == edge["target"]), None)
            
            if source_node and target_node:
                # Get capability info
                source_capability = self.capabilities_cache.get(source_node["agent_name"])
                target_capability = self.capabilities_cache.get(target_node["agent_name"])
                
                if source_capability and target_capability:
                    # Analyze compatibility
                    compatibility = await self.agent_analyzer.analyze_compatibility(
                        source_capability, target_capability
                    )
                    
                    # Add compatibility info to edge
                    edge["compatibility"] = {
                        "score": compatibility.compatibility_score,
                        "mapping_required": compatibility.data_mapping_required,
                        "mapping_suggestions": compatibility.mapping_suggestions
                    }
                
                enhanced_edges.append(edge)
        
        return enhanced_edges
    
    async def _check_execution_readiness(self, workflow_plan: Dict[str, Any]) -> bool:
        """Check if the workflow plan is ready for execution"""
        
        nodes = workflow_plan.get("nodes", [])
        edges = workflow_plan.get("edges", [])
        
        # Check if all agents are available
        for node in nodes:
            if node["agent_name"] not in self.discovered_agents:
                return False
        
        # Check if workflow has at least one node
        if not nodes:
            return False
        
        # Check for basic workflow structure
        if len(nodes) > 1 and not edges:
            return False  # Multi-node workflow needs edges
        
        # Validate node IDs in edges
        node_ids = {node["node_id"] for node in nodes}
        for edge in edges:
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                return False
        
        return True
    
    async def execute_workflow(
        self, 
        workflow_plan: Dict[str, Any], 
        session_id: str = None
    ) -> Dict[str, Any]:
        """Execute the generated workflow plan"""
        
        if not await self._check_execution_readiness(workflow_plan):
            return {
                "status": "error",
                "error": "Workflow plan is not ready for execution",
                "workflow_plan": workflow_plan
            }
        
        # Create agent profiles for execution
        agent_profiles = {}
        for node in workflow_plan.get("nodes", []):
            agent_name = node["agent_name"]
            if agent_name in self.available_agents_info:
                agent_info = self.available_agents_info[agent_name]
                agent_profiles[agent_name] = {
                    "description": node.get("description", f"Agent {agent_name}"),
                    "input_schema": agent_info.get("input_schema"),
                    "output_schema": agent_info.get("output_schema")
                }
        
        # Execute using existing workflow builder
        result = await self.workflow_builder.execute_workflow(
            research_query=workflow_plan.get("research_query", "Generated workflow"),
            workflow_plan=workflow_plan,
            agent_profiles=agent_profiles,
            session_id=session_id or "generated_workflow"
        )
        
        return result
    
    async def stream_workflow_execution(
        self, 
        workflow_plan: Dict[str, Any], 
        session_id: str = None
    ):
        """Stream workflow execution for real-time updates"""
        
        # Create agent profiles
        agent_profiles = {}
        for node in workflow_plan.get("nodes", []):
            agent_name = node["agent_name"]
            if agent_name in self.available_agents_info:
                agent_info = self.available_agents_info[agent_name]
                agent_profiles[agent_name] = {
                    "description": node.get("description", f"Agent {agent_name}"),
                    "input_schema": agent_info.get("input_schema"),
                    "output_schema": agent_info.get("output_schema")
                }
        
        # Stream execution
        async for event in self.workflow_builder.stream_workflow_execution(
            research_query=workflow_plan.get("research_query", "Generated workflow"),
            workflow_plan=workflow_plan,
            agent_profiles=agent_profiles,
            session_id=session_id or "generated_workflow"
        ):
            yield event
    
    async def generate_alternative_plans(
        self, 
        research_query: str, 
        requirements: Optional[Dict[str, Any]] = None,
        num_alternatives: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate alternative workflow plans"""
        
        planning_input = WorkflowPlanningInput(
            research_query=research_query,
            requirements=requirements or {},
            available_agents=self._format_agents_for_planner(),
            context={"num_alternatives": num_alternatives}
        )
        
        alternatives = await self.workflow_planner.generate_alternative_plans(
            planning_input, num_alternatives
        )
        
        return alternatives
    
    async def suggest_improvements(self, workflow_plan: Dict[str, Any]) -> List[str]:
        """Suggest improvements to a workflow plan"""
        
        return await self.agent_analyzer.suggest_workflow_improvements(
            workflow_plan, self.agent_capabilities
        )
    
    async def validate_plan(self, workflow_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a workflow plan for consistency and feasibility"""
        
        return await self.workflow_planner.validate_plan(workflow_plan)
    
    def get_available_agents(self) -> Dict[str, Any]:
        """Get information about available agents"""
        
        return self.available_agents_info
    
    def get_agent_capabilities(self) -> List[AgentCapability]:
        """Get detailed agent capabilities"""
        
        return self.agent_capabilities
    
    async def find_workflow_path(
        self, 
        start_agent: str, 
        end_agent: str, 
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """Find optimal path between two agents"""
        
        return await self.agent_analyzer.find_workflow_path(
            start_agent, end_agent, self.agent_capabilities, max_depth
        )
    
    def register_agent(self, agent_name: str, agent_class: Type[BaseAgent]):
        """Register a new agent"""
        
        self.workflow_builder.register_agent(agent_name, agent_class)
        self.discovered_agents[agent_name] = agent_class
        
        # Update available agents info
        self.available_agents_info = self.workflow_builder.get_available_agents()
        
        # Clear capabilities cache to force re-analysis
        self.agent_capabilities = []
        self.capabilities_cache = {}
    
    async def create_simple_workflow(
        self,
        research_query: str,
        agent_sequence: List[str],
        session_id: str = None
    ) -> Dict[str, Any]:
        """Create a simple linear workflow from agent sequence"""
        
        # Validate agents exist
        for agent_name in agent_sequence:
            if agent_name not in self.discovered_agents:
                raise ValueError(f"Agent '{agent_name}' not found in available agents")
        
        # Create linear workflow
        nodes = []
        edges = []
        
        for i, agent_name in enumerate(agent_sequence):
            node = {
                "node_id": f"node_{i}",
                "agent_name": agent_name,
                "description": f"Execute {agent_name}",
                "inputs": {"query": research_query},
                "config": {},
                "dependencies": [f"node_{i-1}"] if i > 0 else [],
                "estimated_duration": 60,
                "confidence": 0.8
            }
            nodes.append(node)
            
            if i > 0:
                edges.append({
                    "source": f"node_{i-1}",
                    "target": f"node_{i}",
                    "data_mapping": None,
                    "condition": None
                })
        
        workflow_plan = {
            "research_query": research_query,
            "nodes": nodes,
            "edges": edges
        }
        
        # Execute the simple workflow
        return await self.execute_workflow(workflow_plan, session_id)