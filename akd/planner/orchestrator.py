"""
Unified Planner Orchestrator

This module provides a unified interface for the planner system, simplifying
the interaction between different components and providing a clean API.
"""

from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime

from pydantic import BaseModel, Field

from akd._base import InputSchema, OutputSchema
from ..configs.planner_config import PlannerConfig
from .core import PlannerServiceManager, BasePlannerComponent, PlannerError
from .research_planner import ResearchPlanner, ResearchPlanningInput, ResearchPlanningOutput
from .workflow_planner import WorkflowPlanner, WorkflowPlanningInput, WorkflowPlanOutput
from .agent_analyzer import AgentAnalyzer, AgentCapability
from .langgraph_builder import LangGraphWorkflowBuilder


class PlannerRequest(InputSchema):
    """Unified request for planner operations"""
    
    research_query: str = Field(..., description="Research query or task")
    requirements: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Requirements and constraints"
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context"
    )
    preferred_agents: Optional[List[str]] = Field(
        default=None, description="Preferred agent names"
    )
    session_id: Optional[str] = Field(
        default=None, description="Session identifier"
    )
    execution_mode: str = Field(
        default="plan_and_execute", 
        description="Execution mode: plan_only, execute_only, or plan_and_execute"
    )


class PlannerResponse(OutputSchema):
    """Unified response from planner operations"""
    
    workflow_plan: Dict[str, Any] = Field(..., description="Generated workflow plan")
    execution_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Execution results if executed"
    )
    reasoning_trace: str = Field(..., description="Planning reasoning trace")
    confidence_score: float = Field(..., description="Overall confidence score")
    estimated_duration: int = Field(..., description="Estimated duration in seconds")
    available_agents: List[Dict[str, Any]] = Field(
        default_factory=list, description="Available agents"
    )
    agent_capabilities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Agent capabilities"
    )
    planning_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Planning metadata"
    )
    execution_ready: bool = Field(False, description="Whether plan is ready for execution")
    session_id: str = Field(..., description="Session identifier")


class PlannerOrchestrator(BasePlannerComponent):
    """
    Unified orchestrator for the planner system.
    
    This class provides a simplified interface for research planning,
    workflow generation, and execution coordination.
    """
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        config = config or PlannerConfig()
        service_manager = PlannerServiceManager(config)
        super().__init__(service_manager)
        
        # Initialize core components
        self.research_planner = ResearchPlanner(config)
        self.workflow_planner = WorkflowPlanner(service_manager)
        self.agent_analyzer = AgentAnalyzer(config)
        self.workflow_builder = LangGraphWorkflowBuilder(config, service_manager)
        
        # Performance tracking
        self.operation_count = 0
        self.last_operation_time = None
    
    async def _execute(self, request: PlannerRequest) -> PlannerResponse:
        """Execute unified planner request"""
        
        self.operation_count += 1
        self.last_operation_time = datetime.now()
        
        try:
            # Step 1: Generate research plan
            research_input = ResearchPlanningInput(
                research_query=request.research_query,
                requirements=request.requirements,
                constraints=request.context.get("constraints", {}),
                preferred_agents=request.preferred_agents,
                session_id=request.session_id or f"session_{self.operation_count}"
            )
            
            research_output = await self.research_planner.execute_with_error_handling(research_input)
            
            # Step 2: Execute workflow if requested
            execution_result = None
            if request.execution_mode in ["execute_only", "plan_and_execute"]:
                execution_result = await self._execute_workflow(
                    research_output.workflow_plan,
                    research_output.session_id or request.session_id
                )
            
            # Step 3: Compile unified response
            response = PlannerResponse(
                workflow_plan=research_output.workflow_plan,
                execution_result=execution_result,
                reasoning_trace=research_output.reasoning_trace,
                confidence_score=research_output.confidence_score,
                estimated_duration=research_output.estimated_duration,
                available_agents=research_output.available_agents,
                agent_capabilities=research_output.agent_capabilities,
                planning_metadata=research_output.planning_metadata,
                execution_ready=research_output.execution_ready,
                session_id=research_output.session_id or request.session_id or f"session_{self.operation_count}"
            )
            
            return response
            
        except Exception as e:
            raise PlannerError(f"Orchestration failed: {e}")
    
    async def plan_only(self, request: PlannerRequest) -> PlannerResponse:
        """Generate workflow plan without execution"""
        request.execution_mode = "plan_only"
        return await self.execute_with_error_handling(request)
    
    async def execute_plan(self, workflow_plan: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute an existing workflow plan"""
        return await self._execute_workflow(workflow_plan, session_id)
    
    async def plan_and_execute(self, request: PlannerRequest) -> PlannerResponse:
        """Generate workflow plan and execute it"""
        request.execution_mode = "plan_and_execute"
        return await self.execute_with_error_handling(request)
    
    async def stream_execution(
        self, 
        workflow_plan: Dict[str, Any], 
        session_id: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream workflow execution"""
        async for event in self.workflow_builder.stream_workflow_execution(
            research_query=workflow_plan.get("research_query", ""),
            workflow_plan=workflow_plan,
            agent_profiles=workflow_plan.get("agent_profiles", {}),
            session_id=session_id
        ):
            yield event
    
    async def analyze_agents(self, agent_names: Optional[List[str]] = None) -> List[AgentCapability]:
        """Analyze available agents"""
        available_agents = self.services.agent_registry.get_all_agents()
        
        if agent_names:
            # Filter to specific agents
            filtered_agents = {
                name: agent_class for name, agent_class in available_agents.items()
                if name in agent_names
            }
        else:
            filtered_agents = available_agents
        
        agent_classes = list(filtered_agents.values())
        return await self.agent_analyzer.analyze_agent_batch(agent_classes)
    
    async def validate_workflow(self, workflow_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a workflow plan"""
        return await self.workflow_planner.validate_plan(workflow_plan)
    
    async def generate_alternatives(
        self, 
        request: PlannerRequest, 
        num_alternatives: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate alternative workflow plans"""
        workflow_input = WorkflowPlanningInput(
            research_query=request.research_query,
            requirements=request.requirements or {},
            available_agents=self._get_available_agents_list(),
            context=request.context or {}
        )
        
        return await self.workflow_planner.generate_alternative_plans(
            workflow_input, num_alternatives
        )
    
    async def get_suggestions(self, workflow_plan: Dict[str, Any]) -> List[str]:
        """Get suggestions for workflow improvement"""
        agent_capabilities = await self.analyze_agents()
        return await self.agent_analyzer.suggest_workflow_improvements(
            workflow_plan, agent_capabilities
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.services.get_metrics()
        metrics.update({
            "orchestrator_operations": self.operation_count,
            "last_operation_time": self.last_operation_time.isoformat() if self.last_operation_time else None
        })
        return metrics
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.services.reset_metrics()
        self.operation_count = 0
        self.last_operation_time = None
    
    async def _execute_workflow(
        self, 
        workflow_plan: Dict[str, Any], 
        session_id: str
    ) -> Dict[str, Any]:
        """Execute workflow plan"""
        return await self.workflow_builder.execute_workflow(
            research_query=workflow_plan.get("research_query", ""),
            workflow_plan=workflow_plan,
            agent_profiles=workflow_plan.get("agent_profiles", {}),
            session_id=session_id
        )
    
    def _get_available_agents_list(self) -> List[Dict[str, Any]]:
        """Get available agents as list for workflow planner"""
        agents_info = self.workflow_builder.get_available_agents()
        
        agents_list = []
        for name, info in agents_info.items():
            agents_list.append({
                "agent_name": name,
                "description": info.get("description", ""),
                "domain": "general",  # Could be enhanced with domain detection
                "capabilities": [],   # Could be enhanced with capability analysis
                "input_schema": info.get("input_schema"),
                "output_schema": info.get("output_schema")
            })
        
        return agents_list
    
    def get_agent_registry(self):
        """Get the agent registry"""
        return self.services.agent_registry
    
    def register_agent(self, name: str, agent_class):
        """Register a new agent"""
        self.services.agent_registry.register_agent(name, agent_class)
        self.workflow_builder.register_agent(name, agent_class)
    
    def clear_caches(self):
        """Clear all caches"""
        self.workflow_planner.clear_cache()
        # Add other cache clearing as needed
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health_status = {
            "orchestrator": "healthy",
            "services": {},
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check LLM service
        try:
            await self.services.llm_service.invoke_with_retry([
                {"role": "user", "content": "test"}
            ])
            health_status["services"]["llm"] = "healthy"
        except Exception as e:
            health_status["services"]["llm"] = f"unhealthy: {e}"
        
        # Check agent registry
        try:
            agents = self.services.agent_registry.get_all_agents()
            health_status["services"]["agent_registry"] = f"healthy ({len(agents)} agents)"
        except Exception as e:
            health_status["services"]["agent_registry"] = f"unhealthy: {e}"
        
        # Check components
        components = [
            ("research_planner", self.research_planner),
            ("workflow_planner", self.workflow_planner),
            ("agent_analyzer", self.agent_analyzer),
            ("workflow_builder", self.workflow_builder)
        ]
        
        for name, component in components:
            try:
                # Basic health check - component exists and is initialized
                if hasattr(component, 'services') or hasattr(component, 'config'):
                    health_status["components"][name] = "healthy"
                else:
                    health_status["components"][name] = "unhealthy: not properly initialized"
            except Exception as e:
                health_status["components"][name] = f"unhealthy: {e}"
        
        return health_status