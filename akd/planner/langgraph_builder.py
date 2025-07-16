import asyncio
import importlib
import inspect
from typing import Any, Dict, List, Optional, Callable, Type
from langgraph.graph import StateGraph, START, END
from datetime import datetime

from .langgraph_state import PlannerState, WorkflowStatus, NodeExecutionStatus, CheckpointManager
from ..configs.planner_config import PlannerConfig
from ..agents._base import BaseAgent, BaseAgentConfig
from .._base import AbstractBase
from ..nodes.states import GlobalState, NodeState
from .agent_node_template import AgentNodeTemplate, AgentNodeTemplateFactory
from .state_adapters import StateAdapter, PlannerControlMixin
from .data_flow import AutomaticDataFlowManager, DataFlowMiddleware
from .core import PlannerServiceManager


class LangGraphWorkflowBuilder(PlannerControlMixin):
    """Builds LangGraph workflows from research plans with agent node execution"""
    
    def __init__(self, config: PlannerConfig, service_manager: Optional[PlannerServiceManager] = None):
        super().__init__()
        self.config = config
        self.service_manager = service_manager or PlannerServiceManager(config)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(config.model_dump())
        
        # State graph
        self.graph: Optional[StateGraph] = None
        self.compiled_graph = None
        
        # Use service manager's agent registry
        self.agent_registry: Dict[str, Type[BaseAgent]] = self.service_manager.agent_registry.get_all_agents()
        
        # NodeTemplate registry
        self.node_template_registry: Dict[str, AgentNodeTemplate] = {}
        
        # Data flow manager
        self.data_flow_manager = AutomaticDataFlowManager(
            enable_caching=config.enable_caching,
            debug=config.debug if hasattr(config, 'debug') else False
        )
        
        # State adapter
        self.state_adapter = StateAdapter()
        
        # Auto-discover agents if enabled
        if config.auto_discover_agents:
            self.discover_agents()
    
    def discover_agents(self):
        """Discover and register all available agents from configured packages"""
        
        # Use service manager's agent registry
        self.agent_registry = self.service_manager.agent_registry.get_all_agents()
        
        # Fallback to manual discovery if needed
        if not self.agent_registry:
            for package_name in self.config.agent_packages:
                try:
                    self._discover_agents_in_package(package_name)
                except Exception as e:
                    print(f"Warning: Could not discover agents in package {package_name}: {e}")
    
    def _discover_agents_in_package(self, package_name: str):
        """Discover agents in a specific package"""
        
        try:
            # Import the package
            package = importlib.import_module(package_name)
            
            # Get all modules in the package
            if hasattr(package, '__path__'):
                import pkgutil
                for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                    if not ispkg and not modname.endswith('._base'):
                        try:
                            module = importlib.import_module(modname)
                            self._register_agents_from_module(module)
                        except Exception as e:
                            print(f"Warning: Could not import module {modname}: {e}")
            else:
                # Single module
                self._register_agents_from_module(package)
                
        except Exception as e:
            print(f"Error discovering agents in package {package_name}: {e}")
    
    def _register_agents_from_module(self, module):
        """Register all agent classes from a module"""
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseAgent) and 
                obj is not BaseAgent and
                not name.startswith('_')):
                
                # Register the agent with its class name
                self.agent_registry[name] = obj
                self.service_manager.agent_registry.register_agent(name, obj)
                
                # Also register with common name patterns
                if name.endswith('Agent'):
                    short_name = name[:-5]  # Remove 'Agent' suffix
                    self.agent_registry[short_name] = obj
                    self.service_manager.agent_registry.register_agent(short_name, obj)
    
    def register_agent(self, agent_name: str, agent_class: Any):
        """Register an agent class for use in workflows"""
        self.agent_registry[agent_name] = agent_class
        self.service_manager.agent_registry.register_agent(agent_name, agent_class)
    
    def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available agents"""
        
        agents_info = {}
        
        for name, agent_class in self.agent_registry.items():
            agents_info[name] = self.service_manager.agent_registry.get_agent_info(name)
        
        return agents_info
    
    async def build_workflow(
        self,
        workflow_plan: Dict[str, Any],
        agent_profiles: Dict[str, Any]
    ) -> StateGraph:
        """Build LangGraph workflow from research plan with NodeTemplate integration"""
        
        # Initialize state graph with PlannerState
        self.graph = StateGraph(PlannerState)
        
        # Analyze workflow for data flow
        self.data_flow_manager.analyze_workflow_edges(workflow_plan, self.agent_registry)
        
        # Add workflow nodes
        await self._add_workflow_nodes(workflow_plan, agent_profiles)
        
        # Add control nodes
        self._add_control_nodes()
        
        # Add edges
        await self._add_workflow_edges(workflow_plan)
        
        # Set entry and finish points
        self.graph.set_entry_point("start_workflow")
        self.graph.set_finish_point("end_workflow")
        
        return self.graph
    
    async def compile_workflow(self) -> Any:
        """Compile workflow with checkpointing"""
        
        if not self.graph:
            raise ValueError("Workflow must be built before compilation")
        
        # Get checkpointer if enabled
        checkpointer = None
        if self.config.enable_checkpointing:
            checkpointer = self.checkpoint_manager.get_checkpointer()
        
        # Compile graph
        self.compiled_graph = self.graph.compile(
            checkpointer=checkpointer
        )
        
        return self.compiled_graph
    
    async def _add_workflow_nodes(
        self,
        workflow_plan: Dict[str, Any],
        agent_profiles: Dict[str, Any]
    ):
        """Add agent nodes to the graph"""
        
        nodes = workflow_plan.get("nodes", [])
        
        for node in nodes:
            node_id = node["node_id"]
            
            # Create node function with agent execution
            node_func = self._create_agent_node_function(node, agent_profiles)
            
            # Add to graph
            self.graph.add_node(node_id, node_func)
    
    def _add_control_nodes(self):
        """Add control flow nodes"""
        
        # Start workflow node
        self.graph.add_node("start_workflow", self._start_workflow_node)
        
        # End workflow node  
        self.graph.add_node("end_workflow", self._end_workflow_node)
    
    async def _add_workflow_edges(self, workflow_plan: Dict[str, Any]):
        """Add edges between nodes based on workflow plan"""
        
        edges = workflow_plan.get("edges", [])
        
        # Add workflow edges
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            
            if source and target:
                self.graph.add_edge(source, target)
        
        # Control flow edges
        self.graph.add_edge(START, "start_workflow")
        self.graph.add_edge("end_workflow", END)
        
        # Add edges from start_workflow to first nodes
        first_nodes = self._get_first_nodes(workflow_plan)
        for node_id in first_nodes:
            self.graph.add_edge("start_workflow", node_id)
        
        # Add edges from last nodes to end_workflow
        last_nodes = self._get_last_nodes(workflow_plan)
        for node_id in last_nodes:
            self.graph.add_edge(node_id, "end_workflow")
    
    def _get_first_nodes(self, workflow_plan: Dict[str, Any]) -> List[str]:
        """Get nodes that should run first (no incoming edges)"""
        nodes = workflow_plan.get("nodes", [])
        edges = workflow_plan.get("edges", [])
        
        all_node_ids = {node["node_id"] for node in nodes}
        target_nodes = {edge.get("target") for edge in edges if edge.get("target")}
        
        # First nodes are those with no incoming edges
        first_nodes = all_node_ids - target_nodes
        return list(first_nodes)
    
    def _get_last_nodes(self, workflow_plan: Dict[str, Any]) -> List[str]:
        """Get nodes that should run last (no outgoing edges)"""
        nodes = workflow_plan.get("nodes", [])
        edges = workflow_plan.get("edges", [])
        
        all_node_ids = {node["node_id"] for node in nodes}
        source_nodes = {edge.get("source") for edge in edges if edge.get("source")}
        
        # Last nodes are those with no outgoing edges
        last_nodes = all_node_ids - source_nodes
        return list(last_nodes)
    
    def _create_agent_node_function(
        self,
        node: Dict[str, Any],
        agent_profiles: Dict[str, Any]
    ) -> Callable:
        """Create node function for agent execution using NodeTemplate"""
        
        node_id = node["node_id"]
        agent_name = node["agent_name"]
        
        # Create or get NodeTemplate for this agent
        node_template = self._get_or_create_node_template(node, agent_profiles)
        
        async def agent_node_func(state: PlannerState) -> PlannerState:
            """Execute agent node with NodeTemplate integration"""
            
            # Update current node
            state.current_node = node_id
            state.node_statuses[node_id] = NodeExecutionStatus.RUNNING
            
            # Add execution message
            state.add_message("node_start", f"Starting execution of {agent_name}", node_id)
            
            try:
                # Apply planner control - modify inputs if needed
                await self._apply_planner_control(node_id, state)
                
                # Convert PlannerState to GlobalState
                global_state = self.state_adapter.planner_to_global(state)
                
                # Execute NodeTemplate
                updated_node_state = await node_template.arun(global_state)
                
                # Update planner state with node results
                state = self.state_adapter.update_planner_from_node_state(
                    state, node_id, updated_node_state
                )
                
                # Apply data flow transformations
                state = await self.data_flow_manager.apply_data_flow_transformations(state)
                
                state.add_message("node_complete", f"Completed {agent_name}", node_id)
                
                return state
                
            except Exception as e:
                # Handle error
                state.add_error("execution_error", str(e), node_id)
                state.node_statuses[node_id] = NodeExecutionStatus.FAILED
                
                # Add error message
                state.add_message("node_error", f"Error in {agent_name}: {str(e)}", node_id)
                
                return state
        
        return agent_node_func
    
    async def _start_workflow_node(self, state: PlannerState) -> PlannerState:
        """Initialize workflow execution"""
        
        if state.workflow_plan:
            state.initialize_workflow(state.workflow_plan)
        else:
            state.workflow_status = WorkflowStatus.EXECUTING
            state.started_at = datetime.now()
            state.add_message("workflow_start", "Workflow execution started")
        
        return state
    
    async def _end_workflow_node(self, state: PlannerState) -> PlannerState:
        """Finalize workflow execution"""
        
        state.finalize_workflow()
        return state
    
    def _get_or_create_node_template(
        self,
        node: Dict[str, Any],
        agent_profiles: Dict[str, Any]
    ) -> AgentNodeTemplate:
        """Get or create NodeTemplate for an agent"""
        
        node_id = node["node_id"]
        agent_name = node["agent_name"]
        
        # Check if already created
        if node_id in self.node_template_registry:
            return self.node_template_registry[node_id]
        
        # Create new NodeTemplate
        if agent_name in self.agent_registry:
            agent_class = self.agent_registry[agent_name]
            
            # Get agent configuration
            agent_config = node.get("config", {})
            
            # Create config instance
            if hasattr(agent_class, 'config_schema') and agent_class.config_schema:
                config_instance = agent_class.config_schema(**agent_config)
            else:
                config_instance = BaseAgentConfig(**agent_config)
            
            # Create NodeTemplate using factory
            node_template = AgentNodeTemplateFactory.create_from_agent_class(
                agent_class=agent_class,
                agent_config=config_instance,
                node_id=node_id,
                enable_mapping=True,
                mutation=True,
                debug=getattr(self.config, 'debug', False)
            )
            
            # Store in registry
            self.node_template_registry[node_id] = node_template
            
            return node_template
        
        else:
            # Create mock NodeTemplate for unregistered agents
            return self._create_mock_node_template(node_id, agent_name)
    
    def _create_mock_node_template(self, node_id: str, agent_name: str) -> AgentNodeTemplate:
        """Create a mock NodeTemplate for testing"""
        
        from ..agents._base import BaseAgent
        from .._base import InputSchema, OutputSchema
        from typing import Any
        
        class MockInputSchema(InputSchema):
            """Mock input schema for testing"""
            test: str = "default"
            query: str = "default query"
            
        class MockOutputSchema(OutputSchema):
            """Mock output schema for testing"""
            status: str
            output: str
            agent_name: str
            node_id: str
            timestamp: str
            note: str
        
        class MockAgent(BaseAgent):
            input_schema = MockInputSchema
            output_schema = MockOutputSchema
            
            @property
            def memory(self):
                return []
            
            async def get_response_async(self, *args, **kwargs):
                # Mock implementation for abstract method
                await asyncio.sleep(0.1)  # Simulate work
                return MockOutputSchema(
                    status="success",
                    output=f"Mock output from {agent_name}",
                    agent_name=agent_name,
                    node_id=node_id,
                    timestamp=datetime.now().isoformat(),
                    note="Agent not registered, using mock execution"
                )
            
            async def _arun(self, params):
                await asyncio.sleep(0.1)  # Simulate work
                return MockOutputSchema(
                    status="success",
                    output=f"Mock output from {agent_name}",
                    agent_name=agent_name,
                    node_id=node_id,
                    timestamp=datetime.now().isoformat(),
                    note="Agent not registered, using mock execution"
                )
        
        mock_agent = MockAgent(config=BaseAgentConfig())
        
        node_template = AgentNodeTemplateFactory.create_from_agent(
            agent=mock_agent,
            node_id=node_id,
            enable_mapping=False,  # Disable mapping for mock agents
            mutation=True,
            debug=getattr(self.config, 'debug', False)
        )
        
        # Store in registry
        self.node_template_registry[node_id] = node_template
        
        return node_template
    
    async def _apply_planner_control(self, node_id: str, state: PlannerState):
        """Apply planner control to modify node inputs before execution"""
        
        # Check if we need to modify inputs based on workflow state
        if state.workflow_status == WorkflowStatus.EXECUTING:
            # Add dynamic context based on progress
            if state.progress_percentage > 50:
                await self.modify_node_input(
                    node_id,
                    {"workflow_progress": state.progress_percentage},
                    state
                )
            
            # Add error context if previous nodes failed
            if any(status == NodeExecutionStatus.FAILED for status in state.node_statuses.values()):
                await self.inject_dynamic_prompt(
                    node_id,
                    {"error_recovery_mode": True},
                    state
                )
    
    async def execute_workflow(
        self,
        research_query: str,
        workflow_plan: Dict[str, Any],
        agent_profiles: Dict[str, Any],
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """Execute complete workflow with state management"""
        
        # Build and compile workflow
        await self.build_workflow(workflow_plan, agent_profiles)
        compiled_graph = await self.compile_workflow()
        
        # Initialize state
        initial_state = PlannerState(
            research_query=research_query,
            workflow_plan=workflow_plan,
            agent_profiles=agent_profiles,
            session_id=session_id,
            planner_config=self.config.model_dump()
        )
        
        # Execute workflow
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            final_state = await compiled_graph.ainvoke(initial_state, config)
            
            # Handle different return types from LangGraph
            if isinstance(final_state, dict):
                # Extract the actual state from the dict
                final_state = PlannerState(**final_state)
            
            return {
                "status": "success",
                "final_state": final_state,
                "execution_summary": final_state.get_execution_summary(),
                "results": final_state.node_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "partial_state": initial_state.model_dump() if initial_state else None
            }
    
    async def stream_workflow_execution(
        self,
        research_query: str,
        workflow_plan: Dict[str, Any],
        agent_profiles: Dict[str, Any],
        session_id: str = "default"
    ):
        """Stream workflow execution for real-time updates"""
        
        # Build and compile workflow
        await self.build_workflow(workflow_plan, agent_profiles)
        compiled_graph = await self.compile_workflow()
        
        # Initialize state
        initial_state = PlannerState(
            research_query=research_query,
            workflow_plan=workflow_plan,
            agent_profiles=agent_profiles,
            session_id=session_id,
            planner_config=self.config.model_dump()
        )
        
        # Stream execution
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            async for event in compiled_graph.astream(initial_state, config):
                yield {
                    "event": event,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            yield {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }