"""
State adapters for converting between PlannerState and GlobalState.

This module provides utilities to convert between the LangGraph PlannerState
and the NodeTemplate GlobalState systems, enabling seamless integration
between the planner and the node template architecture.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from akd.nodes.states import NodeState, GlobalState
from akd.planner.langgraph_state import PlannerState, WorkflowStatus, NodeExecutionStatus


class StateAdapter:
    """
    Adapter for converting between PlannerState and GlobalState.
    
    This class provides bidirectional conversion between the two state systems,
    allowing the planner to work with NodeTemplate-based nodes while maintaining
    its workflow-level state management.
    """
    
    @staticmethod
    def planner_to_global(planner_state: PlannerState) -> GlobalState:
        """
        Convert PlannerState to GlobalState.
        
        Args:
            planner_state: The PlannerState to convert
            
        Returns:
            GlobalState instance with data from PlannerState
        """
        # Create global state with node states from planner
        global_state = GlobalState(
            node_states=planner_state.node_states.copy(),
            # Copy core NodeState fields from planner state
            messages=StateAdapter._convert_planner_messages(planner_state.messages),
            inputs=StateAdapter._extract_global_inputs(planner_state),
            outputs=StateAdapter._extract_global_outputs(planner_state),
            input_guardrails={},
            output_guardrails={},
            steps=StateAdapter._extract_global_steps(planner_state)
        )
        
        return global_state
    
    @staticmethod
    def global_to_planner(
        global_state: GlobalState, 
        planner_state: PlannerState
    ) -> PlannerState:
        """
        Update PlannerState with data from GlobalState.
        
        Args:
            global_state: The GlobalState to extract data from
            planner_state: The PlannerState to update
            
        Returns:
            Updated PlannerState
        """
        # Update node states (this will trigger LangGraph reducers)
        planner_state.node_states = global_state.node_states.copy()
        
        # Update messages if global state has new messages
        if global_state.messages:
            planner_state.messages = StateAdapter._convert_global_messages(global_state.messages)
        
        # Update last updated timestamp
        planner_state.last_updated = datetime.now()
        
        return planner_state
    
    @staticmethod
    def create_node_state_from_planner(
        planner_state: PlannerState, 
        node_id: str
    ) -> NodeState:
        """
        Create a NodeState for a specific node from PlannerState.
        
        Args:
            planner_state: The PlannerState to extract from
            node_id: The ID of the node to create state for
            
        Returns:
            NodeState instance for the specified node
        """
        # Get existing node state or create new one
        if node_id in planner_state.node_states:
            base_state = planner_state.node_states[node_id]
        else:
            base_state = NodeState()
        
        # Add planner context to inputs
        enhanced_inputs = base_state.inputs.copy()
        enhanced_inputs.update({
            "research_query": planner_state.research_query,
            "session_id": planner_state.session_id,
            "workflow_status": planner_state.workflow_status,
            "planner_context": StateAdapter._create_planner_context(planner_state)
        })
        
        # Create enhanced node state
        node_state = NodeState(
            messages=base_state.messages.copy(),
            inputs=enhanced_inputs,
            outputs=base_state.outputs.copy(),
            input_guardrails=base_state.input_guardrails.copy(),
            output_guardrails=base_state.output_guardrails.copy(),
            steps=base_state.steps.copy()
        )
        
        return node_state
    
    @staticmethod
    def update_planner_from_node_state(
        planner_state: PlannerState,
        node_id: str,
        node_state: NodeState
    ) -> PlannerState:
        """
        Update PlannerState with results from a NodeState.
        
        Args:
            planner_state: The PlannerState to update
            node_id: The ID of the node
            node_state: The NodeState with results
            
        Returns:
            Updated PlannerState
        """
        # Update node state (triggers LangGraph reducer)
        planner_state.node_states = {node_id: node_state}
        
        # Update node results if there are outputs
        if node_state.outputs:
            planner_state.node_results = {node_id: node_state.outputs}
        
        # Update status based on node state
        if "error" in node_state.outputs:
            planner_state.node_statuses[node_id] = NodeExecutionStatus.FAILED
            planner_state.add_error(
                "node_execution_error",
                str(node_state.outputs["error"]),
                node_id
            )
        elif node_state.outputs:
            planner_state.node_statuses[node_id] = NodeExecutionStatus.COMPLETED
            planner_state.completed_nodes = sum(
                1 for status in planner_state.node_statuses.values()
                if status == NodeExecutionStatus.COMPLETED
            )
            planner_state.update_progress()
        
        return planner_state
    
    @staticmethod
    def _convert_planner_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert planner messages to global state format."""
        converted = []
        for msg in messages:
            converted.append({
                "type": msg.get("type", "info"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                "node_id": msg.get("node_id"),
                "source": "planner"
            })
        return converted
    
    @staticmethod
    def _convert_global_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert global state messages to planner format."""
        converted = []
        for msg in messages:
            converted.append({
                "type": msg.get("type", "info"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                "node_id": msg.get("node_id"),
                "source": "node"
            })
        return converted
    
    @staticmethod
    def _extract_global_inputs(planner_state: PlannerState) -> Dict[str, Any]:
        """Extract global inputs from planner state."""
        return {
            "research_query": planner_state.research_query,
            "session_id": planner_state.session_id,
            "workflow_plan": planner_state.workflow_plan,
            "requirements": planner_state.requirements,
            "execution_context": planner_state.execution_context
        }
    
    @staticmethod
    def _extract_global_outputs(planner_state: PlannerState) -> Dict[str, Any]:
        """Extract global outputs from planner state."""
        return {
            "workflow_status": planner_state.workflow_status,
            "progress_percentage": planner_state.progress_percentage,
            "completed_nodes": planner_state.completed_nodes,
            "total_nodes": planner_state.total_nodes,
            "node_results": planner_state.node_results,
            "execution_summary": planner_state.get_execution_summary()
        }
    
    @staticmethod
    def _extract_global_steps(planner_state: PlannerState) -> Dict[str, Any]:
        """Extract global steps from planner state."""
        return {
            "workflow_initialization": planner_state.started_at.isoformat() if planner_state.started_at else None,
            "current_node": planner_state.current_node,
            "node_statuses": planner_state.node_statuses,
            "compatibility_scores": planner_state.compatibility_scores
        }
    
    @staticmethod
    def _create_planner_context(planner_state: PlannerState) -> Dict[str, Any]:
        """Create planner context information for nodes."""
        return {
            "workflow_status": planner_state.workflow_status,
            "progress_percentage": planner_state.progress_percentage,
            "completed_nodes": planner_state.completed_nodes,
            "total_nodes": planner_state.total_nodes,
            "current_node": planner_state.current_node,
            "session_id": planner_state.session_id,
            "started_at": planner_state.started_at.isoformat() if planner_state.started_at else None,
            "agent_profiles": planner_state.agent_profiles,
            "planner_config": planner_state.planner_config
        }


class PlannerStateObserver:
    """
    Observer for monitoring state changes and providing planner control.
    
    This class implements the observer pattern to monitor state changes
    and provide the planner with control over the workflow execution.
    """
    
    def __init__(self, planner_instance):
        """
        Initialize the state observer.
        
        Args:
            planner_instance: Reference to the planner instance
        """
        self.planner = planner_instance
        self.state_change_callbacks = []
    
    def register_callback(self, callback):
        """Register a callback for state changes."""
        self.state_change_callbacks.append(callback)
    
    async def on_node_state_change(
        self, 
        node_id: str, 
        old_state: NodeState, 
        new_state: NodeState
    ):
        """
        Handle node state changes.
        
        Args:
            node_id: The ID of the node that changed
            old_state: Previous node state
            new_state: New node state
        """
        # Notify planner of state change
        if hasattr(self.planner, 'handle_state_change'):
            await self.planner.handle_state_change(node_id, old_state, new_state)
        
        # Call registered callbacks
        for callback in self.state_change_callbacks:
            try:
                await callback(node_id, old_state, new_state)
            except Exception as e:
                # Log error but don't break other callbacks
                print(f"State change callback error: {e}")
    
    async def on_workflow_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Handle workflow-level events.
        
        Args:
            event_type: Type of the event
            event_data: Event data
        """
        if hasattr(self.planner, 'handle_workflow_event'):
            await self.planner.handle_workflow_event(event_type, event_data)


class PlannerControlMixin:
    """
    Mixin to add planner control capabilities to the workflow builder.
    
    This mixin provides methods for the planner to control workflow execution,
    modify node inputs on the fly, and react to state changes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_observer = PlannerStateObserver(self)
        self.control_enabled = True
    
    async def handle_state_change(
        self, 
        node_id: str, 
        old_state: NodeState, 
        new_state: NodeState
    ):
        """
        Handle state changes from nodes.
        
        Args:
            node_id: The ID of the node that changed
            old_state: Previous node state
            new_state: New node state
        """
        if not self.control_enabled:
            return
        
        # Check for low confidence results
        if new_state.outputs.get("confidence_score", 1.0) < 0.5:
            await self._handle_low_confidence_result(node_id, new_state)
        
        # Check for errors
        if "error" in new_state.outputs:
            await self._handle_node_error(node_id, new_state)
        
        # Check for requests for human intervention
        if new_state.outputs.get("requires_human_intervention"):
            await self._handle_human_intervention_request(node_id, new_state)
    
    async def handle_workflow_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Handle workflow-level events.
        
        Args:
            event_type: Type of the event
            event_data: Event data
        """
        if not self.control_enabled:
            return
        
        if event_type == "node_execution_start":
            await self._handle_node_execution_start(event_data)
        elif event_type == "workflow_error":
            await self._handle_workflow_error(event_data)
        elif event_type == "replanning_requested":
            await self._handle_replanning_request(event_data)
    
    async def modify_node_input(
        self, 
        node_id: str, 
        input_modifications: Dict[str, Any],
        planner_state: PlannerState
    ):
        """
        Modify node input before execution.
        
        Args:
            node_id: The ID of the node to modify
            input_modifications: Modifications to apply
            planner_state: Current planner state
        """
        if node_id in planner_state.node_states:
            node_state = planner_state.node_states[node_id]
            node_state.inputs.update(input_modifications)
            
            # Update the planner state with modified node state
            planner_state.node_states = {node_id: node_state}
    
    async def inject_dynamic_prompt(
        self, 
        node_id: str, 
        prompt_context: Dict[str, Any],
        planner_state: PlannerState
    ):
        """
        Inject dynamic prompt based on workflow state.
        
        Args:
            node_id: The ID of the node to modify
            prompt_context: Context for prompt generation
            planner_state: Current planner state
        """
        dynamic_prompt = self._generate_dynamic_prompt(prompt_context)
        
        await self.modify_node_input(
            node_id, 
            {"dynamic_prompt": dynamic_prompt},
            planner_state
        )
    
    def _generate_dynamic_prompt(self, context: Dict[str, Any]) -> str:
        """Generate dynamic prompt based on context."""
        base_prompt = "You are an AI assistant helping with research tasks."
        
        # Add context based on workflow state
        if context.get("low_confidence_detected"):
            base_prompt += " Please be extra thorough and provide detailed explanations."
        
        if context.get("error_recovery_mode"):
            base_prompt += " The previous step encountered an error. Please proceed carefully."
        
        if context.get("final_node"):
            base_prompt += " This is the final step, please provide a comprehensive summary."
        
        return base_prompt
    
    async def _handle_low_confidence_result(self, node_id: str, node_state: NodeState):
        """Handle low confidence results."""
        # Could trigger re-execution with different parameters
        # or modify downstream nodes to be more thorough
        pass
    
    async def _handle_node_error(self, node_id: str, node_state: NodeState):
        """Handle node errors."""
        # Could trigger error recovery protocols
        # or modify downstream nodes for error handling
        pass
    
    async def _handle_human_intervention_request(self, node_id: str, node_state: NodeState):
        """Handle requests for human intervention."""
        # Could pause workflow or trigger human notification
        pass
    
    async def _handle_node_execution_start(self, event_data: Dict[str, Any]):
        """Handle node execution start."""
        # Could modify node inputs based on current state
        pass
    
    async def _handle_workflow_error(self, event_data: Dict[str, Any]):
        """Handle workflow errors."""
        # Could trigger error recovery or workflow modification
        pass
    
    async def _handle_replanning_request(self, event_data: Dict[str, Any]):
        """Handle replanning requests."""
        # Could modify the workflow graph dynamically
        pass