from typing import Annotated, Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from akd.nodes.states import NodeState, merge_node_states


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    NOT_STARTED = "not_started"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class NodeExecutionStatus(str, Enum):
    """Individual node execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


def merge_messages(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """Custom reducer for messages"""
    if not existing:
        return new
    if not new:
        return existing
    return existing + new


def merge_node_results(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Custom reducer for node results"""
    if not existing:
        return new
    if not new:
        return existing
    merged = existing.copy()
    merged.update(new)
    return merged


def merge_errors(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """Custom reducer for errors"""
    if not existing:
        return new
    if not new:
        return existing
    return existing + new


class PlannerState(BaseModel):
    """LangGraph-compatible state for research planner workflow execution"""
    
    model_config = {"arbitrary_types_allowed": True}
    
    # Core workflow data
    research_query: str = ""
    requirements: Dict[str, Any] = Field(default_factory=dict)
    workflow_plan: Optional[Dict[str, Any]] = None
    
    # Execution tracking with LangGraph reducers
    messages: Annotated[List[Dict[str, Any]], merge_messages] = Field(default_factory=list)
    node_results: Annotated[Dict[str, Any], merge_node_results] = Field(default_factory=dict)
    errors: Annotated[List[Dict[str, Any]], merge_errors] = Field(default_factory=list)
    
    # Extend existing NodeState pattern for compatibility
    node_states: Annotated[Dict[str, NodeState], merge_node_states] = Field(default_factory=dict)
    
    # Status tracking
    workflow_status: WorkflowStatus = WorkflowStatus.NOT_STARTED
    node_statuses: Dict[str, NodeExecutionStatus] = Field(default_factory=dict)
    current_node: Optional[str] = None
    
    # Agent profiles and analysis
    agent_profiles: Dict[str, Any] = Field(default_factory=dict)
    compatibility_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Session and metadata
    session_id: str = ""
    thread_id: Optional[str] = None
    started_at: Optional[datetime] = None
    last_updated: datetime = Field(default_factory=datetime.now)
    
    # Progress tracking
    total_nodes: int = 0
    completed_nodes: int = 0
    progress_percentage: float = 0.0
    
    # Configuration and context
    planner_config: Dict[str, Any] = Field(default_factory=dict)
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    
    def update_progress(self):
        """Update progress percentage based on node completion"""
        if self.total_nodes > 0:
            self.progress_percentage = (self.completed_nodes / self.total_nodes) * 100
        else:
            self.progress_percentage = 0.0
        
        self.last_updated = datetime.now()
    
    def add_message(self, message_type: str, content: Any, node_id: Optional[str] = None):
        """Add message with proper formatting for LangGraph"""
        message = {
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "node_id": node_id
        }
        # This will trigger the merge_messages reducer
        self.messages = [message]
    
    def update_node_result(self, node_id: str, result: Any):
        """Update node result with proper reducer"""
        # This will trigger the merge_node_results reducer
        self.node_results = {node_id: result}
        
        # Update completion tracking
        if node_id not in self.node_statuses or self.node_statuses[node_id] != NodeExecutionStatus.COMPLETED:
            self.node_statuses[node_id] = NodeExecutionStatus.COMPLETED
            self.completed_nodes = sum(1 for status in self.node_statuses.values() 
                                     if status == NodeExecutionStatus.COMPLETED)
            self.update_progress()
    
    def add_error(self, error_type: str, message: str, node_id: Optional[str] = None):
        """Add error with proper formatting"""
        error = {
            "type": error_type,
            "message": message,
            "node_id": node_id,
            "timestamp": datetime.now().isoformat()
        }
        # This will trigger the merge_errors reducer
        self.errors = [error]
        
        # Update node status to failed
        if node_id:
            self.node_statuses[node_id] = NodeExecutionStatus.FAILED
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of current execution state"""
        return {
            "workflow_status": self.workflow_status,
            "progress": self.progress_percentage,
            "completed_nodes": self.completed_nodes,
            "total_nodes": self.total_nodes,
            "current_node": self.current_node,
            "errors_count": len(self.errors),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }
    
    def initialize_workflow(self, workflow_plan: Dict[str, Any]):
        """Initialize workflow with plan"""
        self.workflow_plan = workflow_plan
        self.workflow_status = WorkflowStatus.EXECUTING
        self.started_at = datetime.now()
        
        # Initialize node statuses
        nodes = workflow_plan.get("nodes", [])
        self.total_nodes = len(nodes)
        
        for node in nodes:
            node_id = node["node_id"]
            self.node_statuses[node_id] = NodeExecutionStatus.PENDING
            # Initialize node state using existing pattern
            self.node_states[node_id] = NodeState()
        
        self.add_message("workflow_start", "Workflow execution started")
    
    def finalize_workflow(self):
        """Finalize workflow execution"""
        self.workflow_status = WorkflowStatus.COMPLETED
        self.current_node = None
        self.update_progress()
        self.add_message("workflow_complete", "Workflow execution completed")


class CheckpointManager:
    """Manages LangGraph checkpointing for workflow state persistence"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_type = config.get("checkpoint_storage", "memory")
        self.namespace = config.get("checkpoint_namespace", "akd_planner")
        self.max_history = config.get("max_checkpoint_history", 100)
        
        # Initialize checkpointer based on storage type
        if self.storage_type == "sqlite":
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                self.checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
            except ImportError:
                from langgraph.checkpoint.memory import MemorySaver
                self.checkpointer = MemorySaver()
        elif self.storage_type == "postgres":
            # Would need connection string from config
            postgres_url = config.get("postgres_checkpoint_url")
            if postgres_url:
                try:
                    from langgraph.checkpoint.postgres import PostgresSaver
                    self.checkpointer = PostgresSaver.from_conn_string(postgres_url)
                except ImportError:
                    from langgraph.checkpoint.memory import MemorySaver
                    self.checkpointer = MemorySaver()
            else:
                from langgraph.checkpoint.memory import MemorySaver
                self.checkpointer = MemorySaver()
        else:
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
    
    def get_checkpointer(self):
        """Get the configured checkpointer"""
        return self.checkpointer
    
    async def get_checkpoint_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get checkpoint history for thread"""
        try:
            # Get history from checkpointer
            history = []
            async for checkpoint in self.checkpointer.alist({"configurable": {"thread_id": thread_id}}):
                history.append({
                    "checkpoint_id": checkpoint.id,
                    "timestamp": checkpoint.ts,
                    "metadata": checkpoint.metadata,
                    "parent_id": checkpoint.parent_config.get("configurable", {}).get("checkpoint_id") if checkpoint.parent_config else None
                })
            
            return history[:self.max_history]  # Limit history size
            
        except Exception as e:
            print(f"Error retrieving checkpoint history: {e}")
            return []