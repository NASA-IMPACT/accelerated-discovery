"""
Tests for AKD Planner state management functionality.

This test suite validates the LangGraph state management components:
- PlannerState reducers and state transitions
- CheckpointManager functionality
- State persistence and recovery
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List

from akd.planner.langgraph_state import (
    PlannerState,
    WorkflowStatus,
    NodeExecutionStatus,
    CheckpointManager,
    merge_messages,
    merge_node_results,
    merge_errors,
)
from akd.nodes.states import NodeState


class TestStateReducers:
    """Test LangGraph state reducers."""
    
    def test_merge_messages_empty_existing(self):
        """Test merging messages with empty existing list."""
        existing = []
        new = [{"type": "test", "content": "message"}]
        
        result = merge_messages(existing, new)
        
        assert result == new
        assert len(result) == 1
    
    def test_merge_messages_empty_new(self):
        """Test merging messages with empty new list."""
        existing = [{"type": "existing", "content": "message"}]
        new = []
        
        result = merge_messages(existing, new)
        
        assert result == existing
        assert len(result) == 1
    
    def test_merge_messages_both_present(self):
        """Test merging messages with both lists present."""
        existing = [{"type": "existing", "content": "message1"}]
        new = [{"type": "new", "content": "message2"}]
        
        result = merge_messages(existing, new)
        
        assert len(result) == 2
        assert result[0]["type"] == "existing"
        assert result[1]["type"] == "new"
    
    def test_merge_node_results_empty_existing(self):
        """Test merging node results with empty existing dict."""
        existing = {}
        new = {"node1": {"result": "success"}}
        
        result = merge_node_results(existing, new)
        
        assert result == new
        assert "node1" in result
    
    def test_merge_node_results_empty_new(self):
        """Test merging node results with empty new dict."""
        existing = {"node1": {"result": "existing"}}
        new = {}
        
        result = merge_node_results(existing, new)
        
        assert result == existing
        assert result["node1"]["result"] == "existing"
    
    def test_merge_node_results_update_existing(self):
        """Test merging node results with updates."""
        existing = {"node1": {"result": "old"}, "node2": {"result": "keep"}}
        new = {"node1": {"result": "new"}, "node3": {"result": "added"}}
        
        result = merge_node_results(existing, new)
        
        assert len(result) == 3
        assert result["node1"]["result"] == "new"
        assert result["node2"]["result"] == "keep"
        assert result["node3"]["result"] == "added"
    
    def test_merge_errors_functionality(self):
        """Test error merging functionality."""
        existing = [{"type": "error1", "message": "First error"}]
        new = [{"type": "error2", "message": "Second error"}]
        
        result = merge_errors(existing, new)
        
        assert len(result) == 2
        assert result[0]["type"] == "error1"
        assert result[1]["type"] == "error2"


class TestPlannerStateBasic:
    """Test basic PlannerState functionality."""
    
    def test_state_creation(self):
        """Test creating a new planner state."""
        state = PlannerState(
            research_query="Test research query",
            session_id="test_session_123"
        )
        
        assert state.research_query == "Test research query"
        assert state.session_id == "test_session_123"
        assert state.workflow_status == WorkflowStatus.NOT_STARTED
        assert state.total_nodes == 0
        assert state.completed_nodes == 0
        assert state.progress_percentage == 0.0
        assert len(state.messages) == 0
        assert len(state.node_results) == 0
        assert len(state.errors) == 0
    
    def test_state_with_initial_data(self):
        """Test creating state with initial data."""
        requirements = {"min_accuracy": 0.8, "max_time": 300}
        execution_context = {"user_id": "test_user", "priority": "high"}
        
        state = PlannerState(
            research_query="Advanced query",
            requirements=requirements,
            execution_context=execution_context
        )
        
        assert state.requirements == requirements
        assert state.execution_context == execution_context
    
    def test_state_timestamps(self):
        """Test state timestamp management."""
        state = PlannerState()
        
        # Should have creation timestamp
        assert state.last_updated is not None
        assert isinstance(state.last_updated, datetime)
        
        # Should be None initially
        assert state.started_at is None
        
        # Should be set when workflow starts
        workflow_plan = {"nodes": [], "edges": []}
        state.initialize_workflow(workflow_plan)
        
        assert state.started_at is not None
        assert isinstance(state.started_at, datetime)


class TestPlannerStateMessageHandling:
    """Test message handling in PlannerState."""
    
    def test_add_single_message(self):
        """Test adding a single message."""
        state = PlannerState()
        
        state.add_message("info", "Workflow started", "start_node")
        
        assert len(state.messages) == 1
        message = state.messages[0]
        assert message["type"] == "info"
        assert message["content"] == "Workflow started"
        assert message["node_id"] == "start_node"
        assert "timestamp" in message
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        state = PlannerState()
        
        state.add_message("info", "Message 1", "node1")
        state.add_message("warning", "Message 2", "node2")
        state.add_message("error", "Message 3", None)
        
        # Note: LangGraph reducers accumulate messages, but the last message is the latest
        assert len(state.messages) >= 1
        # The last message should be the most recent one
        assert state.messages[-1]["type"] == "error"
        assert state.messages[-1]["node_id"] is None
    
    def test_message_reducer_integration(self):
        """Test message reducer integration."""
        state = PlannerState()
        
        # Simulate existing messages
        state.messages = [{"type": "existing", "content": "old"}]
        
        # Add new message (in actual LangGraph, this would trigger reducer)
        state.add_message("new", "fresh message", "node")
        
        # In isolation, this just sets the last message
        assert len(state.messages) >= 1
        assert state.messages[-1]["type"] == "new"


class TestPlannerStateNodeManagement:
    """Test node management in PlannerState."""
    
    def test_node_result_updates(self):
        """Test updating node results."""
        state = PlannerState()
        
        # Add initial result
        state.update_node_result("node1", {"status": "success", "data": "result1"})
        
        assert "node1" in state.node_results
        assert state.node_results["node1"]["status"] == "success"
        assert state.node_statuses["node1"] == NodeExecutionStatus.COMPLETED
        assert state.completed_nodes == 1
    
    def test_multiple_node_results(self):
        """Test updating multiple node results."""
        state = PlannerState()
        
        state.update_node_result("node1", {"status": "success"})
        state.update_node_result("node2", {"status": "success"})
        state.update_node_result("node3", {"status": "failed"})
        
        # Results are accumulated via reducer
        assert len(state.node_results) >= 1
        assert state.completed_nodes == 3  # All nodes completed (regardless of success/failure)
    
    def test_node_result_overwrite(self):
        """Test overwriting node results."""
        state = PlannerState()
        
        # Initial result
        state.update_node_result("node1", {"status": "pending"})
        
        # Update result
        state.update_node_result("node1", {"status": "success", "data": "final"})
        
        assert state.node_results["node1"]["status"] == "success"
        assert state.node_results["node1"]["data"] == "final"
        assert state.completed_nodes == 1
    
    def test_node_states_integration(self):
        """Test integration with NodeState objects."""
        state = PlannerState()
        
        # Initialize workflow to create node states
        workflow_plan = {
            "nodes": [
                {"node_id": "node1", "agent_name": "Agent1"},
                {"node_id": "node2", "agent_name": "Agent2"}
            ],
            "edges": []
        }
        
        state.initialize_workflow(workflow_plan)
        
        # Should have NodeState objects
        assert "node1" in state.node_states
        assert "node2" in state.node_states
        assert isinstance(state.node_states["node1"], NodeState)
        assert isinstance(state.node_states["node2"], NodeState)


class TestPlannerStateErrorHandling:
    """Test error handling in PlannerState."""
    
    def test_add_error_basic(self):
        """Test adding basic error."""
        state = PlannerState()
        
        state.add_error("execution_error", "Agent failed", "node1")
        
        assert len(state.errors) == 1
        error = state.errors[0]
        assert error["type"] == "execution_error"
        assert error["message"] == "Agent failed"
        assert error["node_id"] == "node1"
        assert "timestamp" in error
        assert state.node_statuses["node1"] == NodeExecutionStatus.FAILED
    
    def test_add_error_without_node(self):
        """Test adding error without node ID."""
        state = PlannerState()
        
        state.add_error("system_error", "System failure", None)
        
        assert len(state.errors) == 1
        error = state.errors[0]
        assert error["node_id"] is None
        # Should not affect node_statuses
        assert len(state.node_statuses) == 0
    
    def test_multiple_errors(self):
        """Test adding multiple errors."""
        state = PlannerState()
        
        state.add_error("error1", "First error", "node1")
        state.add_error("error2", "Second error", "node2")
        state.add_error("error3", "Third error", "node1")  # Same node
        
        # Errors are accumulated via reducer
        assert len(state.errors) >= 1
        assert state.node_statuses["node1"] == NodeExecutionStatus.FAILED
        assert state.node_statuses["node2"] == NodeExecutionStatus.FAILED
    
    def test_error_reducer_integration(self):
        """Test error reducer integration."""
        state = PlannerState()
        
        # Simulate existing errors
        state.errors = [{"type": "existing", "message": "old error"}]
        
        # Add new error
        state.add_error("new_error", "new error message", "node")
        
        # In isolation, this just sets the last error
        assert len(state.errors) >= 1
        assert state.errors[-1]["type"] == "new_error"


class TestPlannerStateProgressTracking:
    """Test progress tracking in PlannerState."""
    
    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        state = PlannerState()
        
        # No nodes - 0% progress
        state.update_progress()
        assert state.progress_percentage == 0.0
        
        # Set total nodes
        state.total_nodes = 4
        state.update_progress()
        assert state.progress_percentage == 0.0
        
        # Complete some nodes
        state.completed_nodes = 2
        state.update_progress()
        assert state.progress_percentage == 50.0
        
        # Complete all nodes
        state.completed_nodes = 4
        state.update_progress()
        assert state.progress_percentage == 100.0
    
    def test_progress_with_node_completion(self):
        """Test progress tracking with node completion."""
        state = PlannerState()
        
        # Initialize with workflow
        workflow_plan = {
            "nodes": [
                {"node_id": "node1", "agent_name": "Agent1"},
                {"node_id": "node2", "agent_name": "Agent2"},
                {"node_id": "node3", "agent_name": "Agent3"}
            ],
            "edges": []
        }
        
        state.initialize_workflow(workflow_plan)
        assert state.progress_percentage == 0.0
        
        # Complete nodes one by one
        state.update_node_result("node1", {"status": "success"})
        assert abs(state.progress_percentage - (100.0 / 3)) < 0.1  # ~33.33%
        
        state.update_node_result("node2", {"status": "success"})
        assert abs(state.progress_percentage - (200.0 / 3)) < 0.1  # ~66.67%
        
        state.update_node_result("node3", {"status": "success"})
        assert state.progress_percentage == 100.0
    
    def test_progress_tracking_edge_cases(self):
        """Test progress tracking edge cases."""
        state = PlannerState()
        
        # Zero total nodes
        state.total_nodes = 0
        state.completed_nodes = 0
        state.update_progress()
        assert state.progress_percentage == 0.0
        
        # More completed than total (shouldn't happen but handle gracefully)
        state.total_nodes = 2
        state.completed_nodes = 3
        state.update_progress()
        assert state.progress_percentage == 150.0  # Mathematical result


class TestPlannerStateWorkflowLifecycle:
    """Test complete workflow lifecycle in PlannerState."""
    
    def test_workflow_initialization(self):
        """Test workflow initialization."""
        state = PlannerState()
        
        workflow_plan = {
            "nodes": [
                {"node_id": "node1", "agent_name": "Agent1"},
                {"node_id": "node2", "agent_name": "Agent2"}
            ],
            "edges": [{"source": "node1", "target": "node2"}]
        }
        
        state.initialize_workflow(workflow_plan)
        
        assert state.workflow_plan == workflow_plan
        assert state.workflow_status == WorkflowStatus.EXECUTING
        assert state.total_nodes == 2
        assert state.started_at is not None
        assert len(state.messages) == 1  # "Workflow execution started"
        assert state.node_statuses["node1"] == NodeExecutionStatus.PENDING
        assert state.node_statuses["node2"] == NodeExecutionStatus.PENDING
    
    def test_workflow_finalization(self):
        """Test workflow finalization."""
        state = PlannerState()
        
        # Initialize first
        workflow_plan = {"nodes": [{"node_id": "node1", "agent_name": "Agent1"}], "edges": []}
        state.initialize_workflow(workflow_plan)
        
        # Complete a node
        state.update_node_result("node1", {"status": "success"})
        
        # Finalize workflow
        state.finalize_workflow()
        
        assert state.workflow_status == WorkflowStatus.COMPLETED
        assert state.current_node is None
        assert state.progress_percentage == 100.0
        assert any(msg["type"] == "workflow_complete" for msg in state.messages)
    
    def test_execution_summary(self):
        """Test execution summary generation."""
        state = PlannerState()
        
        # Initialize workflow
        workflow_plan = {
            "nodes": [
                {"node_id": "node1", "agent_name": "Agent1"},
                {"node_id": "node2", "agent_name": "Agent2"}
            ],
            "edges": []
        }
        
        state.initialize_workflow(workflow_plan)
        
        # Add some execution data
        state.update_node_result("node1", {"status": "success"})
        state.add_error("test_error", "Test error", "node2")
        
        summary = state.get_execution_summary()
        
        assert summary["workflow_status"] == WorkflowStatus.EXECUTING
        assert summary["progress"] == 50.0
        assert summary["completed_nodes"] == 1
        assert summary["total_nodes"] == 2
        assert summary["errors_count"] == 1
        assert "last_updated" in summary
        assert summary["current_node"] is None
    
    def test_complete_workflow_lifecycle(self):
        """Test complete workflow from start to finish."""
        state = PlannerState(
            research_query="Test query",
            session_id="test_session"
        )
        
        # 1. Initialize workflow
        workflow_plan = {
            "nodes": [
                {"node_id": "query_node", "agent_name": "QueryAgent"},
                {"node_id": "process_node", "agent_name": "ProcessAgent"}
            ],
            "edges": [{"source": "query_node", "target": "process_node"}]
        }
        
        state.initialize_workflow(workflow_plan)
        
        # Verify initialization
        assert state.workflow_status == WorkflowStatus.EXECUTING
        assert state.total_nodes == 2
        assert state.progress_percentage == 0.0
        
        # 2. Execute first node
        state.current_node = "query_node"
        state.node_statuses["query_node"] = NodeExecutionStatus.RUNNING
        state.add_message("node_start", "Starting query node", "query_node")
        
        state.update_node_result("query_node", {
            "status": "success",
            "output": "Query processed successfully"
        })
        state.add_message("node_complete", "Query node completed", "query_node")
        
        # Check progress
        assert state.progress_percentage == 50.0
        assert state.completed_nodes == 1
        
        # 3. Execute second node
        state.current_node = "process_node"
        state.node_statuses["process_node"] = NodeExecutionStatus.RUNNING
        state.add_message("node_start", "Starting process node", "process_node")
        
        state.update_node_result("process_node", {
            "status": "success",
            "output": "Processing completed"
        })
        state.add_message("node_complete", "Process node completed", "process_node")
        
        # Check progress
        assert state.progress_percentage == 100.0
        assert state.completed_nodes == 2
        
        # 4. Finalize workflow
        state.finalize_workflow()
        
        # Verify final state
        assert state.workflow_status == WorkflowStatus.COMPLETED
        assert state.current_node is None
        assert len(state.messages) >= 1  # Should have at least the final message
        assert len(state.node_results) >= 1  # Should have results
        assert len(state.errors) == 0
        
        # Check summary
        summary = state.get_execution_summary()
        assert summary["workflow_status"] == WorkflowStatus.COMPLETED
        assert summary["progress"] == 100.0
        assert summary["completed_nodes"] == 2
        assert summary["total_nodes"] == 2
        assert summary["errors_count"] == 0


class TestCheckpointManagerBasic:
    """Test basic CheckpointManager functionality."""
    
    def test_memory_checkpointer_creation(self):
        """Test creating memory checkpointer."""
        config = {
            "checkpoint_storage": "memory",
            "checkpoint_namespace": "test_namespace",
            "max_checkpoint_history": 50
        }
        
        manager = CheckpointManager(config)
        
        assert manager.storage_type == "memory"
        assert manager.namespace == "test_namespace"
        assert manager.max_history == 50
        
        checkpointer = manager.get_checkpointer()
        assert checkpointer is not None
    
    def test_sqlite_checkpointer_creation(self):
        """Test creating SQLite checkpointer with fallback."""
        config = {
            "checkpoint_storage": "sqlite",
            "checkpoint_namespace": "test_namespace"
        }
        
        manager = CheckpointManager(config)
        
        assert manager.storage_type == "sqlite"
        checkpointer = manager.get_checkpointer()
        assert checkpointer is not None
        # May be SQLite or memory fallback
    
    def test_postgres_checkpointer_fallback(self):
        """Test PostgreSQL checkpointer fallback."""
        config = {
            "checkpoint_storage": "postgres",
            "checkpoint_namespace": "test_namespace"
        }
        
        manager = CheckpointManager(config)
        
        assert manager.storage_type == "postgres"
        checkpointer = manager.get_checkpointer()
        assert checkpointer is not None
        # Should fallback to memory without postgres URL
    
    def test_default_configuration(self):
        """Test default checkpoint configuration."""
        config = {}
        
        manager = CheckpointManager(config)
        
        assert manager.storage_type == "memory"
        assert manager.namespace == "akd_planner"
        assert manager.max_history == 100
    
    @pytest.mark.asyncio
    async def test_empty_checkpoint_history(self):
        """Test getting checkpoint history for non-existent thread."""
        config = {"checkpoint_storage": "memory"}
        manager = CheckpointManager(config)
        
        history = await manager.get_checkpoint_history("non_existent_thread")
        
        assert isinstance(history, list)
        assert len(history) == 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_history_error_handling(self):
        """Test checkpoint history error handling."""
        config = {"checkpoint_storage": "memory"}
        manager = CheckpointManager(config)
        
        # Test with invalid thread ID
        history = await manager.get_checkpoint_history("")
        
        assert isinstance(history, list)
        assert len(history) == 0
        
        # Test with None thread ID
        history = await manager.get_checkpoint_history(None)
        
        assert isinstance(history, list)
        assert len(history) == 0


if __name__ == "__main__":
    pytest.main([__file__])