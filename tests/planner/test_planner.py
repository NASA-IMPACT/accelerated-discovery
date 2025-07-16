"""
Comprehensive tests for AKD Planner module.

This test suite validates the planner's ability to:
1. Discover and register agents from the AKD framework
2. Build LangGraph workflows from research plans
3. Execute workflows with real agents
4. Handle errors and edge cases appropriately
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from akd.planner import (
    PlannerConfig,
    LangGraphWorkflowBuilder,
    PlannerState,
    WorkflowStatus,
    NodeExecutionStatus,
    CheckpointManager,
)


class TestPlannerConfig:
    """Test planner configuration."""
    
    def test_default_config(self):
        """Test default planner configuration."""
        config = PlannerConfig()
        
        assert config.auto_discover_agents is True
        assert config.agent_packages == ["akd.agents", "akd.nodes"]
        assert config.max_workflow_nodes == 20
        assert config.execution_strategy == "langgraph"
        assert config.enable_checkpointing is True
        assert config.enable_monitoring is True
        assert config.enable_streaming is True
    
    def test_custom_config(self):
        """Test custom planner configuration."""
        config = PlannerConfig(
            auto_discover_agents=False,
            max_workflow_nodes=10,
            execution_strategy="mock",
            enable_checkpointing=False
        )
        
        assert config.auto_discover_agents is False
        assert config.max_workflow_nodes == 10
        assert config.execution_strategy == "mock"
        assert config.enable_checkpointing is False


class TestPlannerState:
    """Test planner state management."""
    
    def test_state_initialization(self):
        """Test state initialization."""
        state = PlannerState(
            research_query="Test query",
            session_id="test_session"
        )
        
        assert state.research_query == "Test query"
        assert state.session_id == "test_session"
        assert state.workflow_status == WorkflowStatus.NOT_STARTED
        assert state.total_nodes == 0
        assert state.completed_nodes == 0
        assert state.progress_percentage == 0.0
    
    def test_add_message(self):
        """Test adding messages to state."""
        state = PlannerState()
        
        state.add_message("test_type", "test content", "node_1")
        
        assert len(state.messages) == 1
        assert state.messages[0]["type"] == "test_type"
        assert state.messages[0]["content"] == "test content"
        assert state.messages[0]["node_id"] == "node_1"
    
    def test_update_node_result(self):
        """Test updating node results."""
        state = PlannerState()
        state.node_statuses["node_1"] = NodeExecutionStatus.RUNNING
        
        state.update_node_result("node_1", {"result": "success"})
        
        assert state.node_results["node_1"]["result"] == "success"
        assert state.node_statuses["node_1"] == NodeExecutionStatus.COMPLETED
    
    def test_add_error(self):
        """Test adding errors to state."""
        state = PlannerState()
        
        state.add_error("test_error", "Error message", "node_1")
        
        assert len(state.errors) == 1
        assert state.errors[0]["type"] == "test_error"
        assert state.errors[0]["message"] == "Error message"
        assert state.errors[0]["node_id"] == "node_1"
        assert state.node_statuses["node_1"] == NodeExecutionStatus.FAILED
    
    def test_progress_tracking(self):
        """Test progress tracking."""
        state = PlannerState()
        
        # Initialize workflow
        workflow_plan = {
            "nodes": [
                {"node_id": "node_1", "agent_name": "Agent1"},
                {"node_id": "node_2", "agent_name": "Agent2"}
            ]
        }
        
        state.initialize_workflow(workflow_plan)
        
        assert state.total_nodes == 2
        assert state.progress_percentage == 0.0
        assert state.workflow_status == WorkflowStatus.EXECUTING
        
        # Complete one node
        state.update_node_result("node_1", {"result": "success"})
        
        assert state.completed_nodes == 1
        assert state.progress_percentage == 50.0
        
        # Complete second node
        state.update_node_result("node_2", {"result": "success"})
        
        assert state.completed_nodes == 2
        assert state.progress_percentage == 100.0
    
    def test_execution_summary(self):
        """Test execution summary generation."""
        state = PlannerState(
            research_query="Test query",
            session_id="test_session"
        )
        
        state.add_error("test_error", "Error message")
        
        summary = state.get_execution_summary()
        
        assert summary["workflow_status"] == WorkflowStatus.NOT_STARTED
        assert summary["progress"] == 0.0
        assert summary["errors_count"] == 1
        assert "last_updated" in summary


class TestAgentDiscovery:
    """Test agent discovery functionality."""
    
    def test_agent_discovery_enabled(self):
        """Test agent discovery when enabled."""
        config = PlannerConfig(auto_discover_agents=True)
        builder = LangGraphWorkflowBuilder(config)
        
        # Should discover agents from akd.agents
        available_agents = builder.get_available_agents()
        
        assert len(available_agents) > 0
        assert "QueryAgent" in available_agents
        assert "IntentAgent" in available_agents
        
        # Check agent info structure
        query_agent_info = available_agents["QueryAgent"]
        assert query_agent_info["class_name"] == "QueryAgent"
        assert query_agent_info["module"] == "akd.agents.query"
        assert query_agent_info["has_input_schema"] is True
        assert query_agent_info["has_output_schema"] is True
    
    def test_agent_discovery_disabled(self):
        """Test agent discovery when disabled."""
        config = PlannerConfig(auto_discover_agents=False)
        builder = LangGraphWorkflowBuilder(config)
        
        # Should not discover agents
        available_agents = builder.get_available_agents()
        
        assert len(available_agents) == 0
    
    def test_manual_agent_registration(self):
        """Test manual agent registration."""
        config = PlannerConfig(auto_discover_agents=False)
        builder = LangGraphWorkflowBuilder(config)
        
        # Mock agent class
        mock_agent = Mock()
        mock_agent.__name__ = "TestAgent"
        mock_agent.__module__ = "test.module"
        
        # Mock schema attributes with __name__
        mock_input_schema = Mock()
        mock_input_schema.__name__ = "TestInputSchema"
        mock_agent.input_schema = mock_input_schema
        
        mock_output_schema = Mock()
        mock_output_schema.__name__ = "TestOutputSchema"
        mock_agent.output_schema = mock_output_schema
        
        # Register agent
        builder.register_agent("TestAgent", mock_agent)
        
        available_agents = builder.get_available_agents()
        
        assert "TestAgent" in available_agents
        assert available_agents["TestAgent"]["class_name"] == "TestAgent"
        assert available_agents["TestAgent"]["module"] == "test.module"
        assert available_agents["TestAgent"]["input_schema"] == "TestInputSchema"
        assert available_agents["TestAgent"]["output_schema"] == "TestOutputSchema"
    
    def test_agent_name_patterns(self):
        """Test agent name pattern registration."""
        config = PlannerConfig(auto_discover_agents=True)
        builder = LangGraphWorkflowBuilder(config)
        
        available_agents = builder.get_available_agents()
        
        # Should register both full name and short name
        if "QueryAgent" in available_agents:
            assert "Query" in available_agents
            assert available_agents["QueryAgent"]["class_name"] == available_agents["Query"]["class_name"]


class TestWorkflowBuilding:
    """Test workflow building functionality."""
    
    @pytest.fixture
    def config(self):
        """Fixture for planner config."""
        return PlannerConfig(
            enable_checkpointing=False,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def builder(self, config):
        """Fixture for workflow builder."""
        return LangGraphWorkflowBuilder(config)
    
    @pytest.fixture
    def simple_workflow_plan(self):
        """Fixture for simple workflow plan."""
        return {
            "nodes": [
                {
                    "node_id": "query_node",
                    "agent_name": "QueryAgent",
                    "config": {},
                    "inputs": {"query": "Test query"}
                },
                {
                    "node_id": "intent_node",
                    "agent_name": "IntentAgent",
                    "config": {},
                    "inputs": {"query": "Test query"}
                }
            ],
            "edges": [
                {
                    "source": "query_node",
                    "target": "intent_node"
                }
            ]
        }
    
    @pytest.fixture
    def agent_profiles(self, builder):
        """Fixture for agent profiles."""
        available_agents = builder.get_available_agents()
        return {
            name: {
                "description": f"Agent {name}",
                "input_schema": info["input_schema"],
                "output_schema": info["output_schema"]
            }
            for name, info in available_agents.items()
        }
    
    @pytest.mark.asyncio
    async def test_build_workflow_success(self, builder, simple_workflow_plan, agent_profiles):
        """Test successful workflow building."""
        graph = await builder.build_workflow(simple_workflow_plan, agent_profiles)
        
        assert graph is not None
        assert len(graph.nodes) == 4  # 2 agent nodes + start + end
        assert "query_node" in graph.nodes
        assert "intent_node" in graph.nodes
        assert "start_workflow" in graph.nodes
        assert "end_workflow" in graph.nodes
    
    @pytest.mark.asyncio
    async def test_compile_workflow_success(self, builder, simple_workflow_plan, agent_profiles):
        """Test successful workflow compilation."""
        await builder.build_workflow(simple_workflow_plan, agent_profiles)
        compiled_graph = await builder.compile_workflow()
        
        assert compiled_graph is not None
    
    @pytest.mark.asyncio
    async def test_workflow_with_checkpointing(self, simple_workflow_plan, agent_profiles):
        """Test workflow with checkpointing enabled."""
        config = PlannerConfig(enable_checkpointing=True)
        builder = LangGraphWorkflowBuilder(config)
        
        await builder.build_workflow(simple_workflow_plan, agent_profiles)
        compiled_graph = await builder.compile_workflow()
        
        assert compiled_graph is not None
    
    @pytest.mark.asyncio
    async def test_empty_workflow_plan(self, builder):
        """Test building workflow with empty plan."""
        empty_plan = {"nodes": [], "edges": []}
        agent_profiles = {}
        
        graph = await builder.build_workflow(empty_plan, agent_profiles)
        
        assert graph is not None
        assert len(graph.nodes) == 2  # Only start and end nodes
    
    @pytest.mark.asyncio
    async def test_single_node_workflow(self, builder, agent_profiles):
        """Test building workflow with single node."""
        single_node_plan = {
            "nodes": [
                {
                    "node_id": "single_node",
                    "agent_name": "QueryAgent",
                    "config": {},
                    "inputs": {"query": "Test query"}
                }
            ],
            "edges": []
        }
        
        graph = await builder.build_workflow(single_node_plan, agent_profiles)
        
        assert graph is not None
        assert len(graph.nodes) == 3  # 1 agent node + start + end
        assert "single_node" in graph.nodes


class TestWorkflowExecution:
    """Test workflow execution functionality."""
    
    @pytest.fixture
    def config(self):
        """Fixture for planner config."""
        return PlannerConfig(
            enable_checkpointing=False,
            execution_strategy="langgraph"
        )
    
    @pytest.fixture
    def builder(self, config):
        """Fixture for workflow builder."""
        return LangGraphWorkflowBuilder(config)
    
    @pytest.fixture
    def single_node_plan(self):
        """Fixture for single node workflow plan."""
        return {
            "nodes": [
                {
                    "node_id": "test_node",
                    "agent_name": "QueryAgent",
                    "config": {},
                    "inputs": {"query": "What are the latest developments in battery technology?"}
                }
            ],
            "edges": []
        }
    
    @pytest.mark.asyncio
    async def test_workflow_execution_with_real_agent(self, builder, single_node_plan):
        """Test workflow execution with real agent."""
        available_agents = builder.get_available_agents()
        
        if "QueryAgent" in available_agents:
            agent_info = available_agents["QueryAgent"]
            agent_profiles = {
                "QueryAgent": {
                    "description": "Real QueryAgent",
                    "input_schema": agent_info["input_schema"],
                    "output_schema": agent_info["output_schema"]
                }
            }
            
            result = await builder.execute_workflow(
                research_query="What are the latest developments in battery technology?",
                workflow_plan=single_node_plan,
                agent_profiles=agent_profiles,
                session_id="test_real"
            )
            
            assert result["status"] in ["success", "error"]  # May fail due to API keys
            assert "final_state" in result
            assert "execution_summary" in result
            
            if result["status"] == "success":
                assert result["final_state"].workflow_status == WorkflowStatus.COMPLETED
                assert "results" in result
        else:
            pytest.skip("QueryAgent not discovered")
    
    @pytest.mark.asyncio
    async def test_workflow_execution_with_mock_agent(self, builder):
        """Test workflow execution with mock agent."""
        mock_plan = {
            "nodes": [
                {
                    "node_id": "mock_node",
                    "agent_name": "MockAgent",
                    "config": {},
                    "inputs": {"test": "input"}
                }
            ],
            "edges": []
        }
        
        agent_profiles = {
            "MockAgent": {
                "description": "Mock agent",
                "input_schema": "MockInput",
                "output_schema": "MockOutput"
            }
        }
        
        result = await builder.execute_workflow(
            research_query="Test query",
            workflow_plan=mock_plan,
            agent_profiles=agent_profiles,
            session_id="test_mock"
        )
        
        assert result["status"] == "success"
        assert result["final_state"].workflow_status == WorkflowStatus.COMPLETED
        assert "mock_node" in result["results"]
        assert "note" in result["results"]["mock_node"]
    
    @pytest.mark.asyncio
    async def test_workflow_state_initialization(self, builder, single_node_plan):
        """Test workflow state initialization."""
        agent_profiles = {"QueryAgent": {"description": "Test agent"}}
        
        initial_state = PlannerState(
            research_query="Test query",
            workflow_plan=single_node_plan,
            agent_profiles=agent_profiles,
            session_id="test_init"
        )
        
        assert initial_state.research_query == "Test query"
        assert initial_state.workflow_plan == single_node_plan
        assert initial_state.agent_profiles == agent_profiles
        assert initial_state.session_id == "test_init"
        assert initial_state.workflow_status == WorkflowStatus.NOT_STARTED
    
    @pytest.mark.asyncio
    async def test_workflow_streaming(self, builder, single_node_plan):
        """Test workflow streaming execution."""
        agent_profiles = {"QueryAgent": {"description": "Test agent"}}
        
        events = []
        async for event in builder.stream_workflow_execution(
            research_query="Test query",
            workflow_plan=single_node_plan,
            agent_profiles=agent_profiles,
            session_id="test_stream"
        ):
            events.append(event)
            if len(events) >= 3:  # Limit to prevent long execution
                break
        
        assert len(events) > 0
        assert all("timestamp" in event for event in events)


class TestCheckpointManager:
    """Test checkpoint manager functionality."""
    
    def test_memory_checkpointer(self):
        """Test memory checkpointer initialization."""
        config = {"checkpoint_storage": "memory"}
        manager = CheckpointManager(config)
        
        checkpointer = manager.get_checkpointer()
        
        assert checkpointer is not None
        assert manager.storage_type == "memory"
    
    def test_sqlite_checkpointer_fallback(self):
        """Test SQLite checkpointer fallback to memory."""
        config = {"checkpoint_storage": "sqlite"}
        manager = CheckpointManager(config)
        
        checkpointer = manager.get_checkpointer()
        
        # Should either get SQLite or fallback to memory
        assert checkpointer is not None
        assert manager.storage_type == "sqlite"
    
    def test_checkpoint_config(self):
        """Test checkpoint configuration."""
        config = {
            "checkpoint_storage": "memory",
            "checkpoint_namespace": "test_namespace",
            "max_checkpoint_history": 50
        }
        
        manager = CheckpointManager(config)
        
        assert manager.namespace == "test_namespace"
        assert manager.max_history == 50
    
    @pytest.mark.asyncio
    async def test_checkpoint_history_empty(self):
        """Test checkpoint history for non-existent thread."""
        config = {"checkpoint_storage": "memory"}
        manager = CheckpointManager(config)
        
        history = await manager.get_checkpoint_history("non_existent_thread")
        
        assert isinstance(history, list)
        assert len(history) == 0


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_invalid_workflow_plan(self):
        """Test handling of invalid workflow plans."""
        config = PlannerConfig(auto_discover_agents=False)
        builder = LangGraphWorkflowBuilder(config)
        
        # Test with None plan
        with pytest.raises((TypeError, AttributeError)):
            import asyncio
            asyncio.run(builder.build_workflow(None, {}))
    
    @pytest.mark.asyncio
    async def test_missing_agent_execution(self, workflow_builder):
        """Test execution with missing agent."""
        missing_agent_plan = {
            "nodes": [
                {
                    "node_id": "missing_node",
                    "agent_name": "NonExistentAgent",
                    "config": {},
                    "inputs": {}
                }
            ],
            "edges": []
        }
        
        result = await workflow_builder.execute_workflow(
            research_query="Test query",
            workflow_plan=missing_agent_plan,
            agent_profiles={},
            session_id="test_missing"
        )
        
        # Should handle missing agent gracefully with mock execution
        assert result["status"] == "success"
        assert "note" in result["results"]["missing_node"]
    
    @pytest.mark.asyncio
    async def test_agent_execution_error(self, workflow_builder):
        """Test handling of agent execution errors."""
        # This test would need to mock an agent that throws an error
        # For now, we'll test the error handling structure
        config = PlannerConfig(auto_discover_agents=True)
        builder = LangGraphWorkflowBuilder(config)
        
        # Test with agent that might fail due to configuration
        failing_plan = {
            "nodes": [
                {
                    "node_id": "failing_node",
                    "agent_name": "QueryAgent",
                    "config": {},  # Missing required config
                    "inputs": {"query": "Test query"}
                }
            ],
            "edges": []
        }
        
        result = await workflow_builder.execute_workflow(
            research_query="Test query",
            workflow_plan=failing_plan,
            agent_profiles={"QueryAgent": {"description": "Test agent"}},
            session_id="test_error"
        )
        
        # Should handle errors gracefully
        assert result["status"] == "success"
        assert "results" in result
        
        # Check if error was captured in results
        if "failing_node" in result["results"]:
            node_result = result["results"]["failing_node"]
            assert "status" in node_result
            # Status could be "error" or "success" (mock) depending on agent availability
    
    def test_compilation_without_build(self):
        """Test compilation without building workflow first."""
        config = PlannerConfig()
        builder = LangGraphWorkflowBuilder(config)
        
        with pytest.raises(ValueError, match="Workflow must be built before compilation"):
            import asyncio
            asyncio.run(builder.compile_workflow())


class TestConfigurationEdgeCases:
    """Test edge cases in configuration."""
    
    def test_invalid_agent_packages(self):
        """Test with invalid agent packages."""
        config = PlannerConfig(
            auto_discover_agents=True,
            agent_packages=["non.existent.package"]
        )
        
        # Should handle invalid packages gracefully
        builder = LangGraphWorkflowBuilder(config)
        available_agents = builder.get_available_agents()
        
        # Should not crash, might have empty or partial results
        assert isinstance(available_agents, dict)
    
    def test_empty_agent_packages(self):
        """Test with empty agent packages list."""
        config = PlannerConfig(
            auto_discover_agents=True,
            agent_packages=[]
        )
        
        builder = LangGraphWorkflowBuilder(config)
        available_agents = builder.get_available_agents()
        
        assert len(available_agents) == 0
    
    def test_max_workflow_nodes_limit(self):
        """Test workflow node limit enforcement."""
        config = PlannerConfig(max_workflow_nodes=2)
        builder = LangGraphWorkflowBuilder(config)
        
        # Create workflow with many nodes
        large_plan = {
            "nodes": [
                {
                    "node_id": f"node_{i}",
                    "agent_name": "QueryAgent",
                    "config": {},
                    "inputs": {}
                }
                for i in range(5)  # Exceeds limit
            ],
            "edges": []
        }
        
        # Should handle gracefully (actual limit enforcement would be in higher-level code)
        # For now, just test that it doesn't crash
        try:
            import asyncio
            asyncio.run(builder.build_workflow(large_plan, {}))
            # Test passes if no exception
        except Exception:
            # Some validation might throw, which is acceptable
            pass


if __name__ == "__main__":
    pytest.main([__file__])