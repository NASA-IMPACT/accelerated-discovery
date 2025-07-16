"""
Pytest configuration and fixtures for planner tests.
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock

from akd.planner import PlannerConfig, LangGraphWorkflowBuilder, PlannerState


@pytest.fixture
def planner_config():
    """Fixture for basic planner configuration."""
    return PlannerConfig(
        enable_checkpointing=False,
        enable_monitoring=True,
        execution_strategy="langgraph"
    )


@pytest.fixture
def planner_config_with_checkpointing():
    """Fixture for planner configuration with checkpointing."""
    return PlannerConfig(
        enable_checkpointing=True,
        checkpoint_storage="memory",
        enable_monitoring=True
    )


@pytest.fixture
def workflow_builder(planner_config):
    """Fixture for workflow builder."""
    return LangGraphWorkflowBuilder(planner_config)


@pytest.fixture
def workflow_builder_no_discovery():
    """Fixture for workflow builder without agent discovery."""
    config = PlannerConfig(
        auto_discover_agents=False,
        enable_checkpointing=False
    )
    return LangGraphWorkflowBuilder(config)


@pytest.fixture
def simple_workflow_plan():
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
def single_node_workflow_plan():
    """Fixture for single node workflow plan."""
    return {
        "nodes": [
            {
                "node_id": "test_node",
                "agent_name": "QueryAgent",
                "config": {},
                "inputs": {"query": "Test query"}
            }
        ],
        "edges": []
    }


@pytest.fixture
def mock_agent_profiles():
    """Fixture for mock agent profiles."""
    return {
        "QueryAgent": {
            "description": "Query processing agent",
            "input_schema": "QueryAgentInputSchema",
            "output_schema": "QueryAgentOutputSchema"
        },
        "IntentAgent": {
            "description": "Intent detection agent",
            "input_schema": "IntentInputSchema",
            "output_schema": "IntentOutputSchema"
        }
    }


@pytest.fixture
def planner_state():
    """Fixture for basic planner state."""
    return PlannerState(
        research_query="Test research query",
        session_id="test_session_123"
    )


@pytest.fixture
def planner_state_with_workflow(simple_workflow_plan):
    """Fixture for planner state with initialized workflow."""
    state = PlannerState(
        research_query="Test research query",
        session_id="test_session_123"
    )
    state.initialize_workflow(simple_workflow_plan)
    return state


@pytest.fixture
def mock_agent_class():
    """Fixture for mock agent class."""
    mock_agent = Mock()
    mock_agent.__name__ = "MockAgent"
    mock_agent.__module__ = "test.module"
    
    # Mock schema attributes
    mock_input_schema = Mock()
    mock_input_schema.__name__ = "MockInputSchema"
    mock_agent.input_schema = mock_input_schema
    
    mock_output_schema = Mock()
    mock_output_schema.__name__ = "MockOutputSchema"
    mock_agent.output_schema = mock_output_schema
    
    mock_config_schema = Mock()
    mock_config_schema.__name__ = "MockConfigSchema"
    mock_agent.config_schema = mock_config_schema
    
    # Mock instance
    mock_instance = Mock()
    mock_instance.arun.return_value = {"result": "mock_output"}
    mock_agent.return_value = mock_instance
    
    return mock_agent


@pytest.fixture
def sample_execution_result():
    """Fixture for sample execution result."""
    return {
        "status": "success",
        "final_state": PlannerState(
            research_query="Test query",
            session_id="test_session"
        ),
        "execution_summary": {
            "workflow_status": "completed",
            "progress": 100.0,
            "completed_nodes": 1,
            "total_nodes": 1,
            "errors_count": 0
        },
        "results": {
            "test_node": {
                "status": "success",
                "output": "Test output",
                "timestamp": "2024-01-01T00:00:00"
            }
        }
    }


@pytest.fixture
def checkpoint_config():
    """Fixture for checkpoint configuration."""
    return {
        "checkpoint_storage": "memory",
        "checkpoint_namespace": "test_namespace",
        "max_checkpoint_history": 50
    }


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(mark.name in ['integration', 'slow'] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.name for keyword in ['execution', 'workflow_execution', 'streaming']):
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to tests that test real components
        if any(keyword in item.name for keyword in ['real_agent', 'discovery', 'compilation']):
            item.add_marker(pytest.mark.integration)