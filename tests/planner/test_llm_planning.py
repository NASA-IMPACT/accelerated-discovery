"""
Tests for LLM-based workflow planning components.

This test suite validates the new LLM-based planning capabilities including:
1. WorkflowPlanner with reasoning traces
2. AgentAnalyzer with capability analysis  
3. ResearchPlanner unified interface
4. Integration with existing AKD agents

Note on LLM Mocking:
The tests use patch('langchain_openai.ChatOpenAI.ainvoke') instead of patch.object()
because ChatOpenAI is a Pydantic v2 model that doesn't allow dynamic attribute
assignment. We patch the class method directly to mock LLM responses while still
testing the actual LLM invocation code paths in the planner components.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from akd.planner import (
    ResearchPlanner,
    WorkflowPlanner, 
    AgentAnalyzer,
    ResearchPlanningInput,
    ResearchPlanningOutput,
    WorkflowPlanningInput,
    WorkflowPlanOutput,
    AgentCapability,
    AgentCompatibilityScore,
    PlannerConfig,
    PlannerServiceManager
)
from akd.agents._base import BaseAgent
from akd._base import InputSchema, OutputSchema


class MockAgent(BaseAgent):
    """Mock agent for testing"""
    
    class MockInputSchema(InputSchema):
        """Mock input schema for testing"""
        query: str
        context: str = "default"
    
    class MockOutputSchema(OutputSchema):
        """Mock output schema for testing"""
        result: str
        confidence: float
        metadata: Dict[str, Any]
    
    input_schema = MockInputSchema
    output_schema = MockOutputSchema
    
    async def _arun(self, params):
        return self.MockOutputSchema(
            result=f"Processed: {params.query}",
            confidence=0.9,
            metadata={"processing_time": 1.0}
        )


class TestWorkflowPlanner:
    """Test LLM-based workflow planner"""
    
    @pytest.fixture
    def config(self):
        """Fixture for planner config"""
        return PlannerConfig(
            planning_model="gpt-4o-mini",
            enable_reasoning_traces=True,
            max_workflow_nodes=5
        )
    
    @pytest.fixture
    def workflow_planner(self, config):
        """Fixture for workflow planner"""
        service_manager = PlannerServiceManager(config)
        return WorkflowPlanner(service_manager)
    
    @pytest.fixture
    def sample_agents(self):
        """Fixture for sample agent information"""
        return [
            {
                "agent_name": "QueryAgent",
                "description": "Processes research queries",
                "domain": "query_processing",
                "capabilities": ["query_parsing", "intent_understanding"],
                "input_schema": "QueryInput",
                "output_schema": "QueryOutput",
                "input_fields": {"query": {"type": "str", "required": True}},
                "output_fields": {"processed_query": {"type": "str", "required": True}},
                "performance_metrics": {"estimated_runtime": "fast"}
            },
            {
                "agent_name": "LiteratureSearchAgent",
                "description": "Searches scientific literature",
                "domain": "search",
                "capabilities": ["document_search", "literature_retrieval"],
                "input_schema": "SearchInput",
                "output_schema": "SearchOutput",
                "input_fields": {"query": {"type": "str", "required": True}},
                "output_fields": {"documents": {"type": "List[Dict]", "required": True}},
                "performance_metrics": {"estimated_runtime": "slow"}
            }
        ]
    
    @pytest.mark.asyncio
    async def test_workflow_planning_input_validation(self, workflow_planner):
        """Test workflow planning input validation"""
        
        # Valid input
        valid_input = WorkflowPlanningInput(
            research_query="What are the latest advances in quantum computing?",
            requirements={"depth": "comprehensive"},
            available_agents=[],
            context={"session_id": "test_session"}
        )
        
        assert valid_input.research_query == "What are the latest advances in quantum computing?"
        assert valid_input.requirements["depth"] == "comprehensive"
        assert valid_input.context["session_id"] == "test_session"
    
    @pytest.mark.asyncio
    async def test_workflow_planning_with_mock_llm(self, workflow_planner, sample_agents):
        """Test workflow planning with mocked LLM response"""
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '''
        {
            "workflow_plan": {
                "nodes": [
                    {
                        "node_id": "query_node",
                        "agent_name": "QueryAgent",
                        "description": "Process the research query",
                        "inputs": {"query": "What are the latest advances in quantum computing?"},
                        "config": {},
                        "dependencies": [],
                        "estimated_duration": 30,
                        "confidence": 0.9
                    },
                    {
                        "node_id": "search_node",
                        "agent_name": "LiteratureSearchAgent",
                        "description": "Search for relevant literature",
                        "inputs": {},
                        "config": {},
                        "dependencies": ["query_node"],
                        "estimated_duration": 120,
                        "confidence": 0.8
                    }
                ],
                "edges": [
                    {
                        "source": "query_node",
                        "target": "search_node",
                        "data_mapping": null,
                        "condition": null
                    }
                ]
            },
            "reasoning_trace": "Step 1: Analyze the query... Step 2: Select appropriate agents...",
            "confidence_score": 0.85,
            "estimated_duration": 150,
            "alternative_plans": [],
            "planning_metadata": {
                "agents_considered": ["QueryAgent", "LiteratureSearchAgent"],
                "complexity_level": "medium"
            }
        }
        '''
        
        planning_input = WorkflowPlanningInput(
            research_query="What are the latest advances in quantum computing?",
            requirements={"depth": "comprehensive"},
            available_agents=sample_agents,
            context={}
        )
        
        # Mock LLM response while still testing the actual LLM invocation code path
        # The planner will format prompts, create messages, and parse responses normally
        # This ensures we test: prompt formatting, message creation, LLM invocation, and response parsing
        with patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = mock_response
            result = await workflow_planner.execute_with_error_handling(planning_input)
            
            # Verify the LLM was actually called with proper messages
            mock_ainvoke.assert_called_once()
            call_args = mock_ainvoke.call_args[0][0]  # First argument (messages)
            assert len(call_args) == 2  # Should have system and human messages
            assert call_args[0].type == 'system'
            assert call_args[1].type == 'human'
            
            assert isinstance(result, WorkflowPlanOutput)
            assert result.confidence_score == 0.85
            assert result.estimated_duration == 150
            assert len(result.workflow_plan["nodes"]) == 2
            assert len(result.workflow_plan["edges"]) == 1
            assert "Step 1: Analyze the query" in result.reasoning_trace
    
    @pytest.mark.asyncio
    async def test_workflow_planning_fallback(self, workflow_planner, sample_agents):
        """Test fallback planning when LLM fails"""
        
        planning_input = WorkflowPlanningInput(
            research_query="Test query",
            requirements={},
            available_agents=sample_agents,
            context={}
        )
        
        # Mock LLM to throw exception
        with patch('langchain_openai.ChatOpenAI.ainvoke', side_effect=Exception("LLM failed")):
            result = await workflow_planner.execute_with_error_handling(planning_input)
            
            assert isinstance(result, WorkflowPlanOutput)
            assert result.confidence_score == 0.6  # Fallback confidence
            assert "fallback rule-based planning" in result.reasoning_trace
            assert result.planning_metadata["fallback_used"] is True
    
    @pytest.mark.asyncio
    async def test_plan_validation(self, workflow_planner):
        """Test workflow plan validation"""
        
        # Valid plan
        valid_plan = {
            "nodes": [
                {"node_id": "node1", "agent_name": "Agent1"},
                {"node_id": "node2", "agent_name": "Agent2"}
            ],
            "edges": [
                {"source": "node1", "target": "node2"}
            ]
        }
        
        validation_result = await workflow_planner.validate_plan(valid_plan)
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
        
        # Invalid plan - duplicate node IDs
        invalid_plan = {
            "nodes": [
                {"node_id": "node1", "agent_name": "Agent1"},
                {"node_id": "node1", "agent_name": "Agent2"}  # Duplicate ID
            ],
            "edges": []
        }
        
        validation_result = await workflow_planner.validate_plan(invalid_plan)
        assert validation_result["valid"] is False
        assert "Duplicate node IDs" in validation_result["errors"][0]
    
    @pytest.mark.asyncio
    async def test_alternative_plan_generation(self, workflow_planner, sample_agents):
        """Test generation of alternative workflow plans"""
        
        planning_input = WorkflowPlanningInput(
            research_query="Test query",
            requirements={},
            available_agents=sample_agents,
            context={}
        )
        
        # Mock successful plan generation
        mock_response = Mock()
        mock_response.content = '''
        {
            "workflow_plan": {"nodes": [], "edges": []},
            "reasoning_trace": "Alternative approach...",
            "confidence_score": 0.7,
            "estimated_duration": 100,
            "alternative_plans": [],
            "planning_metadata": {}
        }
        '''
        
        with patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock, return_value=mock_response):
            alternatives = await workflow_planner.generate_alternative_plans(planning_input, 2)
            
            assert len(alternatives) <= 2  # May be fewer due to failures
            assert all(isinstance(alt, dict) for alt in alternatives)


class TestAgentAnalyzer:
    """Test agent capability analyzer"""
    
    @pytest.fixture
    def config(self):
        """Fixture for planner config"""
        return PlannerConfig(
            agent_analysis_model="gpt-4o-mini",
            enable_agent_analysis=True,
            max_parallel_analysis=3
        )
    
    @pytest.fixture
    def agent_analyzer(self, config):
        """Fixture for agent analyzer"""
        return AgentAnalyzer(config)
    
    @pytest.mark.asyncio
    async def test_agent_analysis_basic_info_extraction(self, agent_analyzer):
        """Test basic information extraction from agent class"""
        
        basic_info = agent_analyzer._extract_basic_info(MockAgent)
        
        assert basic_info["class_name"] == "MockAgent"
        assert basic_info["input_schema_name"] == "MockInputSchema"
        assert basic_info["output_schema_name"] == "MockOutputSchema"
        assert "query" in basic_info["input_fields"]
        assert "result" in basic_info["output_fields"]
        assert basic_info["input_fields"]["query"]["type"] == "<class 'str'>"
    
    @pytest.mark.asyncio
    async def test_agent_analysis_with_mock_llm(self, agent_analyzer):
        """Test agent analysis with mocked LLM response"""
        
        mock_response = Mock()
        mock_response.content = '''
        {
            "description": "Mock agent for testing purposes",
            "domain": "testing",
            "capabilities": ["mock_processing", "test_execution"],
            "suggested_upstream": ["DataAgent"],
            "suggested_downstream": ["ValidationAgent"],
            "confidence_score": 0.9,
            "reasoning": "Based on class name and schema analysis"
        }
        '''
        
        with patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock, return_value=mock_response):
            capability = await agent_analyzer.analyze_agent(MockAgent)
            
            assert isinstance(capability, AgentCapability)
            assert capability.agent_name == "MockAgent"
            assert capability.description == "Mock agent for testing purposes"
            assert capability.domain == "testing"
            assert "mock_processing" in capability.capabilities
            assert "DataAgent" in capability.suggested_upstream
            assert capability.confidence_score == 0.9
    
    @pytest.mark.asyncio
    async def test_agent_analysis_fallback(self, agent_analyzer):
        """Test fallback analysis when LLM fails"""
        
        with patch('langchain_openai.ChatOpenAI.ainvoke', side_effect=Exception("LLM failed")):
            capability = await agent_analyzer.analyze_agent(MockAgent)
            
            assert isinstance(capability, AgentCapability)
            assert capability.agent_name == "MockAgent"
            assert capability.domain == "general"  # Fallback domain
            assert capability.confidence_score == 0.6  # Fallback confidence
    
    @pytest.mark.asyncio
    async def test_compatibility_analysis(self, agent_analyzer):
        """Test compatibility analysis between agents"""
        
        # Create two mock capabilities
        cap1 = AgentCapability(
            agent_name="Agent1",
            description="First agent",
            domain="query_processing",
            capabilities=["query_parsing"],
            output_schema="QueryOutput",
            output_fields={"processed_query": {"type": "str"}}
        )
        
        cap2 = AgentCapability(
            agent_name="Agent2",
            description="Second agent",
            domain="search",
            capabilities=["document_search"],
            input_schema="SearchInput",
            input_fields={"query": {"type": "str"}}
        )
        
        mock_response = Mock()
        mock_response.content = '''
        {
            "compatibility_score": 0.8,
            "data_mapping_required": true,
            "mapping_suggestions": {"query": "processed_query"},
            "explanation": "Good compatibility with minor mapping needed",
            "workflow_benefits": "Logical progression from query to search",
            "potential_issues": "Minor schema differences"
        }
        '''
        
        with patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock, return_value=mock_response):
            compatibility = await agent_analyzer.analyze_compatibility(cap1, cap2)
            
            assert isinstance(compatibility, AgentCompatibilityScore)
            assert compatibility.source_agent == "Agent1"
            assert compatibility.target_agent == "Agent2"
            assert compatibility.compatibility_score == 0.8
            assert compatibility.data_mapping_required is True
            assert "query" in compatibility.mapping_suggestions
    
    @pytest.mark.asyncio
    async def test_batch_agent_analysis(self, agent_analyzer):
        """Test parallel analysis of multiple agents"""
        
        agent_classes = [MockAgent, MockAgent]  # Use same class for simplicity
        
        mock_response = Mock()
        mock_response.content = '''
        {
            "description": "Test agent",
            "domain": "testing",
            "capabilities": ["testing"],
            "suggested_upstream": [],
            "suggested_downstream": [],
            "confidence_score": 0.8,
            "reasoning": "Test analysis"
        }
        '''
        
        with patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock, return_value=mock_response):
            capabilities = await agent_analyzer.analyze_agent_batch(agent_classes)
            
            assert len(capabilities) == 2
            assert all(isinstance(cap, AgentCapability) for cap in capabilities)
            assert all(cap.agent_name == "MockAgent" for cap in capabilities)
    
    @pytest.mark.asyncio
    async def test_workflow_path_finding(self, agent_analyzer):
        """Test finding workflow paths between agents"""
        
        # Create mock capabilities
        cap1 = AgentCapability(agent_name="Agent1", description="Start", domain="query_processing")
        cap2 = AgentCapability(agent_name="Agent2", description="Middle", domain="search")
        cap3 = AgentCapability(agent_name="Agent3", description="End", domain="analysis")
        
        available_agents = [cap1, cap2, cap3]
        
        # Mock compatibility responses
        mock_response = Mock()
        mock_response.content = '''
        {
            "compatibility_score": 0.8,
            "data_mapping_required": false,
            "mapping_suggestions": {},
            "explanation": "Good compatibility"
        }
        '''
        
        with patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock, return_value=mock_response):
            path = await agent_analyzer.find_workflow_path(
                "Agent1", "Agent3", available_agents, max_depth=3
            )
            
            assert path is not None
            assert path[0] == "Agent1"
            assert path[-1] == "Agent3"
            assert len(path) <= 3


class TestResearchPlanner:
    """Test unified research planner"""
    
    @pytest.fixture
    def config(self):
        """Fixture for planner config"""
        return PlannerConfig(
            auto_discover_agents=False,  # Disable for testing
            planning_model="gpt-4o-mini",
            enable_reasoning_traces=True
        )
    
    @pytest.fixture
    def research_planner(self, config):
        """Fixture for research planner"""
        planner = ResearchPlanner(config)
        
        # Mock discovered agents
        planner.discovered_agents = {"MockAgent": MockAgent}
        planner.available_agents_info = {
            "MockAgent": {
                "class_name": "MockAgent",
                "module": "test_module",
                "has_input_schema": True,
                "has_output_schema": True,
                "input_schema": "MockInputSchema",
                "output_schema": "MockOutputSchema"
            }
        }
        
        return planner
    
    @pytest.mark.asyncio
    async def test_research_planning_input_validation(self, research_planner):
        """Test research planning input validation"""
        
        valid_input = ResearchPlanningInput(
            research_query="What are the latest advances in AI?",
            requirements={"depth": "comprehensive"},
            constraints={"time_limit": "1_hour"},
            preferred_agents=["MockAgent"],
            session_id="test_session"
        )
        
        assert valid_input.research_query == "What are the latest advances in AI?"
        assert valid_input.requirements["depth"] == "comprehensive"
        assert valid_input.constraints["time_limit"] == "1_hour"
        assert "MockAgent" in valid_input.preferred_agents
    
    @pytest.mark.asyncio
    async def test_end_to_end_research_planning(self, research_planner):
        """Test end-to-end research planning workflow"""
        
        # Mock agent analysis
        mock_capability = AgentCapability(
            agent_name="MockAgent",
            description="Mock agent for testing",
            domain="testing",
            capabilities=["mock_processing"],
            input_schema="MockInputSchema",
            output_schema="MockOutputSchema"
        )
        
        research_planner.agent_capabilities = [mock_capability]
        research_planner.capabilities_cache = {"MockAgent": mock_capability}
        
        # Mock workflow planning
        mock_workflow_output = WorkflowPlanOutput(
            workflow_plan={
                "nodes": [
                    {
                        "node_id": "mock_node",
                        "agent_name": "MockAgent",
                        "description": "Process using mock agent",
                        "inputs": {"query": "test query"},
                        "config": {},
                        "dependencies": [],
                        "estimated_duration": 60,
                        "confidence": 0.8
                    }
                ],
                "edges": []
            },
            reasoning_trace="Selected MockAgent for processing",
            confidence_score=0.8,
            estimated_duration=60,
            alternative_plans=[],
            planning_metadata={"complexity_level": "low"}
        )
        
        with patch.object(research_planner.workflow_planner, 'execute_with_error_handling', return_value=mock_workflow_output):
            planning_input = ResearchPlanningInput(
                research_query="Test research query",
                requirements={"depth": "basic"},
                session_id="test_session"
            )
            
            result = await research_planner.execute_with_error_handling(planning_input)
            
            assert isinstance(result, ResearchPlanningOutput)
            assert result.confidence_score == 0.8
            assert result.estimated_duration == 60
            assert len(result.workflow_plan["nodes"]) == 1
            assert result.workflow_plan["nodes"][0]["agent_name"] == "MockAgent"
            assert result.execution_ready is True
            assert "MockAgent" in result.reasoning_trace
    
    @pytest.mark.asyncio
    async def test_workflow_execution_readiness_check(self, research_planner):
        """Test workflow execution readiness validation"""
        
        # Ready workflow
        ready_plan = {
            "nodes": [
                {"node_id": "node1", "agent_name": "MockAgent"}
            ],
            "edges": []
        }
        
        is_ready = await research_planner._check_execution_readiness(ready_plan)
        assert is_ready is True
        
        # Not ready - missing agent
        not_ready_plan = {
            "nodes": [
                {"node_id": "node1", "agent_name": "UnknownAgent"}
            ],
            "edges": []
        }
        
        is_ready = await research_planner._check_execution_readiness(not_ready_plan)
        assert is_ready is False
        
        # Not ready - empty workflow
        empty_plan = {
            "nodes": [],
            "edges": []
        }
        
        is_ready = await research_planner._check_execution_readiness(empty_plan)
        assert is_ready is False
    
    @pytest.mark.asyncio
    async def test_simple_workflow_creation(self, research_planner):
        """Test creation of simple linear workflows"""
        
        # Mock workflow builder execution
        mock_execution_result = {
            "status": "success",
            "final_state": Mock(workflow_status="completed"),
            "execution_summary": {"progress": 100},
            "results": {"node_0": {"result": "success"}}
        }
        
        with patch.object(research_planner.workflow_builder, 'execute_workflow', return_value=mock_execution_result):
            result = await research_planner.create_simple_workflow(
                research_query="Test query",
                agent_sequence=["MockAgent"],
                session_id="test_session"
            )
            
            assert result["status"] == "success"
            assert result["results"]["node_0"]["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, research_planner):
        """Test dynamic agent registration"""
        
        class NewMockAgent(BaseAgent):
            class Input(InputSchema):
                """Input schema for NewMockAgent"""
                data: str
            
            class Output(OutputSchema):
                """Output schema for NewMockAgent"""
                processed: str
            
            input_schema = Input
            output_schema = Output
            
            async def _arun(self, params):
                return self.Output(processed=f"New: {params.data}")
        
        # Register new agent
        research_planner.register_agent("NewMockAgent", NewMockAgent)
        
        assert "NewMockAgent" in research_planner.discovered_agents
        assert research_planner.discovered_agents["NewMockAgent"] == NewMockAgent
        assert "NewMockAgent" in research_planner.available_agents_info
        
        # Check that capabilities cache was cleared
        assert len(research_planner.agent_capabilities) == 0
        assert len(research_planner.capabilities_cache) == 0


class TestIntegrationWithExistingSystem:
    """Test integration with existing AKD agent system"""
    
    @pytest.mark.asyncio
    async def test_integration_with_existing_agents(self):
        """Test integration with existing AKD agents"""
        
        # This test would require actual AKD agents to be available
        # For now, we test the integration pattern
        
        config = PlannerConfig(
            auto_discover_agents=True,
            agent_packages=["akd.agents"]
        )
        
        planner = ResearchPlanner(config)
        
        # Should discover agents without error
        assert isinstance(planner.discovered_agents, dict)
        assert isinstance(planner.available_agents_info, dict)
        
        # Should be able to get agent info
        available_agents = planner.get_available_agents()
        assert isinstance(available_agents, dict)
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self):
        """Test that existing planner functionality still works"""
        
        from akd.planner import LangGraphWorkflowBuilder, PlannerConfig
        
        config = PlannerConfig(auto_discover_agents=True)
        builder = LangGraphWorkflowBuilder(config)
        
        # Should still work with manual workflow plans
        manual_plan = {
            "nodes": [
                {
                    "node_id": "test_node",
                    "agent_name": "TestAgent",
                    "config": {},
                    "inputs": {"query": "test"}
                }
            ],
            "edges": []
        }
        
        # Should build workflow without error
        graph = await builder.build_workflow(manual_plan, {})
        assert graph is not None
        assert len(graph.nodes) >= 1  # At least control nodes


@pytest.mark.asyncio
async def test_reasoning_trace_capture():
    """Test that reasoning traces are properly captured"""
    
    config = PlannerConfig(
        enable_reasoning_traces=True,
        planning_model="gpt-4o-mini"
    )
    
    service_manager = PlannerServiceManager(config)
    planner = WorkflowPlanner(service_manager)
    
    # Mock LLM with reasoning trace
    mock_response = Mock()
    mock_response.content = '''
    {
        "workflow_plan": {"nodes": [], "edges": []},
        "reasoning_trace": "Reasoning step 1: Analyzed query... Reasoning step 2: Selected agents...",
        "confidence_score": 0.8,
        "estimated_duration": 100,
        "alternative_plans": [],
        "planning_metadata": {"reasoning_model": "gpt-4o-mini"}
    }
    '''
    
    planning_input = WorkflowPlanningInput(
        research_query="Test query with reasoning",
        requirements={},
        available_agents=[],
        context={}
    )
    
    with patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock, return_value=mock_response):
        result = await planner.execute_with_error_handling(planning_input)
        
        assert "Reasoning step 1" in result.reasoning_trace
        assert "Reasoning step 2" in result.reasoning_trace
        assert result.planning_metadata.get("reasoning_model") == "gpt-4o-mini"


if __name__ == "__main__":
    pytest.main([__file__])