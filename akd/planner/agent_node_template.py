"""
Agent-to-NodeTemplate wrapper for the planner system.

This module provides a wrapper that converts any BaseAgent into a NodeTemplate,
allowing the planner to use agents while maintaining the NodeTemplate architecture
with proper state management, guardrails, and mapping integration.
"""

import uuid
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from pydantic import BaseModel

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig
from akd.mapping.mappers import WaterfallMapper, MappingInput, MappingOutput
from akd.nodes.states import NodeState, GlobalState
from akd.nodes.templates import AbstractNodeTemplate
from akd.common_types import CallableSpec
from .core import create_temp_model


class AgentNodeTemplate(AbstractNodeTemplate):
    """
    NodeTemplate wrapper for BaseAgent instances.
    
    This class allows any BaseAgent to be used as a NodeTemplate, providing:
    - Automatic input/output mapping between NodeState and agent schemas
    - Integration with the mapping system for data transformation
    - Proper state management following NodeTemplate patterns
    - Support for guardrails and validation
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        node_id: Optional[str] = None,
        input_guardrails: List[CallableSpec] = None,
        output_guardrails: List[CallableSpec] = None,
        enable_mapping: bool = True,
        mutation: bool = True,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize AgentNodeTemplate wrapper.
        
        Args:
            agent: The BaseAgent instance to wrap
            node_id: Optional node ID, generated if not provided
            input_guardrails: Input validation rules
            output_guardrails: Output validation rules
            enable_mapping: Whether to enable automatic data mapping
            mutation: Whether to mutate the global state
            debug: Enable debug logging
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            node_id=node_id,
            input_guardrails=input_guardrails or [],
            output_guardrails=output_guardrails or [],
            mutation=mutation,
            debug=debug,
            **kwargs
        )
        
        self.agent = agent
        self.enable_mapping = enable_mapping
        
        # Initialize mapper if enabled
        if self.enable_mapping:
            self.mapper = WaterfallMapper()
        else:
            self.mapper = None
            
        # Cache agent schemas for performance
        self.agent_input_schema = getattr(agent, 'input_schema', None)
        self.agent_output_schema = getattr(agent, 'output_schema', None)
        
        if self.debug:
            logger.debug(f"AgentNodeTemplate initialized with agent: {agent.__class__.__name__}")
    
    async def _execute(self, node_state: NodeState, global_state: GlobalState) -> NodeState:
        """
        Execute the wrapped agent with proper input/output mapping.
        
        Args:
            node_state: Current node state
            global_state: Global workflow state
            
        Returns:
            Updated node state with agent results
        """
        try:
            # Step 1: Prepare agent input
            agent_input = await self._prepare_agent_input(node_state, global_state)
            
            if self.debug:
                logger.debug(f"Agent input prepared: {type(agent_input).__name__}")
            
            # Step 2: Execute agent
            agent_output = await self.agent.arun(agent_input)
            
            if self.debug:
                logger.debug(f"Agent executed successfully: {type(agent_output).__name__}")
            
            # Step 3: Map agent output back to node state
            updated_node_state = await self._process_agent_output(
                agent_output, node_state, global_state
            )
            
            return updated_node_state
            
        except Exception as e:
            logger.error(f"Agent execution failed in node {self.node_id}: {e}")
            
            # Add error to node state in the format expected by tests
            node_state.outputs["error"] = {
                "type": "agent_execution_error",
                "message": str(e),
                "agent_class": self.agent.__class__.__name__
            }
            
            # Also add status for compatibility with tests
            node_state.outputs["status"] = "error"
            
            return node_state
    
    async def _prepare_agent_input(
        self, 
        node_state: NodeState, 
        global_state: GlobalState
    ) -> Any:
        """
        Prepare input for the agent from node state and global state.
        
        Args:
            node_state: Current node state
            global_state: Global workflow state
            
        Returns:
            Prepared input for the agent
        """
        # If agent doesn't have input schema, use dict input
        if not self.agent_input_schema:
            return {
                "query": node_state.inputs.get("query", ""),
                **node_state.inputs
            }
        
        # Try to map node state to agent input schema
        if self.enable_mapping and self.mapper:
            try:
                # Create a temporary model from node state inputs
                temp_model = create_temp_model(node_state.inputs, exclude_fields=["mapping_hints"])
                
                # Map to agent input schema
                mapping_input = MappingInput(
                    source_model=temp_model,
                    target_schema=self.agent_input_schema,
                    mapping_hints=node_state.inputs.get("mapping_hints")
                )
                
                mapping_result = await self.mapper.arun(mapping_input)
                
                if self.debug:
                    logger.debug(f"Mapping confidence: {mapping_result.mapping_confidence}")
                
                return mapping_result.mapped_model
                
            except Exception as e:
                logger.warning(f"Mapping failed for agent input: {e}")
                # Fallback to direct instantiation
        
        # Direct instantiation with available fields
        try:
            return self.agent_input_schema(**node_state.inputs)
        except Exception as e:
            logger.warning(f"Direct instantiation failed: {e}")
            # Final fallback - create with available fields
            return self._create_fallback_input(node_state.inputs)
    
    async def _process_agent_output(
        self, 
        agent_output: Any, 
        node_state: NodeState, 
        global_state: GlobalState
    ) -> NodeState:
        """
        Process agent output and update node state.
        
        Args:
            agent_output: Output from the agent
            node_state: Current node state
            global_state: Global workflow state
            
        Returns:
            Updated node state
        """
        # Store raw agent output
        node_state.outputs["agent_output"] = agent_output
        
        # If output is a Pydantic model, convert to dict
        if isinstance(agent_output, BaseModel):
            output_dict = agent_output.model_dump()
        else:
            output_dict = agent_output if isinstance(agent_output, dict) else {"result": agent_output}
        
        # Merge output into node state
        node_state.outputs.update(output_dict)
        
        # Add metadata
        node_state.outputs["agent_metadata"] = {
            "agent_class": self.agent.__class__.__name__,
            "agent_module": self.agent.__class__.__module__,
            "node_id": self.node_id,
            "mapping_enabled": self.enable_mapping
        }
        
        return node_state
    
    
    def _create_fallback_input(self, inputs: Dict[str, Any]) -> Any:
        """Create fallback input when schema instantiation fails."""
        
        if not self.agent_input_schema:
            return inputs
        
        # Try to create with minimal required fields
        try:
            # Get field info from schema
            if hasattr(self.agent_input_schema, 'model_fields'):
                fields = self.agent_input_schema.model_fields
                
                # Create dict with available fields
                fallback_data = {}
                for field_name, field_info in fields.items():
                    if field_name in inputs:
                        fallback_data[field_name] = inputs[field_name]
                    elif hasattr(field_info, 'default') and field_info.default is not None:
                        fallback_data[field_name] = field_info.default
                
                return self.agent_input_schema(**fallback_data)
                
        except Exception as e:
            logger.warning(f"Fallback input creation failed: {e}")
        
        # Final fallback - return inputs as dict
        return inputs
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the wrapped agent."""
        return {
            "agent_class": self.agent.__class__.__name__,
            "agent_module": self.agent.__class__.__module__,
            "node_id": self.node_id,
            "has_input_schema": self.agent_input_schema is not None,
            "has_output_schema": self.agent_output_schema is not None,
            "input_schema": self.agent_input_schema.__name__ if self.agent_input_schema else None,
            "output_schema": self.agent_output_schema.__name__ if self.agent_output_schema else None,
            "mapping_enabled": self.enable_mapping
        }
    
    def __repr__(self) -> str:
        return f"AgentNodeTemplate(agent={self.agent.__class__.__name__}, node_id={self.node_id})"


class AgentNodeTemplateFactory:
    """Factory for creating AgentNodeTemplate instances from agents."""
    
    @staticmethod
    def create_from_agent(
        agent: BaseAgent,
        node_id: Optional[str] = None,
        input_guardrails: List[CallableSpec] = None,
        output_guardrails: List[CallableSpec] = None,
        enable_mapping: bool = True,
        mutation: bool = True,
        debug: bool = False,
        **kwargs
    ) -> AgentNodeTemplate:
        """
        Create an AgentNodeTemplate from a BaseAgent instance.
        
        Args:
            agent: The BaseAgent to wrap
            node_id: Optional node ID
            input_guardrails: Input validation rules
            output_guardrails: Output validation rules
            enable_mapping: Whether to enable automatic data mapping
            mutation: Whether to mutate the global state
            debug: Enable debug logging
            **kwargs: Additional keyword arguments
            
        Returns:
            AgentNodeTemplate instance
        """
        return AgentNodeTemplate(
            agent=agent,
            node_id=node_id,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            enable_mapping=enable_mapping,
            mutation=mutation,
            debug=debug,
            **kwargs
        )
    
    @staticmethod
    def create_from_agent_class(
        agent_class: Type[BaseAgent],
        agent_config: Optional[BaseAgentConfig] = None,
        node_id: Optional[str] = None,
        input_guardrails: List[CallableSpec] = None,
        output_guardrails: List[CallableSpec] = None,
        enable_mapping: bool = True,
        mutation: bool = True,
        debug: bool = False,
        **kwargs
    ) -> AgentNodeTemplate:
        """
        Create an AgentNodeTemplate from a BaseAgent class.
        
        Args:
            agent_class: The BaseAgent class to instantiate and wrap
            agent_config: Configuration for the agent
            node_id: Optional node ID
            input_guardrails: Input validation rules
            output_guardrails: Output validation rules
            enable_mapping: Whether to enable automatic data mapping
            mutation: Whether to mutate the global state
            debug: Enable debug logging
            **kwargs: Additional keyword arguments
            
        Returns:
            AgentNodeTemplate instance
        """
        # Create agent instance
        if agent_config is None:
            # Try to create default config
            if hasattr(agent_class, 'config_schema') and agent_class.config_schema:
                agent_config = agent_class.config_schema()
            else:
                agent_config = BaseAgentConfig()
        
        agent_instance = agent_class(config=agent_config)
        
        return AgentNodeTemplateFactory.create_from_agent(
            agent=agent_instance,
            node_id=node_id,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            enable_mapping=enable_mapping,
            mutation=mutation,
            debug=debug,
            **kwargs
        )