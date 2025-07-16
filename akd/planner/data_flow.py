"""
Automatic data flow system for the planner.

This module provides automatic data transformation between connected nodes
using the WaterfallMapper system, enabling seamless data flow in multi-agent
workflows without manual mapping configuration.
"""

from typing import Dict, Any, List, Optional, Type, Callable
from dataclasses import dataclass

from loguru import logger
from pydantic import BaseModel

from akd.agents._base import BaseAgent
from akd.mapping.mappers import WaterfallMapper, MappingInput, MappingOutput
from akd.nodes.states import NodeState, GlobalState
from akd.planner.langgraph_state import PlannerState
from .core import create_temp_model


@dataclass
class DataFlowEdge:
    """
    Represents a data flow edge between two nodes.
    
    This class encapsulates the automatic data transformation logic
    that should be applied when data flows from one node to another.
    """
    source_node_id: str
    target_node_id: str
    source_output_key: str = "agent_output"
    target_input_key: str = "mapped_input"
    mapping_hints: Optional[Dict[str, str]] = None
    transformation_enabled: bool = True
    
    def __post_init__(self):
        """Initialize default values."""
        if self.mapping_hints is None:
            self.mapping_hints = {}


class AutomaticDataFlowManager:
    """
    Manages automatic data flow between nodes in the workflow.
    
    This class analyzes the workflow graph, detects schema mismatches,
    and applies appropriate data transformations using the mapping system.
    """
    
    def __init__(self, enable_caching: bool = True, debug: bool = False):
        """
        Initialize the data flow manager.
        
        Args:
            enable_caching: Whether to enable result caching
            debug: Enable debug logging
        """
        self.mapper = WaterfallMapper()
        self.debug = debug
        self.enable_caching = enable_caching
        
        # Cache for schema analysis and mapping results
        self._schema_cache = {}
        self._mapping_cache = {}
        
        # Edge configurations
        self.data_flow_edges: List[DataFlowEdge] = []
        
        if self.debug:
            logger.debug("AutomaticDataFlowManager initialized")
    
    def analyze_workflow_edges(
        self, 
        workflow_plan: Dict[str, Any], 
        agent_registry: Dict[str, Type[BaseAgent]]
    ) -> List[DataFlowEdge]:
        """
        Analyze workflow edges and create data flow configurations.
        
        Args:
            workflow_plan: The workflow plan with nodes and edges
            agent_registry: Registry of available agents
            
        Returns:
            List of DataFlowEdge configurations
        """
        data_flow_edges = []
        
        edges = workflow_plan.get("edges", [])
        nodes = workflow_plan.get("nodes", [])
        
        # Create node lookup
        node_lookup = {node["node_id"]: node for node in nodes}
        
        for edge in edges:
            source_id = edge.get("source")
            target_id = edge.get("target")
            
            if source_id and target_id:
                # Get agent information
                source_node = node_lookup.get(source_id)
                target_node = node_lookup.get(target_id)
                
                if source_node and target_node:
                    # Analyze schema compatibility
                    compatibility_info = self._analyze_schema_compatibility(
                        source_node, target_node, agent_registry
                    )
                    
                    # Create data flow edge
                    data_flow_edge = DataFlowEdge(
                        source_node_id=source_id,
                        target_node_id=target_id,
                        source_output_key=compatibility_info.get("source_key", "agent_output"),
                        target_input_key=compatibility_info.get("target_key", "mapped_input"),
                        mapping_hints=compatibility_info.get("mapping_hints", {}),
                        transformation_enabled=compatibility_info.get("needs_transformation", True)
                    )
                    
                    data_flow_edges.append(data_flow_edge)
                    
                    if self.debug:
                        logger.debug(f"Created data flow edge: {source_id} -> {target_id}")
        
        self.data_flow_edges = data_flow_edges
        return data_flow_edges
    
    async def apply_data_flow_transformations(
        self, 
        planner_state: PlannerState
    ) -> PlannerState:
        """
        Apply data flow transformations to the planner state.
        
        Args:
            planner_state: Current planner state
            
        Returns:
            Updated planner state with transformed data
        """
        if not self.data_flow_edges:
            return planner_state
        
        for edge in self.data_flow_edges:
            await self._apply_edge_transformation(edge, planner_state)
        
        return planner_state
    
    async def _apply_edge_transformation(
        self, 
        edge: DataFlowEdge, 
        planner_state: PlannerState
    ):
        """
        Apply transformation for a specific edge.
        
        Args:
            edge: The data flow edge configuration
            planner_state: Current planner state
        """
        if not edge.transformation_enabled:
            return
        
        # Get source and target node states
        source_state = planner_state.node_states.get(edge.source_node_id)
        target_state = planner_state.node_states.get(edge.target_node_id)
        
        if not source_state or not target_state:
            return
        
        # Get source data
        source_data = source_state.outputs.get(edge.source_output_key)
        if not source_data:
            return
        
        try:
            # Apply transformation
            transformed_data = await self._transform_data(
                source_data, 
                edge.mapping_hints
            )
            
            # Update target node input
            target_state.inputs[edge.target_input_key] = transformed_data
            
            # Update planner state
            planner_state.node_states = {edge.target_node_id: target_state}
            
            if self.debug:
                logger.debug(f"Applied data transformation: {edge.source_node_id} -> {edge.target_node_id}")
        
        except Exception as e:
            logger.error(f"Data transformation failed for edge {edge.source_node_id} -> {edge.target_node_id}: {e}")
    
    async def _transform_data(
        self, 
        source_data: Any, 
        mapping_hints: Dict[str, str]
    ) -> Any:
        """
        Transform data using the mapping system.
        
        Args:
            source_data: Source data to transform
            mapping_hints: Hints for the transformation
            
        Returns:
            Transformed data
        """
        # If source data is already a BaseModel, use it directly
        if isinstance(source_data, BaseModel):
            source_model = source_data
        else:
            # Create a temporary model
            source_model = create_temp_model(source_data)
        
        # For now, we'll create a generic target schema
        # In a real implementation, this would be based on the target node's input schema
        target_schema = self._create_generic_target_schema(source_data)
        
        # Create mapping input
        mapping_input = MappingInput(
            source_model=source_model,
            target_schema=target_schema,
            mapping_hints=mapping_hints
        )
        
        # Apply mapping
        mapping_result = await self.mapper.arun(mapping_input)
        
        if self.debug:
            logger.debug(f"Mapping confidence: {mapping_result.mapping_confidence}")
        
        return mapping_result.mapped_model
    
    def _analyze_schema_compatibility(
        self, 
        source_node: Dict[str, Any], 
        target_node: Dict[str, Any], 
        agent_registry: Dict[str, Type[BaseAgent]]
    ) -> Dict[str, Any]:
        """
        Analyze schema compatibility between two nodes.
        
        Args:
            source_node: Source node configuration
            target_node: Target node configuration
            agent_registry: Registry of available agents
            
        Returns:
            Compatibility analysis results
        """
        source_agent_name = source_node.get("agent_name")
        target_agent_name = target_node.get("agent_name")
        
        # Get agent classes
        source_agent_class = agent_registry.get(source_agent_name)
        target_agent_class = agent_registry.get(target_agent_name)
        
        if not source_agent_class or not target_agent_class:
            return {
                "needs_transformation": False,
                "source_key": "agent_output",
                "target_key": "mapped_input",
                "mapping_hints": {}
            }
        
        # Get schemas
        source_output_schema = getattr(source_agent_class, 'output_schema', None)
        target_input_schema = getattr(target_agent_class, 'input_schema', None)
        
        # Analyze compatibility
        compatibility_info = {
            "needs_transformation": True,
            "source_key": "agent_output",
            "target_key": "mapped_input",
            "mapping_hints": {}
        }
        
        # If schemas are the same, no transformation needed
        if source_output_schema and target_input_schema:
            if source_output_schema == target_input_schema:
                compatibility_info["needs_transformation"] = False
            else:
                # Generate mapping hints based on schema analysis
                compatibility_info["mapping_hints"] = self._generate_mapping_hints(
                    source_output_schema, target_input_schema
                )
        
        return compatibility_info
    
    def _generate_mapping_hints(
        self, 
        source_schema: Type[BaseModel], 
        target_schema: Type[BaseModel]
    ) -> Dict[str, str]:
        """
        Generate mapping hints based on schema analysis.
        
        Args:
            source_schema: Source schema class
            target_schema: Target schema class
            
        Returns:
            Mapping hints dictionary
        """
        mapping_hints = {}
        
        if not source_schema or not target_schema:
            return mapping_hints
        
        # Get field names
        source_fields = set()
        target_fields = set()
        
        if hasattr(source_schema, 'model_fields'):
            source_fields = set(source_schema.model_fields.keys())
        
        if hasattr(target_schema, 'model_fields'):
            target_fields = set(target_schema.model_fields.keys())
        
        # Find common fields
        common_fields = source_fields.intersection(target_fields)
        
        # Create direct mappings for common fields
        for field in common_fields:
            mapping_hints[field] = field
        
        # Add common patterns
        pattern_mappings = {
            "query": ["question", "search_query", "input_query"],
            "result": ["output", "response", "answer"],
            "results": ["items", "data", "documents"],
            "content": ["text", "body", "description"],
            "score": ["confidence", "relevance", "rating"]
        }
        
        for target_field in target_fields:
            if target_field not in mapping_hints:
                for pattern, alternatives in pattern_mappings.items():
                    if target_field == pattern:
                        # Find matching source field
                        for alt in alternatives:
                            if alt in source_fields:
                                mapping_hints[alt] = target_field
                                break
        
        return mapping_hints
    
    
    def _create_generic_target_schema(self, source_data: Any) -> Type[BaseModel]:
        """Create a generic target schema based on source data."""
        # For now, return a generic schema
        # In a real implementation, this would be based on the target node's input schema
        class GenericTargetSchema(BaseModel):
            transformed_data: Any
        
        return GenericTargetSchema


class DataFlowMiddleware:
    """
    Middleware for intercepting and transforming data flow between nodes.
    
    This class provides a middleware pattern for intercepting data flow
    and applying transformations, validations, or other processing.
    """
    
    def __init__(self, data_flow_manager: AutomaticDataFlowManager):
        """
        Initialize the middleware.
        
        Args:
            data_flow_manager: The data flow manager to use
        """
        self.data_flow_manager = data_flow_manager
        self.middleware_stack = []
    
    def add_middleware(self, middleware_func: Callable):
        """Add a middleware function to the stack."""
        self.middleware_stack.append(middleware_func)
    
    async def process_data_flow(
        self, 
        source_node_id: str, 
        target_node_id: str, 
        data: Any,
        planner_state: PlannerState
    ) -> Any:
        """
        Process data flow through the middleware stack.
        
        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            data: Data to process
            planner_state: Current planner state
            
        Returns:
            Processed data
        """
        processed_data = data
        
        # Apply middleware functions
        for middleware_func in self.middleware_stack:
            try:
                processed_data = await middleware_func(
                    source_node_id, 
                    target_node_id, 
                    processed_data, 
                    planner_state
                )
            except Exception as e:
                logger.error(f"Middleware error: {e}")
                # Continue with unprocessed data
        
        return processed_data


# Example middleware functions
async def validation_middleware(
    source_node_id: str, 
    target_node_id: str, 
    data: Any,
    planner_state: PlannerState
) -> Any:
    """Validate data before transformation."""
    # Add validation logic here
    return data


async def logging_middleware(
    source_node_id: str, 
    target_node_id: str, 
    data: Any,
    planner_state: PlannerState
) -> Any:
    """Log data flow for debugging."""
    logger.info(f"Data flow: {source_node_id} -> {target_node_id}")
    return data


async def caching_middleware(
    source_node_id: str, 
    target_node_id: str, 
    data: Any,
    planner_state: PlannerState
) -> Any:
    """Cache transformation results."""
    # Add caching logic here
    return data