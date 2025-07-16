"""
Core planner services and utilities.

This module provides shared services and utilities for the planner system,
reducing code duplication and improving maintainability.
"""

import asyncio
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from loguru import logger
from pydantic import BaseModel

from akd.agents._base import BaseAgent
from ..configs.planner_config import PlannerConfig


class PlannerError(Exception):
    """Base exception for planner errors."""
    pass


class PlanningError(PlannerError):
    """Error in planning operations."""
    pass


class ExecutionError(PlannerError):
    """Error in execution operations."""
    pass


class LLMService:
    """Centralized LLM service for the planner system."""
    
    def __init__(self, config: PlannerConfig):
        self.config = config
        self._llm = None
        self._json_parser = JsonOutputParser()
        self._rate_limiter = asyncio.Semaphore(config.max_parallel_analysis)
    
    @property
    def llm(self) -> ChatOpenAI:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance with optimal settings."""
        model_name = self.config.planning_model
        
        # Use reasoning models for complex planning
        if self.config.enable_reasoning_traces:
            reasoning_models = ["o3-mini", "o3", "o1-preview", "o1-mini"]
            if any(model in model_name for model in reasoning_models):
                model_name = next(model for model in reasoning_models if model in model_name)
        
        return ChatOpenAI(
            model=model_name,
            temperature=0.1,
            max_tokens=8192,
            request_timeout=30,  # Default timeout
            max_retries=3,  # Default retries
            model_kwargs={
                "response_format": {"type": "json_object"}
                if "gpt" in model_name else None
            }
        )
    
    async def invoke_with_retry(self, messages: List, max_retries: int = 3) -> Dict[str, Any]:
        """Invoke LLM with retry logic and rate limiting."""
        async with self._rate_limiter:
            for attempt in range(max_retries):
                try:
                    response = await self.llm.ainvoke(messages)
                    return self._json_parser.parse(response.content)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"LLM invocation failed after {max_retries} attempts: {e}")
                        raise PlanningError(f"LLM service unavailable: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def invoke_batch(self, batch_requests: List[List]) -> List[Dict[str, Any]]:
        """Process multiple LLM requests in parallel with rate limiting."""
        semaphore = asyncio.Semaphore(self.config.max_parallel_analysis)
        
        async def bounded_invoke(messages):
            async with semaphore:
                return await self.invoke_with_retry(messages)
        
        tasks = [bounded_invoke(messages) for messages in batch_requests]
        return await asyncio.gather(*tasks, return_exceptions=True)


class AgentRegistry:
    """Centralized agent registry with lazy loading and caching."""
    
    def __init__(self, config: PlannerConfig):
        self.config = config
        self._agents: Dict[str, Type[BaseAgent]] = {}
        self._agent_info_cache: Dict[str, Dict[str, Any]] = {}
        self._discovery_complete = False
    
    def register_agent(self, name: str, agent_class: Type[BaseAgent]):
        """Register an agent class."""
        self._agents[name] = agent_class
        # Clear cache for this agent
        self._agent_info_cache.pop(name, None)
    
    def get_agent(self, name: str) -> Optional[Type[BaseAgent]]:
        """Get agent class by name."""
        if not self._discovery_complete:
            self._lazy_discover()
        return self._agents.get(name)
    
    def get_all_agents(self) -> Dict[str, Type[BaseAgent]]:
        """Get all registered agents."""
        if not self._discovery_complete:
            self._lazy_discover()
        return self._agents.copy()
    
    def get_agent_info(self, name: str) -> Dict[str, Any]:
        """Get cached agent information."""
        if name not in self._agent_info_cache:
            agent_class = self.get_agent(name)
            if agent_class:
                self._agent_info_cache[name] = self._extract_agent_info(agent_class)
        return self._agent_info_cache.get(name, {})
    
    def _lazy_discover(self):
        """Lazy agent discovery."""
        if self._discovery_complete:
            return
        
        # Prevent circular dependency by using direct discovery instead of LangGraphWorkflowBuilder
        self._discovery_complete = True  # Set early to prevent recursion
        
        try:
            # Direct agent discovery without creating another PlannerServiceManager
            self._discover_agents_directly()
        except Exception as e:
            logger.warning(f"Agent discovery failed: {e}")
    
    def _discover_agents_directly(self):
        """Direct agent discovery without circular dependencies."""
        import importlib
        import inspect
        
        # Discover agents in configured packages
        for package_name in self.config.agent_packages:
            try:
                self._discover_agents_in_package(package_name)
            except Exception as e:
                logger.warning(f"Could not discover agents in package {package_name}: {e}")
    
    def _discover_agents_in_package(self, package_name: str):
        """Discover agents in a specific package."""
        import importlib
        import inspect
        
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
                            logger.warning(f"Could not import module {modname}: {e}")
            else:
                # Single module
                self._register_agents_from_module(package)
                
        except Exception as e:
            logger.warning(f"Error discovering agents in package {package_name}: {e}")
    
    def _register_agents_from_module(self, module):
        """Register all agent classes from a module."""
        import inspect
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseAgent) and 
                obj is not BaseAgent and
                not name.startswith('_')):
                
                # Register the agent with its class name
                self._agents[name] = obj
                
                # Also register with common name patterns
                if name.endswith('Agent'):
                    short_name = name[:-5]  # Remove 'Agent' suffix
                    self._agents[short_name] = obj
    
    def _extract_agent_info(self, agent_class: Type[BaseAgent]) -> Dict[str, Any]:
        """Extract agent information for caching."""
        return {
            "class_name": agent_class.__name__,
            "module": agent_class.__module__,
            "has_input_schema": hasattr(agent_class, 'input_schema'),
            "has_output_schema": hasattr(agent_class, 'output_schema'),
            "input_schema": getattr(agent_class, 'input_schema', None),
            "output_schema": getattr(agent_class, 'output_schema', None),
            "description": agent_class.__doc__ or ""
        }


class ValidationService:
    """Centralized validation service."""
    
    @staticmethod
    def validate_workflow_plan(workflow_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow plan structure."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        nodes = workflow_plan.get("nodes", [])
        edges = workflow_plan.get("edges", [])
        
        # Validate basic structure
        if not nodes:
            result["errors"].append("Workflow must have at least one node")
            result["valid"] = False
            return result
        
        # Validate node IDs are unique
        node_ids = [node["node_id"] for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            result["errors"].append("Duplicate node IDs found")
            result["valid"] = False
        
        # Validate edges reference existing nodes
        for edge in edges:
            if edge["source"] not in node_ids:
                result["errors"].append(f"Edge source '{edge['source']}' not found")
                result["valid"] = False
            if edge["target"] not in node_ids:
                result["errors"].append(f"Edge target '{edge['target']}' not found")
                result["valid"] = False
        
        # Check for cycles
        if ValidationService._has_cycles(nodes, edges):
            result["errors"].append("Workflow contains cycles")
            result["valid"] = False
        
        return result
    
    @staticmethod
    def _has_cycles(nodes: List[Dict], edges: List[Dict]) -> bool:
        """Check for cycles in workflow graph."""
        # Create adjacency list
        graph = {node["node_id"]: [] for node in nodes}
        for edge in edges:
            graph[edge["source"]].append(edge["target"])
        
        # DFS cycle detection
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node_id in graph:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        
        return False


class PlannerServiceManager:
    """Manager for all planner services."""
    
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.llm_service = LLMService(config)
        self.agent_registry = AgentRegistry(config)
        self.validation_service = ValidationService()
        
        # Performance metrics
        self.metrics = {
            "plans_generated": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "errors": 0
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {key: 0 for key in self.metrics}


class BasePlannerComponent(ABC):
    """Base class for planner components with shared functionality."""
    
    def __init__(self, service_manager: PlannerServiceManager):
        self.services = service_manager
        self.config = service_manager.config
    
    @abstractmethod
    async def _execute(self, *args, **kwargs) -> Any:
        """Execute the component's main functionality."""
        pass
    
    async def execute_with_error_handling(self, *args, **kwargs) -> Any:
        """Execute with standardized error handling."""
        try:
            return await self._execute(*args, **kwargs)
        except PlannerError:
            self.services.metrics["errors"] += 1
            raise
        except Exception as e:
            self.services.metrics["errors"] += 1
            logger.error(f"Unexpected error in {self.__class__.__name__}: {e}")
            raise PlannerError(f"Component execution failed: {e}")


class CacheManager:
    """Simple cache manager for planner results."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now().timestamp() - entry["timestamp"] < self.ttl_seconds:
                return entry["value"]
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value."""
        # Simple LRU eviction
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]
        
        self._cache[key] = {
            "value": value,
            "timestamp": datetime.now().timestamp()
        }
    
    def clear(self):
        """Clear all cached values."""
        self._cache.clear()


def create_temp_model(data: Any, exclude_fields: Optional[List[str]] = None) -> BaseModel:
    """
    Create a temporary Pydantic model from data.
    
    Args:
        data: Data to convert to model (dict or other)
        exclude_fields: Fields to exclude from the model
        
    Returns:
        Temporary Pydantic model instance
    """
    exclude_fields = exclude_fields or []
    
    if isinstance(data, dict):
        # Create dynamic model class
        class TempModel(BaseModel):
            pass
        
        # Add fields dynamically, excluding specified fields
        for key, value in data.items():
            if key not in exclude_fields:
                setattr(TempModel, key, value)
        
        # Filter data for instantiation
        filtered_data = {k: v for k, v in data.items() if k not in exclude_fields}
        return TempModel(**filtered_data)
    else:
        # Wrap non-dict data
        class TempModel(BaseModel):
            data: Any
        
        return TempModel(data=data)
