"""
Agent Profile Analyzer for AKD Planner

This module analyzes available agents to create detailed capability profiles
for intelligent workflow planning.
"""

import asyncio
import inspect
from typing import Any, Dict, List, Optional, Type
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from akd._base import UnrestrictedAbstractBase, InputSchema, OutputSchema
from akd.agents._base import BaseAgent
from ..configs.planner_config import PlannerConfig


class AgentCapability(BaseModel):
    """Represents the capabilities of an agent"""
    agent_name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Human-readable description")
    domain: str = Field(..., description="Primary domain (e.g., search, extraction, analysis)")
    capabilities: List[str] = Field(default_factory=list, description="List of specific capabilities")
    input_schema: Optional[str] = Field(None, description="Input schema description")
    output_schema: Optional[str] = Field(None, description="Output schema description")
    input_fields: Dict[str, Any] = Field(default_factory=dict, description="Input field details")
    output_fields: Dict[str, Any] = Field(default_factory=dict, description="Output field details")
    suggested_upstream: List[str] = Field(default_factory=list, description="Suggested upstream agents")
    suggested_downstream: List[str] = Field(default_factory=list, description="Suggested downstream agents")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance characteristics")
    confidence_score: float = Field(0.8, description="Confidence in this analysis")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)


class AgentCompatibilityScore(BaseModel):
    """Compatibility score between two agents"""
    source_agent: str = Field(..., description="Source agent name")
    target_agent: str = Field(..., description="Target agent name")
    compatibility_score: float = Field(..., description="Compatibility score (0-1)")
    data_mapping_required: bool = Field(False, description="Whether data mapping is required")
    mapping_suggestions: Dict[str, Any] = Field(default_factory=dict, description="Suggested field mappings")
    explanation: str = Field("", description="Explanation of compatibility assessment")
    compatibility_notes: str = Field("", description="Additional compatibility notes")


class AgentAnalyzer(UnrestrictedAbstractBase):
    """Analyzes agents to create detailed capability profiles"""
    
    def __init__(self, config: Optional[PlannerConfig] = None):
        super().__init__(config=config or PlannerConfig())
        
        # Initialize LLM for analysis
        self.llm = ChatOpenAI(
            model=getattr(self.config, 'planning_model', 'gpt-4o-mini'),
            temperature=0.1,
            max_tokens=4096
        )
        
        # JSON parser for structured outputs
        self.json_parser = JsonOutputParser()
        
        # Cache for analyzed agents
        self.analysis_cache: Dict[str, AgentCapability] = {}
        self.compatibility_cache: Dict[tuple, AgentCompatibilityScore] = {}
    
    async def analyze_agent(self, agent_class: Type[BaseAgent]) -> AgentCapability:
        """Analyze a single agent class to create capability profile"""
        
        agent_name = agent_class.__name__
        
        # Check cache first
        if agent_name in self.analysis_cache:
            return self.analysis_cache[agent_name]
        
        # Extract basic information
        basic_info = self._extract_basic_info(agent_class)
        
        # Get LLM analysis
        llm_analysis = await self._get_llm_analysis(agent_class, basic_info)
        
        # Combine with code analysis
        capability = AgentCapability(
            agent_name=agent_name,
            description=llm_analysis.get("description", basic_info.get("description", "")),
            domain=llm_analysis.get("domain", self._infer_domain(agent_name)),
            capabilities=llm_analysis.get("capabilities", []),
            input_schema=basic_info.get("input_schema_name"),
            output_schema=basic_info.get("output_schema_name"),
            input_fields=basic_info.get("input_fields", {}),
            output_fields=basic_info.get("output_fields", {}),
            suggested_upstream=llm_analysis.get("suggested_upstream", []),
            suggested_downstream=llm_analysis.get("suggested_downstream", []),
            performance_metrics=self._analyze_performance(agent_class),
            confidence_score=llm_analysis.get("confidence_score", 0.8)
        )
        
        # Cache the result
        self.analysis_cache[agent_name] = capability
        return capability
    
    def _extract_basic_info(self, agent_class: Type[BaseAgent]) -> Dict[str, Any]:
        """Extract basic information from agent class through code inspection"""
        
        info = {
            "class_name": agent_class.__name__,
            "module": agent_class.__module__,
            "description": agent_class.__doc__ or "",
            "input_fields": {},
            "output_fields": {}
        }
        
        # Extract schema information
        if hasattr(agent_class, 'input_schema') and agent_class.input_schema:
            info["input_schema_name"] = agent_class.input_schema.__name__
            info["input_fields"] = self._extract_schema_fields(agent_class.input_schema)
        
        if hasattr(agent_class, 'output_schema') and agent_class.output_schema:
            info["output_schema_name"] = agent_class.output_schema.__name__
            info["output_fields"] = self._extract_schema_fields(agent_class.output_schema)
        
        # Extract method signatures
        if hasattr(agent_class, '_arun'):
            info["arun_signature"] = str(inspect.signature(agent_class._arun))
        
        return info
    
    def _extract_schema_fields(self, schema_class: Type) -> Dict[str, Any]:
        """Extract field information from Pydantic schema"""
        
        fields = {}
        
        if hasattr(schema_class, 'model_fields'):
            for field_name, field_info in schema_class.model_fields.items():
                fields[field_name] = {
                    "type": str(field_info.annotation),
                    "description": field_info.description or "",
                    "required": getattr(field_info, 'is_required', lambda: True)(),
                    "default": getattr(field_info, 'default', None)
                }
        
        return fields
    
    async def _get_llm_analysis(self, agent_class: Type[BaseAgent], basic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze agent capabilities"""
        
        analysis_prompt = f"""
Analyze this AKD research agent and provide a comprehensive capability profile:

Agent: {basic_info['class_name']}
Module: {basic_info['module']}
Description: {basic_info['description']}

Input Schema: {basic_info.get('input_schema_name', 'Unknown')}
Input Fields: {basic_info.get('input_fields', {})}

Output Schema: {basic_info.get('output_schema_name', 'Unknown')}
Output Fields: {basic_info.get('output_fields', {})}

Method Signature: {basic_info.get('arun_signature', 'Unknown')}

Please provide a detailed analysis in JSON format:
{{
    "description": "Clear, concise description of what this agent does",
    "domain": "Primary domain (search, extraction, analysis, synthesis, validation, etc.)",
    "capabilities": ["list", "of", "specific", "capabilities"],
    "suggested_upstream": ["agents", "that", "could", "feed", "into", "this"],
    "suggested_downstream": ["agents", "that", "could", "consume", "this", "output"],
    "confidence_score": 0.9,
    "reasoning": "Explanation of analysis decisions"
}}

Focus on:
1. What specific research task this agent performs
2. What type of data it consumes and produces
3. How it fits into typical research workflows
4. What other agents it works well with
5. Any limitations or requirements
"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert in analyzing AI research agents and their capabilities."),
                HumanMessage(content=analysis_prompt)
            ])
            
            return self.json_parser.parse(response.content)
            
        except Exception as e:
            # Fallback to rule-based analysis
            return self._fallback_analysis(agent_class, basic_info)
    
    def _fallback_analysis(self, agent_class: Type[BaseAgent], basic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        
        agent_name = basic_info['class_name']
        
        # Rule-based domain inference
        domain = self._infer_domain(agent_name)
        
        # Rule-based capability inference
        capabilities = self._infer_capabilities(agent_name, basic_info)
        
        return {
            "description": f"Agent for {domain} tasks",
            "domain": domain,
            "capabilities": capabilities,
            "suggested_upstream": [],
            "suggested_downstream": [],
            "confidence_score": 0.6,
            "reasoning": "Fallback analysis due to LLM failure"
        }
    
    def _infer_domain(self, agent_name: str) -> str:
        """Infer domain from agent name"""
        
        name_lower = agent_name.lower()
        
        if any(keyword in name_lower for keyword in ['search', 'lit', 'literature', 'paper']):
            return "search"
        elif any(keyword in name_lower for keyword in ['extract', 'estimation', 'data']):
            return "extraction"
        elif any(keyword in name_lower for keyword in ['analysis', 'analyze', 'process']):
            return "analysis"
        elif any(keyword in name_lower for keyword in ['query', 'intent', 'understand']):
            return "query_processing"
        elif any(keyword in name_lower for keyword in ['relevancy', 'relevant', 'check']):
            return "validation"
        elif any(keyword in name_lower for keyword in ['synthesis', 'combine', 'merge']):
            return "synthesis"
        else:
            return "general"
    
    def _infer_capabilities(self, agent_name: str, basic_info: Dict[str, Any]) -> List[str]:
        """Infer capabilities from agent name and structure"""
        
        capabilities = []
        name_lower = agent_name.lower()
        
        # Common capability patterns
        if 'search' in name_lower:
            capabilities.extend(['document_search', 'information_retrieval'])
        if 'extract' in name_lower:
            capabilities.extend(['data_extraction', 'structured_parsing'])
        if 'query' in name_lower:
            capabilities.extend(['query_processing', 'intent_understanding'])
        if 'relevancy' in name_lower:
            capabilities.extend(['relevance_scoring', 'quality_assessment'])
        if 'analysis' in name_lower:
            capabilities.extend(['data_analysis', 'pattern_recognition'])
        
        # Add capabilities based on input/output fields
        input_fields = basic_info.get('input_fields', {})
        output_fields = basic_info.get('output_fields', {})
        
        if 'query' in input_fields:
            capabilities.append('query_handling')
        if 'documents' in input_fields:
            capabilities.append('document_processing')
        if 'results' in output_fields:
            capabilities.append('result_generation')
        if 'score' in output_fields:
            capabilities.append('scoring')
        
        return capabilities or ['general_processing']
    
    def _analyze_performance(self, agent_class: Type[BaseAgent]) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        
        # Basic performance heuristics
        performance = {
            "estimated_runtime": "medium",
            "memory_usage": "low",
            "requires_network": True,
            "parallelizable": True,
            "resource_intensive": False
        }
        
        agent_name = agent_class.__name__.lower()
        
        # Adjust based on agent type
        if 'search' in agent_name or 'lit' in agent_name:
            performance["estimated_runtime"] = "slow"
            performance["requires_network"] = True
        elif 'extract' in agent_name:
            performance["estimated_runtime"] = "medium"
            performance["memory_usage"] = "medium"
        elif 'query' in agent_name or 'intent' in agent_name:
            performance["estimated_runtime"] = "fast"
            performance["memory_usage"] = "low"
        
        return performance
    
    async def analyze_compatibility(
        self, 
        source_agent: AgentCapability, 
        target_agent: AgentCapability
    ) -> AgentCompatibilityScore:
        """Analyze compatibility between two agents"""
        
        cache_key = (source_agent.agent_name, target_agent.agent_name)
        
        # Check cache
        if cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]
        
        # Get LLM compatibility analysis
        compatibility_analysis = await self._get_llm_compatibility_analysis(source_agent, target_agent)
        
        # Create compatibility score
        score = AgentCompatibilityScore(
            source_agent=source_agent.agent_name,
            target_agent=target_agent.agent_name,
            compatibility_score=compatibility_analysis.get("compatibility_score", 0.5),
            data_mapping_required=compatibility_analysis.get("data_mapping_required", False),
            mapping_suggestions=compatibility_analysis.get("mapping_suggestions", {}),
            explanation=compatibility_analysis.get("explanation", ""),
            compatibility_notes=compatibility_analysis.get("workflow_benefits", "")
        )
        
        # Cache the result
        self.compatibility_cache[cache_key] = score
        return score
    
    async def _get_llm_compatibility_analysis(
        self, 
        source_agent: AgentCapability, 
        target_agent: AgentCapability
    ) -> Dict[str, Any]:
        """Use LLM to analyze compatibility between agents"""
        
        compatibility_prompt = f"""
Analyze the compatibility between these two research agents for workflow integration:

SOURCE AGENT: {source_agent.agent_name}
- Domain: {source_agent.domain}
- Capabilities: {source_agent.capabilities}
- Output Schema: {source_agent.output_schema}
- Output Fields: {source_agent.output_fields}

TARGET AGENT: {target_agent.agent_name}
- Domain: {target_agent.domain}
- Capabilities: {target_agent.capabilities}
- Input Schema: {target_agent.input_schema}
- Input Fields: {target_agent.input_fields}

Assess compatibility for direct workflow connection and provide JSON response:
{{
    "compatibility_score": 0.8,
    "data_mapping_required": true,
    "mapping_suggestions": {{
        "target_field": "source_field"
    }},
    "explanation": "Detailed explanation of compatibility assessment",
    "workflow_benefits": "How this connection benefits the workflow",
    "potential_issues": "Any potential issues or limitations"
}}

Consider:
1. Data format compatibility
2. Semantic alignment of purpose
3. Logical workflow progression
4. Required data transformations
5. Performance implications
"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert in research workflow optimization and agent integration."),
                HumanMessage(content=compatibility_prompt)
            ])
            
            return self.json_parser.parse(response.content)
            
        except Exception:
            # Fallback to simple heuristic
            return self._fallback_compatibility_analysis(source_agent, target_agent)
    
    def _fallback_compatibility_analysis(
        self, 
        source_agent: AgentCapability, 
        target_agent: AgentCapability
    ) -> Dict[str, Any]:
        """Fallback compatibility analysis"""
        
        # Simple heuristic based on domain compatibility
        domain_compatibility = {
            ("search", "extraction"): 0.9,
            ("extraction", "analysis"): 0.8,
            ("analysis", "synthesis"): 0.8,
            ("query_processing", "search"): 0.9,
            ("validation", "synthesis"): 0.7,
        }
        
        compatibility_score = domain_compatibility.get(
            (source_agent.domain, target_agent.domain), 0.5
        )
        
        return {
            "compatibility_score": compatibility_score,
            "data_mapping_required": source_agent.output_schema != target_agent.input_schema,
            "mapping_suggestions": {},
            "explanation": "Fallback heuristic compatibility analysis"
        }
    
    async def analyze_agent_batch(self, agent_classes: List[Type[BaseAgent]]) -> List[AgentCapability]:
        """Analyze multiple agents in parallel"""
        
        # Limit parallelism to avoid rate limits
        semaphore = asyncio.Semaphore(getattr(self.config, 'max_parallel_analysis', 3))
        
        async def bounded_analyze(agent_class):
            async with semaphore:
                return await self.analyze_agent(agent_class)
        
        tasks = [bounded_analyze(agent_class) for agent_class in agent_classes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        capabilities = []
        for result in results:
            if isinstance(result, AgentCapability):
                capabilities.append(result)
            else:
                # Log error but continue
                if hasattr(self.config, 'debug') and self.config.debug:
                    print(f"Agent analysis failed: {result}")
        
        return capabilities
    
    async def find_workflow_path(
        self, 
        start_agent: str, 
        end_agent: str, 
        available_agents: List[AgentCapability],
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """Find optimal path between two agents through available agents"""
        
        # Simple BFS to find path
        from collections import deque
        
        queue = deque([(start_agent, [start_agent])])
        visited = {start_agent}
        
        while queue:
            current_agent, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current_agent == end_agent:
                return path
            
            # Find compatible next agents
            current_capability = next(
                (cap for cap in available_agents if cap.agent_name == current_agent), None
            )
            
            if current_capability:
                for next_capability in available_agents:
                    if (next_capability.agent_name not in visited and
                        next_capability.agent_name != current_agent):
                        
                        # Check compatibility
                        compatibility = await self.analyze_compatibility(
                            current_capability, next_capability
                        )
                        
                        if compatibility.compatibility_score > 0.6:
                            visited.add(next_capability.agent_name)
                            queue.append((next_capability.agent_name, path + [next_capability.agent_name]))
        
        return None  # No path found
    
    async def suggest_workflow_improvements(
        self, 
        workflow_plan: Dict[str, Any], 
        available_agents: List[AgentCapability]
    ) -> List[str]:
        """Suggest improvements to an existing workflow plan"""
        
        suggestions = []
        
        nodes = workflow_plan.get("nodes", [])
        edges = workflow_plan.get("edges", [])
        
        # Check for missing validation steps
        has_validation = any(
            "validation" in node.get("agent_name", "").lower() or
            "relevancy" in node.get("agent_name", "").lower()
            for node in nodes
        )
        
        if not has_validation:
            validation_agents = [
                agent for agent in available_agents 
                if agent.domain == "validation"
            ]
            if validation_agents:
                suggestions.append(f"Consider adding validation step using {validation_agents[0].agent_name}")
        
        # Check for parallelization opportunities
        independent_nodes = []
        for node in nodes:
            if not node.get("dependencies", []):
                independent_nodes.append(node["node_id"])
        
        if len(independent_nodes) > 1:
            suggestions.append("Consider parallelizing independent nodes for better performance")
        
        # Check for missing synthesis step
        has_synthesis = any(
            "synthesis" in node.get("agent_name", "").lower() or
            "combine" in node.get("agent_name", "").lower()
            for node in nodes
        )
        
        if len(nodes) > 2 and not has_synthesis:
            suggestions.append("Consider adding synthesis step to combine results")
        
        return suggestions
    
    async def _arun(self, params: Any) -> Any:
        """Not used directly - utility class for analysis"""
        pass