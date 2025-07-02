"""
Planner agent for coordinating multi-agent research workflows.
Handles plan generation, execution coordination, and human interaction.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from akd.agents._base import BaseAgent
from akd.utils import AsyncRunMixin

from .states import (
    AgentStatus,
    GlobalWorkflowState,
    HumanInteraction,
    PlanStep,
    ResearchPlan,
)


class PlanGenerationInput(BaseModel):
    """Input for plan generation"""

    research_goal: str = Field(..., description="High-level research goal")
    research_domain: str = Field(default="general", description="Research domain/field")
    user_constraints: Dict[str, Any] = Field(
        default_factory=dict, description="User constraints and preferences"
    )
    available_agents: List[str] = Field(
        default_factory=list, description="Available agent types"
    )


class PlanGenerationOutput(BaseModel):
    """Output from plan generation"""

    plan: ResearchPlan = Field(..., description="Generated research plan")
    confidence_score: float = Field(..., description="Confidence in the plan quality")
    reasoning: str = Field(..., description="Reasoning behind the plan")


class PlannerAgent(BaseAgent[PlanGenerationInput, PlanGenerationOutput], AsyncRunMixin):
    """
    Planner agent responsible for:
    1. Generating research plans based on user goals
    2. Coordinating agent execution
    3. Managing human interactions
    4. Adapting plans based on results and feedback
    """

    input_schema = PlanGenerationInput
    output_schema = PlanGenerationOutput

    def __init__(
        self,
        llm_client: Any = None,
        available_agents: List[str] = None,
        agent_capabilities: Dict[str, List[str]] = None,
        default_plan_template: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.llm_client = llm_client
        self.available_agents = available_agents or [
            "literature_search",
            "extraction",
            "analysis",
            "synthesis",
            "validation",
        ]
        self.agent_capabilities = agent_capabilities or self._get_default_capabilities()
        self.default_plan_template = (
            default_plan_template or self._get_default_plan_template()
        )

    @property
    def memory(self) -> Any:
        """Return empty memory for planner agent"""
        return []

    async def get_response_async(
        self, response_model: type[PlanGenerationOutput] = None, **kwargs
    ) -> PlanGenerationOutput:
        """
        Get response for plan generation (not used directly, plan generation goes through _arun)
        """
        # This method is required by BaseAgent but not used in the planner workflow
        # The actual plan generation happens in _arun method
        raise NotImplementedError("Use arun() method for plan generation")

    def _get_default_capabilities(self) -> Dict[str, List[str]]:
        """Define default capabilities for each agent type"""
        return {
            "literature_search": [
                "search_scientific_papers",
                "search_web_content",
                "extract_paper_metadata",
                "filter_by_relevancy",
            ],
            "extraction": [
                "extract_structured_data",
                "extract_key_findings",
                "extract_methodology",
                "extract_results",
            ],
            "analysis": [
                "analyze_trends",
                "statistical_analysis",
                "comparative_analysis",
                "identify_gaps",
            ],
            "synthesis": [
                "synthesize_findings",
                "generate_summary",
                "create_report",
                "identify_patterns",
            ],
            "validation": [
                "fact_check",
                "verify_sources",
                "validate_methodology",
                "assess_quality",
            ],
        }

    def _get_default_plan_template(self) -> Dict[str, Any]:
        """Define default plan template for research workflows"""
        return {
            "literature_search_first": True,
            "require_human_approval": True,
            "parallel_execution": True,
            "quality_gates": ["extraction", "synthesis"],
            "final_review": True,
        }

    async def _arun(
        self,
        params: PlanGenerationInput,
        **kwargs: Any,
    ) -> PlanGenerationOutput:
        """Generate a research plan based on the input parameters"""

        # Create plan ID and basic structure
        plan_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        if self.debug:
            logger.debug(f"Generating plan for goal: {params.research_goal}")

        # Generate plan steps using LLM if available, otherwise use template
        if self.llm_client:
            plan = await self._generate_plan_with_llm(params, plan_id, timestamp)
        else:
            plan = self._generate_template_plan(params, plan_id, timestamp)

        # Calculate confidence score
        confidence_score = self._calculate_plan_confidence(plan, params)

        # Generate reasoning
        reasoning = self._generate_plan_reasoning(plan, params)

        return PlanGenerationOutput(
            plan=plan,
            confidence_score=confidence_score,
            reasoning=reasoning,
        )

    async def _generate_plan_with_llm(
        self,
        params: PlanGenerationInput,
        plan_id: str,
        timestamp: str,
    ) -> ResearchPlan:
        """Generate plan using LLM"""

        # Create prompt for plan generation
        prompt = self._create_planning_prompt(params)

        try:
            # Use LLM to generate structured plan
            response = await self.llm_client.agenerate([prompt])

            # Parse LLM response into plan structure
            plan = self._parse_llm_plan_response(response, params, plan_id, timestamp)

        except Exception as e:
            if self.debug:
                logger.warning(
                    f"LLM plan generation failed: {e}. Using template fallback."
                )
            plan = self._generate_template_plan(params, plan_id, timestamp)

        return plan

    def _create_planning_prompt(self, params: PlanGenerationInput) -> str:
        """Create prompt for LLM-based plan generation"""

        capabilities_text = "\n".join(
            [
                f"- {agent}: {', '.join(caps)}"
                for agent, caps in self.agent_capabilities.items()
            ]
        )

        return f"""
        You are a research planning expert. Generate a detailed research plan for the following goal:
        
        Research Goal: {params.research_goal}
        Research Domain: {params.research_domain}
        User Constraints: {params.user_constraints}
        
        Available Agents and Capabilities:
        {capabilities_text}
        
        Create a plan with these characteristics:
        1. Start with literature search to gather foundational knowledge
        2. Include structured extraction of key information
        3. Add analysis and synthesis steps
        4. Include human review points at critical junctures
        5. Ensure logical dependencies between steps
        6. Prioritize steps appropriately
        
        Generate a JSON response with the following structure:
        {{
            "title": "Plan title",
            "description": "Plan description",
            "steps": [
                {{
                    "step_id": "unique_id",
                    "agent_type": "agent_name",
                    "description": "what this step does",
                    "inputs": {{}},
                    "dependencies": ["step_ids"],
                    "priority": 1-10,
                    "human_review_required": true/false
                }}
            ]
        }}
        """

    def _parse_llm_plan_response(
        self,
        response: Any,
        params: PlanGenerationInput,
        plan_id: str,
        timestamp: str,
    ) -> ResearchPlan:
        """Parse LLM response into ResearchPlan structure"""

        # This would parse the LLM JSON response
        # For now, fall back to template-based generation
        return self._generate_template_plan(params, plan_id, timestamp)

    def _generate_template_plan(
        self,
        params: PlanGenerationInput,
        plan_id: str,
        timestamp: str,
    ) -> ResearchPlan:
        """Generate plan using predefined template"""

        steps = {}

        # Step 1: Literature Search (always first)
        lit_search_id = "lit_search_001"
        steps[lit_search_id] = PlanStep(
            step_id=lit_search_id,
            agent_type="literature_search",
            description=f"Conduct comprehensive literature search for: {params.research_goal}",
            inputs={"query": params.research_goal, "max_results": 50},
            dependencies=[],
            priority=10,
            human_review_required=False,
        )

        # Step 2: Human Review of Literature Search
        review_lit_id = "review_lit_001"
        steps[review_lit_id] = PlanStep(
            step_id=review_lit_id,
            agent_type="human_review",
            description="Review literature search results and provide feedback",
            inputs={"results_to_review": "literature_search_results"},
            dependencies=[lit_search_id],
            priority=9,
            human_review_required=True,
        )

        # Step 3: Structured Extraction
        extraction_id = "extraction_001"
        steps[extraction_id] = PlanStep(
            step_id=extraction_id,
            agent_type="extraction",
            description="Extract structured information from literature sources",
            inputs={"sources": "filtered_literature", "schema": "auto_detect"},
            dependencies=[review_lit_id],
            priority=8,
            human_review_required=False,
        )

        # Step 4: Analysis
        analysis_id = "analysis_001"
        steps[analysis_id] = PlanStep(
            step_id=analysis_id,
            agent_type="analysis",
            description="Analyze extracted data for patterns and insights",
            inputs={"extracted_data": "structured_extractions"},
            dependencies=[extraction_id],
            priority=7,
            human_review_required=False,
        )

        # Step 5: Synthesis
        synthesis_id = "synthesis_001"
        steps[synthesis_id] = PlanStep(
            step_id=synthesis_id,
            agent_type="synthesis",
            description="Synthesize findings into coherent research summary",
            inputs={"analysis_results": "analysis_output"},
            dependencies=[analysis_id],
            priority=6,
            human_review_required=True,
        )

        return ResearchPlan(
            plan_id=plan_id,
            title=f"Research Plan: {params.research_goal}",
            description=f"Comprehensive research plan for investigating {params.research_goal}",
            steps=steps,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _calculate_plan_confidence(
        self,
        plan: ResearchPlan,
        params: PlanGenerationInput,
    ) -> float:
        """Calculate confidence score for the generated plan"""

        confidence_factors = []

        # Factor 1: Plan completeness
        if len(plan.steps) >= 3:
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.1)

        # Factor 2: Dependency structure
        has_dependencies = any(step.dependencies for step in plan.steps.values())
        confidence_factors.append(0.2 if has_dependencies else 0.1)

        # Factor 3: Human review points
        has_human_review = any(
            step.human_review_required for step in plan.steps.values()
        )
        confidence_factors.append(0.2 if has_human_review else 0.1)

        # Factor 4: Agent availability
        used_agents = {step.agent_type for step in plan.steps.values()}
        available_agents = set(self.available_agents + ["human_review"])
        agent_coverage = len(used_agents.intersection(available_agents)) / len(
            used_agents
        )
        confidence_factors.append(agent_coverage * 0.3)

        return sum(confidence_factors)

    def _generate_plan_reasoning(
        self,
        plan: ResearchPlan,
        params: PlanGenerationInput,
    ) -> str:
        """Generate human-readable reasoning for the plan"""

        return f"""
        Generated research plan for '{params.research_goal}' with {len(plan.steps)} steps:
        
        1. Starts with literature search to establish knowledge foundation
        2. Includes human review points for quality control
        3. Follows logical dependency chain: search → review → extract → analyze → synthesize
        4. Balances automation with human oversight
        5. Prioritizes steps based on research workflow best practices
        
        The plan is designed to be thorough while allowing for human guidance and adaptation.
        """

    async def adapt_plan(
        self,
        current_plan: ResearchPlan,
        step_results: Dict[str, Any],
        human_feedback: Optional[Dict[str, Any]] = None,
    ) -> ResearchPlan:
        """Adapt the current plan based on execution results and feedback"""

        adapted_plan = current_plan.model_copy(deep=True)
        adapted_plan.updated_at = datetime.now().isoformat()

        # Add human feedback to plan history
        if human_feedback:
            adapted_plan.human_feedback.append(
                {
                    "timestamp": adapted_plan.updated_at,
                    "feedback": human_feedback,
                }
            )

        # Adapt based on step results
        if step_results:
            await self._adapt_based_on_results(adapted_plan, step_results)

        # Adapt based on human feedback
        if human_feedback:
            await self._adapt_based_on_feedback(adapted_plan, human_feedback)

        return adapted_plan

    async def _adapt_based_on_results(
        self,
        plan: ResearchPlan,
        step_results: Dict[str, Any],
    ) -> None:
        """Adapt plan based on step execution results"""

        # Example adaptations:
        # - If literature search yields few results, add more search steps
        # - If extraction fails, add validation step
        # - If analysis shows gaps, add targeted literature search

        for step_id, result in step_results.items():
            if step_id in plan.steps:
                step = plan.steps[step_id]

                # Adapt based on step type and results
                if step.agent_type == "literature_search":
                    await self._adapt_literature_search_step(plan, step, result)
                elif step.agent_type == "extraction":
                    await self._adapt_extraction_step(plan, step, result)

    async def _adapt_literature_search_step(
        self,
        plan: ResearchPlan,
        step: PlanStep,
        result: Dict[str, Any],
    ) -> None:
        """Adapt plan based on literature search results"""

        # If too few results, add more targeted search
        result_count = result.get("result_count", 0)
        if result_count < 5:
            new_step_id = f"lit_search_targeted_{len(plan.steps) + 1:03d}"
            new_step = PlanStep(
                step_id=new_step_id,
                agent_type="literature_search",
                description="Targeted literature search with refined queries",
                inputs={"query": "refined_query", "max_results": 30},
                dependencies=[step.step_id],
                priority=step.priority - 1,
                human_review_required=True,
            )
            plan.steps[new_step_id] = new_step

    async def _adapt_extraction_step(
        self,
        plan: ResearchPlan,
        step: PlanStep,
        result: Dict[str, Any],
    ) -> None:
        """Adapt plan based on extraction results"""

        # If extraction confidence is low, add validation step
        confidence = result.get("average_confidence", 1.0)
        if confidence < 0.7:
            new_step_id = f"validation_{len(plan.steps) + 1:03d}"
            new_step = PlanStep(
                step_id=new_step_id,
                agent_type="validation",
                description="Validate extraction results due to low confidence",
                inputs={"extracted_data": "extraction_output"},
                dependencies=[step.step_id],
                priority=step.priority - 1,
                human_review_required=True,
            )
            plan.steps[new_step_id] = new_step

    async def _adapt_based_on_feedback(
        self,
        plan: ResearchPlan,
        feedback: Dict[str, Any],
    ) -> None:
        """Adapt plan based on human feedback"""

        feedback_type = feedback.get("type", "general")

        if feedback_type == "add_step":
            self._add_step_from_feedback(plan, feedback)
        elif feedback_type == "modify_step":
            self._modify_step_from_feedback(plan, feedback)
        elif feedback_type == "change_priority":
            self._change_priority_from_feedback(plan, feedback)

    def _add_step_from_feedback(
        self, plan: ResearchPlan, feedback: Dict[str, Any]
    ) -> None:
        """Add a step based on human feedback"""
        step_config = feedback.get("step_config", {})
        new_step_id = f"human_requested_{len(plan.steps) + 1:03d}"

        new_step = PlanStep(
            step_id=new_step_id,
            agent_type=step_config.get("agent_type", "analysis"),
            description=step_config.get("description", "Human-requested step"),
            inputs=step_config.get("inputs", {}),
            dependencies=step_config.get("dependencies", []),
            priority=step_config.get("priority", 5),
            human_review_required=step_config.get("human_review_required", False),
        )
        plan.steps[new_step_id] = new_step

    def _modify_step_from_feedback(
        self, plan: ResearchPlan, feedback: Dict[str, Any]
    ) -> None:
        """Modify a step based on human feedback"""
        step_id = feedback.get("step_id")
        modifications = feedback.get("modifications", {})

        if step_id in plan.steps:
            step = plan.steps[step_id]
            for field, value in modifications.items():
                if hasattr(step, field):
                    setattr(step, field, value)

    def _change_priority_from_feedback(
        self, plan: ResearchPlan, feedback: Dict[str, Any]
    ) -> None:
        """Change step priorities based on human feedback"""
        priority_changes = feedback.get("priority_changes", {})

        for step_id, new_priority in priority_changes.items():
            if step_id in plan.steps:
                plan.steps[step_id].priority = new_priority


class WorkflowOrchestrator(AsyncRunMixin):
    """
    Orchestrates the execution of multi-agent workflows based on plans.
    Handles step execution, human interactions, and state management.
    """

    def __init__(
        self,
        planner: PlannerAgent,
        node_registry: Dict[str, Any] = None,
        debug: bool = False,
    ):
        self.planner = planner
        self.node_registry = node_registry or {}
        self.debug = debug

    async def execute_workflow(
        self,
        state: GlobalWorkflowState,
    ) -> GlobalWorkflowState:
        """Execute the workflow according to the current plan"""

        if state.workflow_status == "waiting_human":
            # Don't continue execution while waiting for human input
            return state

        current_plan = state.planner_state.current_plan
        if not current_plan:
            if self.debug:
                logger.warning("No current plan found. Workflow cannot proceed.")
            state.workflow_status = "failed"
            return state

        # Get next steps ready for execution
        ready_steps = state.get_ready_steps()

        if not ready_steps:
            # Check if workflow is complete
            all_completed = all(
                step.status == AgentStatus.COMPLETED
                for step in current_plan.steps.values()
            )

            if all_completed:
                state.workflow_status = "completed"
                if self.debug:
                    logger.info("Workflow completed successfully!")
            else:
                # Check for blocked steps
                blocked_steps = [
                    step
                    for step in current_plan.steps.values()
                    if step.status == AgentStatus.BLOCKED
                ]
                if blocked_steps:
                    state.workflow_status = "failed"
                    if self.debug:
                        logger.error(
                            f"Workflow blocked by steps: {[s.step_id for s in blocked_steps]}"
                        )

            return state

        # Execute ready steps
        for step in ready_steps:
            if step.human_review_required:
                # Create human interaction
                interaction = HumanInteraction(
                    interaction_id=str(uuid.uuid4()),
                    interaction_type="review",
                    prompt=f"Please review step: {step.description}",
                    context={
                        "step_id": step.step_id,
                        "step_description": step.description,
                        "step_inputs": step.inputs,
                        "agent_type": step.agent_type,
                    },
                )
                state.add_human_interaction(interaction)
                step.status = AgentStatus.BLOCKED  # Block until human reviews
                break  # Only handle one human interaction at a time
            else:
                # Execute the step automatically
                try:
                    step.status = AgentStatus.IN_PROGRESS
                    current_plan.current_step_id = step.step_id

                    # Execute step using appropriate node/agent
                    result = await self._execute_step(step, state)

                    # Update step with results
                    step.outputs = result
                    step.status = AgentStatus.COMPLETED
                    current_plan.completed_steps.append(step.step_id)

                    # Add to execution history
                    state.execution_history.append(
                        {
                            "step_id": step.step_id,
                            "timestamp": datetime.now().isoformat(),
                            "status": "completed",
                            "outputs": result,
                        }
                    )

                    if self.debug:
                        logger.info(f"Completed step: {step.step_id}")

                except Exception as e:
                    step.status = AgentStatus.FAILED
                    step.error_message = str(e)

                    # Add to error log
                    state.error_log.append(
                        {
                            "step_id": step.step_id,
                            "timestamp": datetime.now().isoformat(),
                            "error": str(e),
                        }
                    )

                    if self.debug:
                        logger.error(f"Step {step.step_id} failed: {e}")

        return state

    async def _execute_step(
        self,
        step: PlanStep,
        state: GlobalWorkflowState,
    ) -> Dict[str, Any]:
        """Execute a single plan step"""

        agent_type = step.agent_type

        # Route to appropriate handler based on agent type
        if agent_type == "literature_search":
            return await self._execute_literature_search(step, state)
        elif agent_type == "extraction":
            return await self._execute_extraction(step, state)
        elif agent_type == "analysis":
            return await self._execute_analysis(step, state)
        elif agent_type == "synthesis":
            return await self._execute_synthesis(step, state)
        elif agent_type == "validation":
            return await self._execute_validation(step, state)
        else:
            # Try to find in node registry
            if agent_type in self.node_registry:
                node = self.node_registry[agent_type]
                return await node.arun(state)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

    async def _execute_literature_search(
        self,
        step: PlanStep,
        state: GlobalWorkflowState,
    ) -> Dict[str, Any]:
        """Execute literature search step"""

        query = step.inputs.get("query", "")
        max_results = step.inputs.get("max_results", 10)

        # This would integrate with your existing LitAgent
        # For now, return mock results
        return {
            "search_results": [],
            "result_count": 0,
            "queries_used": [query],
            "execution_time": 0,
        }

    async def _execute_extraction(
        self,
        step: PlanStep,
        state: GlobalWorkflowState,
    ) -> Dict[str, Any]:
        """Execute extraction step"""

        # Extract from literature search results
        sources = step.inputs.get("sources", [])
        schema = step.inputs.get("schema", "auto_detect")

        return {
            "extracted_data": [],
            "extraction_count": 0,
            "average_confidence": 0.0,
            "schema_used": schema,
        }

    async def _execute_analysis(
        self,
        step: PlanStep,
        state: GlobalWorkflowState,
    ) -> Dict[str, Any]:
        """Execute analysis step"""

        extracted_data = step.inputs.get("extracted_data", [])

        return {
            "analysis_results": {},
            "patterns_found": [],
            "insights": [],
            "confidence_score": 0.0,
        }

    async def _execute_synthesis(
        self,
        step: PlanStep,
        state: GlobalWorkflowState,
    ) -> Dict[str, Any]:
        """Execute synthesis step"""

        analysis_results = step.inputs.get("analysis_results", {})

        return {
            "synthesis_report": "",
            "key_findings": [],
            "recommendations": [],
            "confidence_score": 0.0,
        }

    async def _execute_validation(
        self,
        step: PlanStep,
        state: GlobalWorkflowState,
    ) -> Dict[str, Any]:
        """Execute validation step"""

        data_to_validate = step.inputs.get("extracted_data", [])

        return {
            "validation_results": {},
            "issues_found": [],
            "confidence_score": 0.0,
            "recommendations": [],
        }

    async def handle_human_response(
        self,
        state: GlobalWorkflowState,
        interaction_id: str,
        response: Dict[str, Any],
    ) -> GlobalWorkflowState:
        """Handle human response to an interaction"""

        if state.complete_human_interaction(interaction_id, response):
            # Process the human response
            interaction = next(
                (
                    hi
                    for hi in state.human_interactions
                    if hi.interaction_id == interaction_id
                ),
                None,
            )

            if interaction and interaction.context.get("step_id"):
                step_id = interaction.context["step_id"]
                plan = state.planner_state.current_plan

                if plan and step_id in plan.steps:
                    step = plan.steps[step_id]

                    # Process response based on interaction type
                    if response.get("action") == "approve":
                        step.status = AgentStatus.PENDING  # Ready for execution
                    elif response.get("action") == "modify":
                        # Apply modifications from human response
                        modifications = response.get("modifications", {})
                        for field, value in modifications.items():
                            if hasattr(step, field):
                                setattr(step, field, value)
                        step.status = AgentStatus.PENDING
                    elif response.get("action") == "reject":
                        step.status = AgentStatus.FAILED
                        step.error_message = response.get(
                            "reason", "Rejected by human reviewer"
                        )

            if self.debug:
                logger.info(
                    f"Processed human response for interaction {interaction_id}"
                )

        return state
