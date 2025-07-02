"""
Enhanced workflow template for planner-driven multi-agent research workflows.
Integrates with LangGraph for stateful execution with human interaction.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger

from akd.agents.litsearch import LitAgent, LitAgentInputSchema
from akd.utils import AsyncRunMixin

from .planner import PlanGenerationInput, PlannerAgent, WorkflowOrchestrator
from .states import AgentStatus, GlobalWorkflowState, HumanInteraction, PlannerState


class ResearchWorkflowTemplate(AsyncRunMixin):
    """
    Template for creating planner-driven research workflows with LangGraph.

    Key Features:
    - Literature search as entry point
    - Human-driven planning and interaction
    - Dynamic agent coordination
    - Stateful execution with checkpoints
    - Conditional routing based on plan status
    """

    def __init__(
        self,
        planner_agent: PlannerAgent,
        lit_agent: LitAgent,
        node_registry: Optional[Dict[str, Any]] = None,
        checkpointer: Optional[Any] = None,
        debug: bool = False,
    ):
        self.planner_agent = planner_agent
        self.lit_agent = lit_agent
        self.orchestrator = WorkflowOrchestrator(
            planner=planner_agent,
            node_registry=node_registry or {},
            debug=debug,
        )
        self.checkpointer = checkpointer
        self.debug = debug
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the workflow"""

        # Create the graph with our enhanced state
        graph = StateGraph(GlobalWorkflowState)

        # Add nodes
        graph.add_node("initialize_workflow", self._initialize_workflow)
        graph.add_node("generate_plan", self._generate_plan)
        graph.add_node("human_plan_review", self._human_plan_review)
        graph.add_node("execute_literature_search", self._execute_literature_search)
        graph.add_node("execute_step", self._execute_step)
        graph.add_node("human_interaction", self._handle_human_interaction)
        graph.add_node("adapt_plan", self._adapt_plan)
        graph.add_node("finalize_workflow", self._finalize_workflow)

        # Add edges
        graph.add_edge(START, "initialize_workflow")
        graph.add_edge("initialize_workflow", "generate_plan")
        graph.add_edge("generate_plan", "human_plan_review")

        # Conditional edges for workflow routing
        graph.add_conditional_edges(
            "human_plan_review",
            self._route_after_plan_review,
            {
                "execute": "execute_literature_search",
                "modify": "generate_plan",
                "abort": END,
            },
        )

        graph.add_conditional_edges(
            "execute_literature_search",
            self._route_after_literature_search,
            {
                "continue": "execute_step",
                "human_review": "human_interaction",
                "complete": "finalize_workflow",
            },
        )

        graph.add_conditional_edges(
            "execute_step",
            self._route_after_step_execution,
            {
                "continue": "execute_step",
                "human_review": "human_interaction",
                "adapt_plan": "adapt_plan",
                "complete": "finalize_workflow",
                "error": END,
            },
        )

        graph.add_conditional_edges(
            "human_interaction",
            self._route_after_human_interaction,
            {
                "continue": "execute_step",
                "adapt_plan": "adapt_plan",
                "complete": "finalize_workflow",
            },
        )

        graph.add_edge("adapt_plan", "execute_step")
        graph.add_edge("finalize_workflow", END)

        return graph

    def compile_workflow(self) -> Any:
        """Compile the workflow graph for execution"""
        compile_kwargs = {}
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer

        return self.graph.compile(**compile_kwargs)

    # Node implementations

    async def _initialize_workflow(
        self,
        state: GlobalWorkflowState,
        config: Optional[RunnableConfig] = None,
    ) -> GlobalWorkflowState:
        """Initialize the workflow with user input"""

        if self.debug:
            logger.info("Initializing research workflow")

        # Extract research goal from messages
        research_goal = ""
        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, "content"):
                research_goal = last_message.content
            elif isinstance(last_message, dict):
                research_goal = last_message.get("content", "")

        # Initialize planner state
        state.planner_state = PlannerState(
            research_goal=research_goal,
            available_agents=self.planner_agent.available_agents,
            agent_capabilities=self.planner_agent.agent_capabilities,
        )

        # Set workflow status
        state.workflow_status = "planning"
        state.workflow_id = str(uuid.uuid4())

        if self.debug:
            logger.info(
                f"Initialized workflow {state.workflow_id} for goal: {research_goal}"
            )

        return state

    async def _generate_plan(
        self,
        state: GlobalWorkflowState,
        config: Optional[RunnableConfig] = None,
    ) -> GlobalWorkflowState:
        """Generate the research plan"""

        if self.debug:
            logger.info("Generating research plan")

        # Prepare plan generation input
        plan_input = PlanGenerationInput(
            research_goal=state.planner_state.research_goal,
            research_domain=state.planner_state.research_domain,
            available_agents=state.planner_state.available_agents,
        )

        # Generate plan
        plan_output = await self.planner_agent.arun(plan_input)

        # Update state
        state.planner_state.current_plan = plan_output.plan
        state.planner_state.plan_history.append(plan_output.plan)

        # Add AI message with plan summary
        plan_message = AIMessage(
            content=f"Generated research plan: {plan_output.plan.title}\n"
            f"Confidence: {plan_output.confidence_score:.2f}\n"
            f"Steps: {len(plan_output.plan.steps)}\n\n"
            f"Reasoning: {plan_output.reasoning}"
        )
        state.messages.append(plan_message)

        if self.debug:
            logger.info(f"Generated plan with {len(plan_output.plan.steps)} steps")

        return state

    async def _human_plan_review(
        self,
        state: GlobalWorkflowState,
        config: Optional[RunnableConfig] = None,
    ) -> GlobalWorkflowState:
        """Handle human review of the generated plan"""

        if self.debug:
            logger.info("Requesting human plan review")

        current_plan = state.planner_state.current_plan
        if not current_plan:
            state.workflow_status = "failed"
            return state

        # Create human interaction for plan review
        interaction = HumanInteraction(
            interaction_id=str(uuid.uuid4()),
            interaction_type="review",
            prompt="Please review the generated research plan. You can approve, request modifications, or abort.",
            context={
                "plan_id": current_plan.plan_id,
                "plan_title": current_plan.title,
                "plan_description": current_plan.description,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "agent_type": step.agent_type,
                        "description": step.description,
                        "priority": step.priority,
                        "human_review_required": step.human_review_required,
                    }
                    for step in current_plan.steps.values()
                ],
                "expected_actions": ["approve", "modify", "abort"],
            },
        )

        state.add_human_interaction(interaction)

        return state

    async def _execute_literature_search(
        self,
        state: GlobalWorkflowState,
        config: Optional[RunnableConfig] = None,
    ) -> GlobalWorkflowState:
        """Execute the literature search step"""

        if self.debug:
            logger.info("Executing literature search")

        # Find literature search step
        current_plan = state.planner_state.current_plan
        lit_search_step = None

        for step in current_plan.steps.values():
            if (
                step.agent_type == "literature_search"
                and step.status == AgentStatus.PENDING
            ):
                lit_search_step = step
                break

        if not lit_search_step:
            if self.debug:
                logger.warning("No pending literature search step found")
            return state

        # Update step status
        lit_search_step.status = AgentStatus.IN_PROGRESS
        current_plan.current_step_id = lit_search_step.step_id

        try:
            # Execute literature search
            query = lit_search_step.inputs.get(
                "query", state.planner_state.research_goal
            )
            max_results = lit_search_step.inputs.get("max_results", 10)

            lit_result = await self.lit_agent.arun(
                LitAgentInputSchema(
                    query=query,
                    max_search_results=max_results,
                )
            )

            # Update state with results
            state.literature_search_state.search_results.extend(
                [result.model_dump() for result in lit_result.results]
            )
            state.literature_search_state.search_iteration += 1

            # Update step
            lit_search_step.outputs = {
                "results": [result.model_dump() for result in lit_result.results],
                "result_count": len(lit_result.results),
                "query_used": query,
            }
            lit_search_step.status = AgentStatus.COMPLETED
            current_plan.completed_steps.append(lit_search_step.step_id)

            # Add AI message with results summary
            result_message = AIMessage(
                content=f"Literature search completed. Found {len(lit_result.results)} relevant sources."
            )
            state.messages.append(result_message)

            if self.debug:
                logger.info(
                    f"Literature search completed with {len(lit_result.results)} results"
                )

        except Exception as e:
            lit_search_step.status = AgentStatus.FAILED
            lit_search_step.error_message = str(e)

            state.error_log.append(
                {
                    "step_id": lit_search_step.step_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
            )

            if self.debug:
                logger.error(f"Literature search failed: {e}")

        return state

    async def _execute_step(
        self,
        state: GlobalWorkflowState,
        config: Optional[RunnableConfig] = None,
    ) -> GlobalWorkflowState:
        """Execute the next ready step in the plan"""

        if self.debug:
            logger.info("Executing next workflow step")

        # Use orchestrator to execute workflow
        updated_state = await self.orchestrator.execute_workflow(state)

        return updated_state

    async def _handle_human_interaction(
        self,
        state: GlobalWorkflowState,
        config: Optional[RunnableConfig] = None,
    ) -> GlobalWorkflowState:
        """Handle pending human interactions"""

        if self.debug:
            logger.info("Handling human interaction")

        # This node waits for human input through the checkpointer
        # The actual response handling is done via the orchestrator's handle_human_response method

        return state

    async def _adapt_plan(
        self,
        state: GlobalWorkflowState,
        config: Optional[RunnableConfig] = None,
    ) -> GlobalWorkflowState:
        """Adapt the plan based on results and feedback"""

        if self.debug:
            logger.info("Adapting research plan")

        current_plan = state.planner_state.current_plan
        if not current_plan:
            return state

        # Gather step results for adaptation
        step_results = {}
        for step in current_plan.steps.values():
            if step.status == AgentStatus.COMPLETED:
                step_results[step.step_id] = step.outputs

        # Gather human feedback
        human_feedback = None
        if state.human_interactions:
            latest_interaction = state.human_interactions[-1]
            if latest_interaction.status == "completed" and latest_interaction.response:
                human_feedback = latest_interaction.response

        # Adapt plan
        adapted_plan = await self.planner_agent.adapt_plan(
            current_plan,
            step_results,
            human_feedback,
        )

        # Update state
        state.planner_state.current_plan = adapted_plan
        state.planner_state.plan_history.append(adapted_plan)

        # Add AI message about adaptation
        adaptation_message = AIMessage(
            content=f"Plan adapted based on results and feedback. "
            f"Now has {len(adapted_plan.steps)} steps."
        )
        state.messages.append(adaptation_message)

        if self.debug:
            logger.info("Plan adaptation completed")

        return state

    async def _finalize_workflow(
        self,
        state: GlobalWorkflowState,
        config: Optional[RunnableConfig] = None,
    ) -> GlobalWorkflowState:
        """Finalize the workflow and prepare final results"""

        if self.debug:
            logger.info("Finalizing research workflow")

        # Gather all results
        all_results = {}
        current_plan = state.planner_state.current_plan

        if current_plan:
            for step in current_plan.steps.values():
                if step.status == AgentStatus.COMPLETED:
                    all_results[step.step_id] = {
                        "agent_type": step.agent_type,
                        "description": step.description,
                        "outputs": step.outputs,
                    }

        # Update research artifacts
        state.research_artifacts.update(
            {
                "final_results": all_results,
                "execution_summary": {
                    "total_steps": len(current_plan.steps) if current_plan else 0,
                    "completed_steps": len(current_plan.completed_steps)
                    if current_plan
                    else 0,
                    "failed_steps": len(
                        [
                            s
                            for s in (
                                current_plan.steps.values() if current_plan else []
                            )
                            if s.status == AgentStatus.FAILED
                        ]
                    ),
                    "human_interactions": len(state.human_interactions),
                    "execution_time": datetime.now().isoformat(),
                },
            }
        )

        # Set final status
        state.workflow_status = "completed"

        # Add final AI message
        final_message = AIMessage(
            content=f"Research workflow completed successfully. "
            f"Processed {len(all_results)} steps with "
            f"{len(state.human_interactions)} human interactions."
        )
        state.messages.append(final_message)

        if self.debug:
            logger.info("Workflow finalization completed")

        return state

    # Routing functions

    def _route_after_plan_review(self, state: GlobalWorkflowState) -> str:
        """Route after human plan review"""

        if not state.pending_human_interaction:
            # No pending interaction, check latest interaction
            if state.human_interactions:
                latest = state.human_interactions[-1]
                if latest.status == "completed" and latest.response:
                    action = latest.response.get("action", "execute")
                    if action == "approve":
                        return "execute"
                    elif action == "modify":
                        return "modify"
                    elif action == "abort":
                        return "abort"
            return "execute"  # Default to execute

        # Still waiting for human input
        return "execute"  # This will be handled by the workflow status

    def _route_after_literature_search(self, state: GlobalWorkflowState) -> str:
        """Route after literature search execution"""

        current_plan = state.planner_state.current_plan
        if not current_plan:
            return "complete"

        # Check if we need human review
        ready_steps = state.get_ready_steps()
        if ready_steps:
            next_step = ready_steps[0]
            if next_step.human_review_required:
                return "human_review"
            else:
                return "continue"

        return "complete"

    def _route_after_step_execution(self, state: GlobalWorkflowState) -> str:
        """Route after step execution"""

        if state.workflow_status == "waiting_human":
            return "human_review"
        elif state.workflow_status == "completed":
            return "complete"
        elif state.workflow_status == "failed":
            return "error"

        # Check if we need to adapt the plan
        current_plan = state.planner_state.current_plan
        if current_plan:
            # Check for failed steps that might require plan adaptation
            failed_steps = [
                step
                for step in current_plan.steps.values()
                if step.status == AgentStatus.FAILED
            ]
            if failed_steps:
                return "adapt_plan"

        # Check if there are more steps
        ready_steps = state.get_ready_steps()
        if ready_steps:
            return "continue"

        return "complete"

    def _route_after_human_interaction(self, state: GlobalWorkflowState) -> str:
        """Route after human interaction"""

        if state.pending_human_interaction:
            # Still waiting for human input
            return "continue"

        # Check latest interaction response
        if state.human_interactions:
            latest = state.human_interactions[-1]
            if latest.status == "completed" and latest.response:
                response_type = latest.response.get("type", "continue")
                if response_type == "adapt_plan":
                    return "adapt_plan"
                elif response_type == "complete":
                    return "complete"

        return "continue"


# Example usage and factory functions


def create_research_workflow(
    planner_agent: PlannerAgent,
    lit_agent: LitAgent,
    checkpointer: Optional[Any] = None,
    debug: bool = False,
) -> ResearchWorkflowTemplate:
    """Factory function to create a research workflow"""

    return ResearchWorkflowTemplate(
        planner_agent=planner_agent,
        lit_agent=lit_agent,
        checkpointer=checkpointer,
        debug=debug,
    )


def create_default_research_workflow(
    llm_client: Any,
    lit_agent: LitAgent,
    checkpointer: Optional[Any] = None,
    debug: bool = False,
) -> ResearchWorkflowTemplate:
    """Create a research workflow with default planner configuration"""

    planner = PlannerAgent(
        llm_client=llm_client,
        debug=debug,
    )

    return create_research_workflow(
        planner_agent=planner,
        lit_agent=lit_agent,
        checkpointer=checkpointer,
        debug=debug,
    )
