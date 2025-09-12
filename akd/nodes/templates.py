import uuid
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional

from jsonpath_ng import parse as jsonpath_parse
from loguru import logger

from akd._base import AbstractBase
from akd.agents._base import BaseAgent
from akd.common_types import CallableSpec
from akd.configs.guardrails_config import GuardrailsConfig
from akd.guardrails import apply_guardrails
from akd.tools.granite_guardian_tool import RiskDefinition
from akd.tools.utils import ToolRunner

from .states import GlobalState, NodeState
from .supervisor import BaseSupervisor


class AbstractNodeTemplate(AbstractBase[GlobalState, NodeState]):
    """
    An abstract base class for Node template.
    Implementation should do the following things:
    - Take in global state
    - Run through the input guardrails
    - Run the _execute method
    - Run through the output guardrails
    - Update the global state with the updated per-node state
    - Return the updated state
    """

    input_schema = GlobalState
    output_schema = NodeState

    def __init__(
        self,
        node_id: Optional[str] = None,
        input_guardrails: List[CallableSpec] | None = None,
        output_guardrails: List[CallableSpec] | None = None,
        tool_runner: Optional[ToolRunner] = None,
        mutation: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(debug=debug, **kwargs)
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.node_id = node_id or str(uuid.uuid4().hex)
        self.tool_runner = tool_runner or ToolRunner(debug=debug)
        self.mutation = mutation

    async def _arun(self, params: GlobalState, **kwargs) -> NodeState:
        """Run the node with the given state."""
        # 0) grab or create this node's local state slice
        global_state = params
        node_id = self.node_id
        node_state = global_state.node_states.get(node_id, NodeState())

        if not node_state.inputs:
            logger.warning(f"Node {node_id} has no inputs to process.")

        # If mutation is not enabled, we create a copy of the node state
        # to avoid modifying the original state in place.
        # This is useful for testing or when we want to keep the original state intact.
        if not self.mutation:
            node_state = node_state.model_copy(deep=True)

        if self.debug:
            logger.debug(f"[Node {node_id}] node_state={node_state}")

        # 1) run input guardrails against node_state.inputs
        node_state.input_guardrails = await self._apply_guardrails(
            self.input_guardrails,
            node_state.inputs.copy(),
        )

        # 2) execute core logic (either through supervisor or custom _execute method)
        node_state = await self._execute(node_state, global_state)

        # 3) run output guardrails against that output
        node_state.output_guardrails = await self._apply_guardrails(
            self.output_guardrails,
            node_state.outputs.copy(),
        )

        # 4) write back into the global state, in place
        if self.mutation:
            global_state.node_states[node_id] = node_state

        # 5) return the updated per-node state
        return node_state

    @abstractmethod
    async def _execute(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> NodeState:
        """Execute the core logic of the node.

        This method should be implemented by subclasses to define the specific
        execution logic for the node. It runs between input and output guardrails.

        Args:
            node_state: The current state of the node
            global_state: The global system state

        Returns:
            Updated node state after execution
        """
        raise NotImplementedError()

    async def _apply_guardrails(
        self,
        guardrails: List[CallableSpec],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Note:
            If guardrails are callables,
            they will be wrapped into BaseTool via tool_wrapper.

            If they are BaseTool instances, they will be used as is.

            If they are Tuple,
            the 2nd element is the mapping of input keys.
        """
        results: Dict[str, Any] = {}
        for guard in guardrails:
            if isinstance(guard, tuple):
                name = guard[0].__class__.__name__
            else:
                name = guard.__class__.__name__
            try:
                # build the guard's inputs from matching keys in the data
                tool_output = await self.tool_runner.arun(
                    spec=guard,
                    data=data,
                )
                results[name] = getattr(tool_output, "result", tool_output)
            except Exception as e:
                if self.debug:
                    logger.error(f"[{name}] guardrail {name!r} error: {e!r}")
                results[name] = None
        return results

    def to_langgraph_node(
        self,
        key: str | None = None,
    ) -> Callable[[GlobalState], Awaitable[GlobalState]]:
        """
        Convert to langgraph compatile node.
        Global state in -> global state out
        Assumption:
            - NodeTemplate should mutate the global state itself
            - Supervisor should handle how to access keys
        """

        key = key or self.node_id

        async def _node_fn(gs: GlobalState) -> Dict[str, NodeState]:
            ns = await self.arun(gs)
            # return {key: ns} -> return per-node partial state
            # return gs # return full global state -> not recommended
            # return partial state based on global key
            return {
                "node_states": {
                    self.node_id: ns,
                },
            }

        # _node_fn.__name__ = f"node_{key}"
        return _node_fn


class SupervisedNodeTemplate(AbstractNodeTemplate):
    """
    A node template that uses a supervisor for execution.
    This class implements the _execute method using supervisor-based logic.
    """

    def __init__(
        self,
        supervisor: BaseSupervisor,
        input_guardrails: List[CallableSpec] | None = None,
        output_guardrails: List[CallableSpec] | None = None,
        node_id: Optional[str] = None,
        tool_runner: Optional[ToolRunner] = None,
        mutation: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            node_id=node_id,
            tool_runner=tool_runner,
            mutation=mutation,
            debug=debug,
            **kwargs,
        )
        assert isinstance(
            supervisor,
            BaseSupervisor,
        ), "supervisor must be an instance of BaseSupervisor"
        self.supervisor = supervisor

    async def _execute(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> NodeState:
        """Execute using supervisor-based logic."""
        # Create a temporary supervisor state from node state
        temp_supervisor_state = NodeState(
            messages=node_state.messages.copy(),
            inputs=node_state.inputs.copy(),
            outputs=node_state.outputs.copy(),
            steps=node_state.steps.copy(),
        )

        # Run supervisor
        sup_out = await self.supervisor.arun(
            temp_supervisor_state,
            global_state=global_state,
        )

        # Copy back supervisor outputs & messages into node_state
        node_state.messages += sup_out.messages
        node_state.outputs.update(sup_out.outputs)

        # sanity check for tool calls
        # default: Node State does not have tool_calls because of serialization issues
        # run only if the state has it
        # TODO: fix serialization issues
        if hasattr(node_state, "tool_calls") and hasattr(sup_out, "tool_calls"):
            node_state.tool_calls.extend(sup_out.tool_calls)

        node_state.steps.update(sup_out.steps)

        return node_state


class SingleAgentNodeTemplate(AbstractNodeTemplate):
    """
    A node template that wraps a single agent and automatically binds the agent's IO schema.
    This class simplifies node creation for single-agent workflows by automatically extracting
    the input and output schemas from the provided agent.
    """

    def __init__(
        self,
        agent: BaseAgent,
        node_id: Optional[str] = None,
        input_guardrails: Optional[List[RiskDefinition]] = None,
        output_guardrails: Optional[List[RiskDefinition]] = None,
        guardrails_config: Optional[GuardrailsConfig] = None,
        io_map: Optional[Dict[str, str]] = None,
        mutation: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the SingleAgentNodeTemplate with an agent.

        Args:
            agent: The BaseAgent instance to wrap
            input_guardrails: RiskDefinition list for AI safety input validation
            output_guardrails: RiskDefinition list for AI safety output validation
            guardrails_config: Configuration for RiskDefinition-style guardrails
            io_map: Optional mapping of input fields to other node fields (e.g., {"query": "lit_search.query"})
            node_id: Unique identifier for this node
            tool_runner: Tool runner instance
            mutation: Whether to mutate global state in place
            debug: Enable debug logging
            **kwargs: Additional keyword arguments
        """
        if not isinstance(agent, BaseAgent):
            raise TypeError("agent must be an instance of BaseAgent")

        self.agent = apply_guardrails(
            component=agent,
            config=guardrails_config,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            input_fields=kwargs.get("input_fields", []),
            output_fields=kwargs.get("output_fields", []),
        )
        # Validate that agent has required schemas
        if not hasattr(self.agent, "input_schema") or self.agent.input_schema is None:
            raise ValueError(
                f"Agent {self.agent.__class__.__name__} must have an input_schema",
            )
        if not hasattr(self.agent, "output_schema") or self.agent.output_schema is None:
            raise ValueError(
                f"Agent {self.agent.__class__.__name__} must have an output_schema",
            )

        # Dynamically set the input and output schemas from the agent
        self._input_schema = self.agent.input_schema
        self._output_schema = self.agent.output_schema

        # Store io_map for cross-node input mapping with JSONPath support
        self.io_map = io_map or {}

        # Call parent constructor with no CallableSpec guardrails (agent handles its own guardrails)
        super().__init__(
            input_guardrails=[],  # no need to run callable spec guardrails
            output_guardrails=[],  # no need to run callable spec guardrails
            node_id=node_id,
            mutation=mutation,
            debug=debug,
            **kwargs,
        )

    def _build_jsonpath_context(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> Dict[str, Any]:
        """
        Build JSONPath context from global state for cross-node data access.

        Args:
            node_state: Current node's state
            global_state: Global state containing all node states

        Returns:
            Dictionary context for JSONPath expressions
        """
        context = {
            # Current node access
            "current": {
                "inputs": node_state.inputs,
                "outputs": node_state.outputs,
            },
            # All other nodes
            **{
                node_id: {
                    "inputs": ns.inputs,
                    "outputs": ns.outputs,
                }
                for node_id, ns in global_state.node_states.items()
            },
        }

        if self.debug:
            logger.debug(
                f"[SingleAgentNodeTemplate {self.node_id}] "
                f"JSONPath context keys: {list(context.keys())}",
            )

        return context

    def _apply_io_mapping(
        self,
        base_inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply io_map transformations to fill missing inputs using JSONPath.

        Args:
            base_inputs: Starting inputs (never overridden)
            context: JSONPath context for expression evaluation

        Returns:
            Dictionary with resolved inputs from io_map
        """
        resolved_inputs = base_inputs.copy()

        # Apply io_map using JSONPath to fill missing fields
        for target_field, jsonpath_expr in self.io_map.items():
            if target_field in resolved_inputs:
                if self.debug:
                    logger.debug(
                        f"[SingleAgentNodeTemplate {self.node_id}] "
                        f"Skipping '{target_field}' - already exists in inputs",
                    )
                continue  # Don't override existing inputs

            try:
                jsonpath = jsonpath_parse(jsonpath_expr)
                matches = jsonpath.find(context)

                if matches:
                    # Use first match
                    resolved_inputs[target_field] = matches[0].value
                    if self.debug:
                        logger.debug(
                            f"[SingleAgentNodeTemplate {self.node_id}] "
                            f"Mapped '{target_field}' from '{jsonpath_expr}': {matches[0].value}",
                        )
                else:
                    if self.debug:
                        logger.warning(
                            f"[SingleAgentNodeTemplate {self.node_id}] "
                            f"JSONPath '{jsonpath_expr}' returned no matches for field '{target_field}'",
                        )

            except Exception as e:
                logger.warning(
                    f"[SingleAgentNodeTemplate {self.node_id}] "
                    f"JSONPath '{jsonpath_expr}' failed for field '{target_field}': {e}",
                )

        return resolved_inputs

    def _validate_resolved_inputs(
        self,
        resolved_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate that resolved inputs can create agent input schema.

        Args:
            resolved_inputs: Dictionary of resolved inputs

        Returns:
            Same dictionary if validation passes

        Raises:
            ValueError: If validation fails
        """
        try:
            self.agent.input_schema(**resolved_inputs)
            if self.debug:
                logger.debug(
                    f"[SingleAgentNodeTemplate {self.node_id}] "
                    f"Successfully resolved inputs: {resolved_inputs}",
                )
            return resolved_inputs

        except Exception as validation_error:
            raise ValueError(
                f"Failed to resolve required inputs for agent {self.agent.__class__.__name__} "
                f"in node '{self.node_id}'. After applying io_map, validation failed: {validation_error}",
            )

    async def _resolve_inputs(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> Dict[str, Any]:
        """
        Resolve inputs for the agent using JSONPath for complex cross-node data access.

        Supports merging current node inputs with data from other nodes using JSONPath expressions.
        Never overrides existing inputs in the current node - only fills missing fields.

        Args:
            node_state: Current node's state
            global_state: Global state containing all node states

        Returns:
            Dictionary of resolved inputs for the agent

        Examples:
            io_map = {
                "query": "$.lit_search.outputs.query",              # Simple field access
                "context": "$.preprocessing.outputs.cleaned_text",  # Cross-node data
                "limit": "$.config.inputs.params.limit",            # Nested access
                "results": "$.search.outputs.items[*].title"        # Array extraction
            }
        """
        # Start with node's existing inputs (never override existing)
        resolved_inputs = node_state.inputs.copy()

        # If no io_map configured, just validate and return current inputs
        if not self.io_map:
            try:
                return self._validate_resolved_inputs(resolved_inputs)
            except ValueError as e:
                raise ValueError(
                    f"Node '{self.node_id}' missing required inputs for agent "
                    f"{self.agent.__class__.__name__}. Consider using io_map parameter.\nError: {e}",
                ) from e

        # Build JSONPath context and apply io_map transformations
        context = self._build_jsonpath_context(node_state, global_state)
        resolved_inputs = self._apply_io_mapping(resolved_inputs, context)

        # Validate and return final inputs
        return self._validate_resolved_inputs(resolved_inputs)

    async def _execute(
        self,
        node_state: NodeState,
        global_state: GlobalState,
    ) -> NodeState:
        """Execute the agent using enhanced input resolution with cross-node data access."""
        try:
            # Resolve inputs using JSONPath-based cross-node mapping
            resolved_inputs = await self._resolve_inputs(node_state, global_state)

            # Create agent input schema instance
            agent_input = self.agent.input_schema(**resolved_inputs)

            if self.debug:
                logger.debug(
                    f"[SingleAgentNodeTemplate {self.node_id}] "
                    f"Running agent {self.agent.__class__.__name__} with input: {agent_input}",
                )

            # Run the agent
            agent_output = await self.agent._arun(agent_input)

            # Store the agent output in node state outputs
            # Convert the agent output to dict format for storage
            node_state.outputs.update(agent_output.model_dump())

            if self.debug:
                logger.debug(
                    f"[SingleAgentNodeTemplate {self.node_id}] "
                    f"Agent output: {agent_output}",
                )

        except Exception as e:
            logger.error(
                f"[SingleAgentNodeTemplate {self.node_id}] "
                f"Error executing agent {self.agent.__class__.__name__}: {e}",
            )
            raise

        return node_state
