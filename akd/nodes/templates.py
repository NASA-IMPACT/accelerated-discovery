import uuid
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional

from loguru import logger

from akd._base import AbstractBase
from akd.common_types import CallableSpec
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
            tool_calls=node_state.tool_calls.copy(),
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
        node_state.tool_calls.extend(sup_out.tool_calls)
        node_state.steps.update(sup_out.steps)

        return node_state
