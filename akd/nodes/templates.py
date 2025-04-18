import uuid
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional

from loguru import logger

from ..structures import GuardrailType
from ..tools._base import BaseTool
from ..tools.utils import tool_wrapper
from ..utils import AsyncRunMixin, LangchainToolMixin
from .states import GlobalState, NodeTemplateState
from .supervisor import BaseSupervisor


class AbstractNodeTemplate(ABC, AsyncRunMixin, LangchainToolMixin):
    """
    An abstract base class for Node template.
    Implementation should do the following things:
    - Take in global state
    - Run through the input guardrails
    - Run the supervisor
    - Run through the output guardrails
    - Update the global state with the updated per-node state
    - Return the updated state
    """

    def __init__(
        self,
        supervisor: BaseSupervisor,
        input_guardrails: List[GuardrailType],
        output_guardrails: List[GuardrailType],
        node_id: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        assert isinstance(
            supervisor,
            BaseSupervisor,
        ), "supervisor must be an instance of BaseSupervisor"
        self.supervisor = supervisor
        self.input_guardrails = input_guardrails
        self.output_guardrails = output_guardrails
        self.debug = bool(debug)
        self.node_id = node_id or str(uuid.uuid4().hex)

    @abstractmethod
    async def arun(self, state: GlobalState) -> NodeTemplateState:
        """Run the node with the given state."""
        raise NotImplementedError()

    async def _apply_guardrails(
        self,
        guardrails: List[GuardrailType],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        for guard in guardrails:
            name = guard.__class__.__name__
            if not isinstance(guard, BaseTool):
                guard = tool_wrapper(guard)
            try:
                # build the guard's inputs from matching keys in the data
                schema = guard.input_schema
                kwargs = {k: data.get(k) for k in schema.model_fields}
                tool_input = schema(**kwargs)
                tool_output = await guard.arun(tool_input)
                results[name] = getattr(tool_output, "result", tool_output)
            except Exception as e:
                if self.debug:
                    logger.error(f"[{name}] guardrail {name!r} error: {e!r}")
                results[name] = None
        return results

    def to_langgraph_node(self) -> Callable[[GlobalState], Awaitable[GlobalState]]:
        """
        Convert to langgraph compatile node.
        Global state in -> global state out
        Assumption:
            - NodeTemplate should mutate the global state itself
            - Supervisor should handle how to access keys
        """

        async def _node_fn(gs: GlobalState) -> GlobalState:
            # 1) make sure this node has its local state slice
            if self.node_id not in gs.node_states:
                gs.node_states[self.node_id] = NodeTemplateState()
            # 2) run the node’s logic (guardrails → supervisor → guardrails → write‐back)
            await self.arun(gs)
            # 3) return the (mutated) GlobalState
            return gs

        return _node_fn


class DefaultNodeTemplate(AbstractNodeTemplate):
    async def arun(self, global_state: GlobalState) -> NodeTemplateState:
        # 0) grab or create this node’s local state slice
        node_id = self.node_id
        node_state = global_state.node_states.get(node_id, NodeTemplateState())

        if self.debug:
            logger.debug(f"[Node {node_id}] node_state={node_state}")

        # 1) run input guardrails against node_state.inputs
        node_state.input_guardrails = await self._apply_guardrails(
            self.input_guardrails,
            node_state.inputs.copy(),
        )

        # 2) merge validated inputs into the supervisor_state
        node_state.supervisor_state.inputs.update(node_state.inputs)

        # 3) call the supervisor with its slice of state
        sup_out = await self.supervisor.arun(
            node_state.supervisor_state,
            global_state=global_state,
        )

        #    copy back supervisor outputs & messages into node_state
        node_state.supervisor_state = sup_out
        node_state.output = sup_out.output
        node_state.messages += sup_out.messages

        # 4) run output guardrails against that output
        node_state.output_guardrails = await self._apply_guardrails(
            self.output_guardrails,
            node_state.output.copy(),
        )

        # 5) write back into the global state, in place
        global_state.node_states[node_id] = node_state

        # 6) return the updated per-node state
        return node_state
