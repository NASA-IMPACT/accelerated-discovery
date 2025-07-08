import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from loguru import logger
from pydantic import BaseModel, Field

from akd.common_types import CallableSpec
from akd.tools.utils import ToolRunner
from akd.utils import AsyncRunMixin, LangchainToolMixin

from .states import GlobalState, NodeTemplateState
from .supervisor import BaseSupervisor


class NodeTemplateConfig(BaseModel):
    """Configuration for Node Template."""

    name: Optional[str] = Field(
        default=None,
        description="Human-readable name for this node",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of what this node does",
    )
    version: str = Field(default="1.0.0", description="Version of this node")
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing this node",
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Timeout in seconds for node execution",
    )
    retry_policy: Dict[str, Any] = Field(
        default_factory=lambda: {"max_retries": 3, "retry_delay": 1.0},
        description="Retry policy for failed executions",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of node IDs this node depends on",
    )
    parallel_execution: bool = Field(
        default=False,
        description="Whether this node can be executed in parallel with others",
    )
    interrupt_before: bool = Field(
        default=False,
        description="Whether to interrupt before executing this node",
    )
    interrupt_after: bool = Field(
        default=False,
        description="Whether to interrupt after executing this node",
    )
    checkpoint_enabled: bool = Field(
        default=True,
        description="Whether to enable checkpointing for this node",
    )
    required_inputs: Set[str] = Field(
        default_factory=set,
        description="Required input keys for this node",
    )
    expected_outputs: Set[str] = Field(
        default_factory=set,
        description="Expected output keys from this node",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this node",
    )


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
        input_guardrails: List[CallableSpec],
        output_guardrails: List[CallableSpec],
        node_id: Optional[str] = None,
        tool_runner: Optional[ToolRunner] = None,
        debug: bool = False,
        config: Optional[NodeTemplateConfig] = None,
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
        self.tool_runner = tool_runner or ToolRunner(debug=debug)
        self.config = config or NodeTemplateConfig()
        self.created_at = datetime.now()
        self.execution_count = 0
        self.last_execution_time: Optional[datetime] = None
        self.last_execution_duration: Optional[float] = None

        # Set node name from config if not provided
        if not self.config.name:
            self.config.name = f"{self.__class__.__name__}_{self.node_id[:8]}"

    @abstractmethod
    async def arun(self, state: GlobalState) -> NodeTemplateState:
        """Run the node with the given state."""
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

    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate that all required inputs are present."""
        missing_inputs = self.config.required_inputs - set(inputs.keys())
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")

    def validate_outputs(self, outputs: Dict[str, Any]) -> None:
        """Validate that all expected outputs are present."""
        if self.config.expected_outputs:
            missing_outputs = self.config.expected_outputs - set(outputs.keys())
            if missing_outputs:
                logger.warning(f"Missing expected outputs: {missing_outputs}")

    def get_node_info(self) -> Dict[str, Any]:
        """Get comprehensive node information."""
        return {
            "node_id": self.node_id,
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "tags": self.config.tags,
            "dependencies": self.config.dependencies,
            "created_at": self.created_at,
            "execution_count": self.execution_count,
            "last_execution_time": self.last_execution_time,
            "last_execution_duration": self.last_execution_duration,
            "metadata": self.config.metadata,
        }

    def can_execute_in_parallel(self) -> bool:
        """Check if this node can be executed in parallel."""
        return self.config.parallel_execution

    def should_interrupt_before(self) -> bool:
        """Check if execution should be interrupted before this node."""
        return self.config.interrupt_before

    def should_interrupt_after(self) -> bool:
        """Check if execution should be interrupted after this node."""
        return self.config.interrupt_after

    def to_langgraph_node(
        self,
        key: str | None = None,
        enable_interrupts: bool = True,
        enable_checkpointing: bool = True,
    ) -> Callable[[GlobalState], Awaitable[GlobalState]]:
        """
        Convert to langgraph compatible node with enhanced features.

        Args:
            key: Optional key for the node in the graph
            enable_interrupts: Whether to enable interrupt handling
            enable_checkpointing: Whether to enable checkpointing

        Returns:
            Async function that can be used as a LangGraph node
        """

        key = key or self.node_id

        async def _node_fn(gs: GlobalState) -> Dict[str, NodeTemplateState]:
            # 1) make sure this node has its local state slice
            if self.node_id not in gs.node_states:
                gs.node_states[self.node_id] = NodeTemplateState()
            # 2) run the node’s logic (guardrails → supervisor → guardrails → write‐back)
            ns = await self.arun(gs)
            # 3) return the (mutated) NodeTemplateState as mapping with its id
            return {key: ns}

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
