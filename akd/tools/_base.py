from __future__ import annotations

from typing import Optional

from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool as AtomicBaseTool
from atomic_agents.lib.base.base_tool import BaseToolConfig


class BaseTool(AtomicBaseTool):
    def __init__(
        self,
        config: Optional[BaseToolConfig] = None,
        debug: bool = False,
    ) -> None:
        """
        Initializes the BaseTool with an optional configuration override.

        Args:
            config (BaseToolConfig, optional):
                Configuration for the tool, including optional title
                and description overrides.
            debug (bool, optional):
                Boolean flag for debug mode.
        """
        config = config or BaseToolConfig()
        self.config = config
        self.debug = bool(debug) or getattr(config, "debug", False)
        super().__init__(config)

    @classmethod
    def from_params(cls, **params) -> BaseTool:
        raise NotImplementedError()

    async def arun(self, params: BaseIOSchema) -> BaseIOSchema:
        """
        Executes the tool with the provided parameters in async.

        Args:
            params (BaseIOSchema): Input parameters adhering to the input schema.

        Returns:
            BaseIOSchema: Output resulting from executing the tool, adhering to the output schema.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def run(self, params: BaseIOSchema) -> BaseIOSchema:
        """
        Executes the tool with the provided parameters.

        Args:
            params (BaseIOSchema): Input parameters adhering to the input schema.

        Returns:
            BaseIOSchema: Output resulting from executing the tool, adhering to the output schema.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method")
