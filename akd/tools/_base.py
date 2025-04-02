from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Optional

from atomic_agents.agents.base_agent import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool as AtomicBaseTool
from atomic_agents.lib.base.base_tool import BaseToolConfig

from ..utils import AsyncRunMixin, get_event_loop


class BaseTool(AtomicBaseTool, AsyncRunMixin):
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

    @abstractmethod
    async def arun(self, params: BaseIOSchema, **kwargs) -> BaseIOSchema:
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

    async def run_async(self, params: BaseIOSchema, **kwargs) -> BaseIOSchema:
        """
        Executes the tool with the provided parameters in async.

        Args:
            params (BaseIOSchema): Input parameters adhering to the input schema.

        Returns:
            BaseIOSchema: Output resulting from executing the tool, adhering to the output schema.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        return await self.arun(params, **kwargs)

    def run(self, params: BaseIOSchema, **kwargs) -> BaseIOSchema:
        """
        Executes the tool with the provided parameters by running the async implementation.

        This method creates an event loop to run the async implementation if one doesn't exist,
        or uses the current event loop if it's already running.

        Args:
            params (BaseIOSchema): Input parameters adhering to the input schema.

        Returns:
            BaseIOSchema: Output resulting from executing the tool, adhering to the output schema.
        """
        try:
            # Check if there's a running event loop
            loop = get_event_loop()
            # If we're already in an event loop, we need to use create_task and wait for it
            if loop and loop.is_running():
                # This creates a new task in the current event loop
                future = asyncio.ensure_future(self.arun(params, **kwargs))
                return loop.run_until_complete(future)
            else:
                return asyncio.run(self.arun(params, **kwargs))
        except RuntimeError:
            # No running event loop, create a new one
            return asyncio.run(self.arun(params, **kwargs))
