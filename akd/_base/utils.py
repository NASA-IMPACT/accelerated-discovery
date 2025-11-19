import asyncio
from abc import abstractmethod
from typing import Any

from pydantic import BaseModel, Field


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class AsyncRunMixin:
    """
    Mixin for adding interface to run
    async methods in a sync context.
    """

    @abstractmethod
    async def arun(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses should implement this method")

    # async def ainvoke(self, *args, **kwargs) -> Any:
    #     return await self.arun(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        """
        Runs the async method in a sync context.
        """
        if not hasattr(self, "arun"):
            raise AttributeError("Method 'arun' not implemented in the class")
        try:
            # Check if there's a running event loop
            loop = get_event_loop()
            # If we're already in an event loop, we need to use create_task and wait for it
            if loop and loop.is_running():
                # This creates a new task in the current event loop
                future = asyncio.ensure_future(self.arun(*args, **kwargs))
                return loop.run_until_complete(future)
            else:
                return asyncio.run(self.arun(*args, **kwargs))
        except RuntimeError:
            # No running event loop, create a new one
            return asyncio.run(self.arun(*args, **kwargs))


### Exposed Param Decorator and Metadata ###


class ExposedParam(BaseModel):
    """Metadata for exposed paramters."""

    description: str = Field(default="", description="Human readable description")
    extra: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Additional metadata for the parameter",
    )


def exposed_param(description: str, **kwargs):
    """
    Decorator to mark a property as exposed to external system

    Args:
        description: Human readable description of the parameter
        **kwargs: Additional metadata stored in 'extra' field
    """

    def decorator(func):
        func._exposed_meta = ExposedParam(
            description=description,
            extra=kwargs or {},
        )
        return func

    return decorator
