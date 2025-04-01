import asyncio
from abc import abstractmethod
from typing import Any


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class AsyncRunMixin:
    @abstractmethod
    async def arun(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses should implement this method")

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
