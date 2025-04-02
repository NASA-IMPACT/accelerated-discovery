import asyncio
from abc import abstractmethod
from typing import Any, Dict

try:
    from langchain_core.tools.structured import StructuredTool

    LANGCHAIN_CORE_INSTALLED = True
except:
    LANGCHAIN_CORE_INSTALLED = False


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


class LangchainToolMixin:
    """
    Mixin for converting tools/agents to Langchain structured tools.

    Preconditions:
        Each tool/agent should have:
        - input schema (input_schema)
        - output schema (output_schema)
        - async run method (arun)
    """

    def to_langchain_structured_tool(self) -> Any:
        if not LANGCHAIN_CORE_INSTALLED:
            raise ImportError("langchain-core is required to use this method")

        async def _wrapped_arun(**input_data: Dict[str, Any]) -> Dict[str, Any]:
            """
            A wrapper around the tool's run method to accept JSON-like input and return JSON-like output.
            """
            # Validate and parse input using Pydantic
            validated_input = self.input_schema(**input_data)

            # Execute the tool
            output_obj = await self.arun(validated_input)

            # Convert output Pydantic object to dictionary
            return output_obj.model_dump()

        return StructuredTool.from_function(
            func=_wrapped_arun,
            name=self.__class__.__name__,
            description=f"A tool that executes {self.__class__.__name__}",
            args_schema=self.input_schema,
        )
