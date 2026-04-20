from __future__ import annotations

import asyncio
import inspect
import types
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Generic, Type, TypeVar, Union, get_args, get_origin

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, computed_field, create_model

from .config_binding import ConfigBindingMixin
from .errors import HumanInputRequired
from .streaming import (
    CompletedEvent,
    CompletedEventData,
    FailedEvent,
    FailedEventData,
    RunningEvent,
    StartingEvent,
    StreamEvent,
    StreamEventType,
)
from .structures import RunContext
from .validation import validate_input, validate_output


class BaseConfig(BaseModel):
    """
    Base configuration class for agents and tools.
    This class can be extended to define specific configurations for agents or tools.
    """

    model_config = {
        "extra": "forbid",  # Disallow extra fields
    }

    name: str | None = Field(
        default=None,
        description="Name of the tool/agent. Defaults to snake_case of class name.",
    )
    description: str | None = None
    io_hints: bool = Field(
        default=True,
        description="Whether to include input/output field hints in the agent/tool description. "
        "This replaces the deprecated 'input_hints' parameter in BaseAgentConfig.",
    )
    debug: bool = Field(
        default=False,
        description="Whether to enable debug mode",
    )


class IOSchema(BaseModel):
    "Base schema for any input or output schema for agents and tools"

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._validate_description()

    @classmethod
    def _validate_description(cls):
        description = (cls.__doc__ or "").strip()

        if not description:
            raise ValueError(
                f"{cls.__name__} must have a non-empty docstring to serve as its description",
            )

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        if "description" not in schema and cls.__doc__:
            schema["description"] = inspect.cleandoc(cls.__doc__)
        if "title" not in schema:
            schema["title"] = cls.__name__
        return schema


class InputSchema(IOSchema):
    "Input schema for the agent or tool"


class OutputSchema(IOSchema):
    "Output schema for the agent or tool"

    __response_field__: str | None = None

    _run_context: RunContext | None = PrivateAttr(default=None)

    @computed_field
    def _response(self) -> str:
        """
        Private response field that points to the actual output text as per agent usage.

        Subclasses can specify which field to use by setting the __response_field__ class attribute.
        Example:
            class MyOutputSchema(OutputSchema):
                __response_field__ = "output"
                output: str = Field(...)
        """
        if self.__response_field__ is not None:
            return getattr(self, self.__response_field__, "")
        return ""

    @property
    def run_context(self) -> RunContext | None:
        """Get the run context associated with this output."""
        return self._run_context


class TextInput(InputSchema):
    """Simple text-based input schema for unstructured content.

    Use this schema when you need a simple input format without structured fields,
    such as for conversational agents, chat interfaces, or any scenario where
    the input is just free-form text content.

    This is the text-specific implementation of InputSchema. For other modalities,
    use corresponding schemas like ImageInput, DocumentInput, etc. (when available).

    Example:
        # Simple chat agent usage
        agent = ChatAgent(config=config)
        result = await agent.arun(TextInput(content="Hello, how are you?"))

        # Multi-turn conversation with stateless=False
        await agent.arun(TextInput(content="My name is Alice"))
        await agent.arun(TextInput(content="What's my name?"))  # Remembers context
    """

    content: str = Field(description="The text content to process")


class TextOutput(OutputSchema):
    """Free-form text output. Used for conversational responses, clarifying questions, summaries, or any unstructured content."""

    __response_field__: str | None = "content"

    content: str = Field(description="The text content response")


InSchema = TypeVar("InSchema", bound=InputSchema)
OutSchema = TypeVar("OutSchema", bound=OutputSchema)


class AbstractBase(Generic[InSchema, OutSchema], ConfigBindingMixin, ABC):
    """Abstract base class for agents and tools.

    Generic is declared first in the bases so multi-inheritance with
    third-party frameworks using the standard ``(Generic[T], ABC)``
    layout (e.g. pydantic-ai) resolves cleanly.

    Includes streaming (astream/_astream) and sync run() directly.
    Formerly split across StreamingMixin and AsyncRunMixin.
    """

    input_schema: Type[InSchema]
    output_schema: Type[OutSchema]

    config_schema: Type[BaseModel] | None = None

    @classmethod
    def __class_getitem__(cls, params):
        """Create a concrete subclass with schemas set as class attributes.

        Enables runtime generic specialization:
            agent = AKDAgent[MyInput, MyOutput](config=...)
            agent = AKDAgent[MyInput, MyOutput | TextOutput, MyConfig](config=...)

        Config is optional (inherits from parent if omitted).
        """
        if not isinstance(params, tuple):
            params = (params,)
        if len(params) < 2:
            return super().__class_getitem__(params)

        in_schema, out_schema = params[0], params[1]
        attrs = {"input_schema": in_schema, "output_schema": out_schema}
        if len(params) == 3:
            attrs["config_schema"] = params[2]

        attrs["__module__"] = cls.__module__
        out_label = getattr(out_schema, "__name__", str(out_schema))
        attrs["__qualname__"] = f"{cls.__qualname__}[{in_schema.__name__}, {out_label}]"
        return type(cls.__name__, (cls,), attrs)

    # ── Streaming (folded from StreamingMixin) ──────────────────────

    async def astream(
        self,
        params: Any,
        run_context: RunContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Public streaming API with input/output validation.

        Wraps _astream() with validation:
        - Input validation before any events are yielded
        - Output validation on COMPLETED event before yielding
        """
        params = self._validate_input(params)

        async for event in self._astream(params, run_context, **kwargs):
            if event.event_type == StreamEventType.COMPLETED:
                output = event.output
                if output is not None:
                    output = self._validate_output(output)
                    yield CompletedEvent(
                        source=event.source,
                        message=event.message,
                        data=CompletedEventData(output=output),
                        run_context=event.run_context,
                    )
                    continue
            yield event

    async def _astream(
        self,
        params: Any,
        run_context: RunContext,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Internal streaming implementation. Override for custom streaming.

        Default yields STARTING/RUNNING, calls _arun(), yields COMPLETED/FAILED.
        """
        class_name = self.__class__.__name__

        yield StartingEvent(
            source=class_name,
            message=f"Starting {class_name}",
            run_context=run_context,
        )

        try:
            yield RunningEvent(
                source=class_name,
                message=f"Running {class_name}",
                run_context=run_context,
            )

            output = await self._arun(params, **kwargs)

            yield CompletedEvent(
                source=class_name,
                message=f"Completed {class_name}",
                data=CompletedEventData(output=output),
                run_context=run_context,
            )
        except Exception as e:
            yield FailedEvent(
                source=class_name,
                message=f"Failed: {e!s}",
                data=FailedEventData(error=str(e), error_type=type(e).__name__),
                run_context=run_context,
            )
            raise

    # ── Sync wrapper (folded from AsyncRunMixin) ─────────────────────

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run arun() synchronously."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                future = asyncio.ensure_future(self.arun(*args, **kwargs))
                return loop.run_until_complete(future)
            else:
                return asyncio.run(self.arun(*args, **kwargs))
        except RuntimeError:
            return asyncio.run(self.arun(*args, **kwargs))

    def __init__(
        self,
        config: BaseConfig | BaseModel | None = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the agent/tool.

        Validates required schemas at instantiation time. Abstract and
        intermediate base classes (e.g. BaseAgent, AKDAgent) can be defined
        without concrete schemas, but can only be instantiated through
        concrete subclasses that do define them.
        """
        self._validate_schemas()
        config = config or (self.config_schema() if self.config_schema else None) or BaseConfig()
        self.config = config
        self._kwargs = kwargs
        self._post_init()
        self.debug = debug or getattr(config, "debug", False)

    def _validate_schemas(self) -> None:
        """Validate input_schema / output_schema are defined and of proper types.

        Called at __init__ time. Raises TypeError if schemas are missing or
        invalid — which naturally prevents instantiation of abstract/intermediate
        classes without requiring hardcoded class name checks.
        """
        cls_name = type(self).__name__

        # input_schema
        input_schema = getattr(self, "input_schema", None)
        if input_schema is None:
            raise TypeError(f"{cls_name} must define 'input_schema' class attribute")
        if not isinstance(input_schema, type) or not issubclass(input_schema, (InputSchema, BaseModel)):
            raise TypeError(f"{cls_name}.input_schema must be a subclass of InputSchema")

        # output_schema (supports unions)
        output_schema = getattr(self, "output_schema", None)
        if output_schema is None:
            raise TypeError(f"{cls_name} must define 'output_schema' class attribute")
        origin = get_origin(output_schema)
        if origin in (types.UnionType, Union):
            args = get_args(output_schema)
            if not args or not all(
                isinstance(arg, type) and issubclass(arg, (OutputSchema, BaseModel)) for arg in args
            ):
                raise TypeError(f"{cls_name}.output_schema union members must be subclasses of OutputSchema")
        elif not isinstance(output_schema, type) or not issubclass(output_schema, (OutputSchema, BaseModel)):
            raise TypeError(f"{cls_name}.output_schema must be a subclass of OutputSchema or a union of them")

    def _post_init(self) -> None:
        """Post-initialization hook. Subclasses override for custom behavior.

        Config properties are created by ConfigBindingMixin at class definition time.
        Name/description/IO hints are bound via ConfigBindingMixin._bind_metadata().
        """
        for key, value in self._kwargs.items():
            setattr(self, key, value)

        self._bind_metadata()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AbstractBase:
        """Create instance from dict, with dynamic config model if needed."""
        debug = config_dict.pop("debug", False)

        # Use existing config_schema or create dynamic one
        if cls.config_schema is None and config_dict:
            fields = {k: (type(v), v) for k, v in config_dict.items() if v is not None}
            cls.config_schema = create_model(  # type: ignore[call-arg]
                f"{cls.__name__}Config",
                __base__=BaseConfig,
                **fields,
            )

        config = cls.config_schema(**config_dict) if cls.config_schema and config_dict else None
        return cls(config=config, debug=debug)

    def _validate_input(self, params: Any) -> InSchema:
        """Validate and convert input parameters."""
        return validate_input(self.input_schema, params)

    def _validate_output(self, output: Any) -> OutSchema:
        """Validate output against schema."""
        return validate_output(self.output_schema, output)

    async def arun(
        self,
        params: InSchema,
        **kwargs,
    ) -> OutSchema:
        """
        Runs the agent with the provided parameters asynchronously.
        Args:
            params (InSchema): The structured input parameters for the agent.
            **kwargs: Additional keyword arguments.
        Returns:
            OutSchema: The output from the agent after processing the input.
        """

        params = self._validate_input(params)
        if self.debug:
            logger.debug(
                f"Running {self.__class__.__name__} with params: {params}",
            )
        output = None
        try:
            output = await self._arun(params, **kwargs)
            output = self._validate_output(output)
        except HumanInputRequired:
            logger.warning(f"{self.__class__.__name__}: HumanInputRequired (flow control)")
            raise
        except Exception as e:
            logger.error(f"Error running {self.__class__.__name__}: {e}")
            raise
        return output

    @abstractmethod
    async def _arun(
        self,
        params: InSchema,
        **kwargs,
    ) -> OutSchema:
        """Internal method to run the agent with the provided parameters asynchronously.
        Args:
            params (InSchema): The structured input parameters for the agent.
            **kwargs: Additional keyword arguments.
        Returns:
            OutSchema: The output from the agent after processing the input.
        """
        raise NotImplementedError()


__all__ = [
    "AbstractBase",
    "BaseConfig",
    "IOSchema",
    "InputSchema",
    "OutputSchema",
]
