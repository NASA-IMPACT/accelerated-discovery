from __future__ import annotations

import asyncio
import inspect
import types
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Type, Union, cast, get_args, get_origin

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, computed_field, create_model

from akd.utils import get_model_fields, to_snake_case

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


class AbstractBaseMeta(ABCMeta):
    """Metaclass that validates required schema attributes.

    Config property creation is handled by ConfigBindingMixin.__init_subclass__.
    """

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        # Skip schema validation for base classes
        if name in [
            "AbstractBase",
            "UnrestrictedAbstractBase",
            "BaseAgent",
            "AKDAgent",
            "InstructorBaseAgent",
            "LiteLLMInstructorBaseAgent",
            "BaseTool",
        ]:
            return cls

        # Check if this class inherits from AbstractBase
        if any(isinstance(base, AbstractBaseMeta) for base in bases):
            # Validate input_schema
            if "input_schema" not in dct and not any(hasattr(base, "input_schema") for base in bases):
                raise TypeError(f"{name} must define 'input_schema' class attribute")

            # Validate output_schema
            if "output_schema" not in dct and not any(hasattr(base, "output_schema") for base in bases):
                raise TypeError(f"{name} must define 'output_schema' class attribute")

            # Validate schema types if they exist
            if hasattr(cls, "input_schema") and cls.input_schema is not None:
                if not isinstance(cls.input_schema, type) or not issubclass(
                    cls.input_schema,
                    (InputSchema, BaseModel),
                ):
                    raise TypeError(
                        f"{name}.input_schema must be a subclass of InputSchema",
                    )

            if hasattr(cls, "output_schema") and cls.output_schema is not None:
                output_schema_decl = cls.output_schema
                origin = get_origin(output_schema_decl)
                if origin in (types.UnionType, Union):
                    args = get_args(output_schema_decl)
                    valid_union = bool(args) and all(
                        isinstance(arg, type) and issubclass(arg, (OutputSchema, BaseModel)) for arg in args
                    )
                    if not valid_union:
                        raise TypeError(
                            f"{name}.output_schema union members must be subclasses of OutputSchema",
                        )
                elif not isinstance(output_schema_decl, type) or not issubclass(
                    output_schema_decl,
                    (OutputSchema, BaseModel),
                ):
                    raise TypeError(
                        f"{name}.output_schema must be a subclass of OutputSchema or a union of them",
                    )

        return cls


class AbstractBase[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](ConfigBindingMixin, ABC, metaclass=AbstractBaseMeta):
    """Abstract base class for agents and tools.

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
        """
        Initializes the BaseAgent with a language model client and memory.

        Args:
            debug (bool): If True, enables debug mode for additional logging.
            config (BaseModel, optional): Configuration object containing all parameters
            debug (bool): If True, enables debug mode for additional logging.
            **kwargs: Additional keyword arguments (merged with config)
        """
        config = config or (self.config_schema() if self.config_schema else None) or BaseConfig()
        self.config = config
        self._kwargs = kwargs
        self._post_init()
        self.debug = debug or getattr(config, "debug", False)

    def _post_init(self) -> None:
        """Post-initialization hook. Subclasses override for custom behavior.

        Config properties are created by ConfigBindingMixin at class definition time.
        Name/description/IO hints are bound via ConfigBindingMixin._bind_metadata().
        """
        for key, value in self._kwargs.items():
            setattr(self, key, value)

        self._bind_metadata()

    @property
    def _input_schema_info(self) -> str:
        """
        Extract field names and descriptions from input schema.

        Returns:
            str: Formatted string with field information, empty if no input schema.
        """
        # avoid circular dependency
        if not hasattr(self, "input_schema") or not self.input_schema:
            return ""

        fields = get_model_fields(self.input_schema, skip_no_description=False)
        if not fields:
            return ""

        return "\n".join(
            [f"- **{field['name']}**: {field.get('description', field['name'].replace('_', ' '))}" for field in fields],
        )

    @property
    def _output_schema_info(self) -> str:
        """
        Extract field names and descriptions from output schema.

        Returns:
            str: Formatted string with field information, empty if no output schema.
        """

        # avoid circular dependency
        if not hasattr(self, "output_schema") or not self.output_schema:
            return ""
        schema_decl = self.output_schema
        if not isinstance(schema_decl, type):
            return ""
        fields = get_model_fields(schema_decl, skip_no_description=False)
        if not fields:
            return ""

        return "\n".join(
            [f"- **{field['name']}**: {field.get('description', field['name'].replace('_', ' '))}" for field in fields],
        )

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


class UnrestrictedAbstractBase[
    InSchema: BaseModel,
    OutSchema: BaseModel,
](ABC, metaclass=AbstractBaseMeta):
    """
    Abstract base class for agents and tools that interact with a language model.
    This class provides the basic structure for an agent or tool that can handle
    asynchronous operations, manage memory, and utilize a language model
    for generating responses based on user input.

    This class does not enforce input and output schema types, allowing for more flexibility
    in the types of parameters and outputs used.
    It is intended for use cases where strict type checking is not required.
    It is recommended to use this class only when necessary, as it bypasses the type safety
    provided by the schema validation in the AbstractBase class.
    """

    config_schema: Type[BaseModel] | None = None

    def __init__(
        self,
        config: BaseConfig | BaseModel | None = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the BaseAgent with a language model client and memory.

        Args:
            debug (bool): If True, enables debug mode for additional logging.
            config (BaseModel, optional): Configuration object containing all parameters
            debug (bool): If True, enables debug mode for additional logging.
            **kwargs: Additional keyword arguments (merged with config)
        """
        debug = getattr(config, "debug", False) or debug
        self.debug = debug
        self.config = config
        self._kwargs = kwargs
        self._post_init()

    def _post_init(self) -> None:
        """
        Post-initialization hook to perform any additional setup after
        the instance has been initialized.
        This can be overridden by subclasses for custom behavior.

        Note: Config properties are created by AbstractBaseMeta at class definition time,
        not during instance initialization.
        """
        for key, value in self._kwargs.items():
            setattr(self, key, value)

        # Set default name from class name if not provided
        if getattr(self, "name", None) is None:
            self.name = to_snake_case(self.__class__.__name__)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> UnrestrictedAbstractBase:
        """Create instance from dict, with dynamic config model if needed."""
        debug = config_dict.pop("debug", False)

        # Use existing config_schema or create dynamic one
        if cls.config_schema is None and config_dict:
            fields = {k: (type(v), v) for k, v in config_dict.items() if v is not None}
            cls.config_schema = create_model(
                f"{cls.__name__}Config",
                __base__=BaseConfig,
                **fields,
            )

        config = cls.config_schema(**config_dict) if cls.config_schema and config_dict else None
        return cls(config=config, debug=debug)

    def _validate_input(self, params: Any) -> InSchema:
        """Validate and convert input parameters."""
        if not isinstance(params, BaseModel):
            raise TypeError("params must be an instance of pydantic BaseModel")
        return cast(InSchema, params)

    def _validate_output(self, output: Any) -> OutSchema:
        """Validate and convert input parameters."""
        if not isinstance(output, BaseModel):
            raise TypeError("output must be an instance of pydantic BaseModel")
        return cast(OutSchema, output)

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
    "UnrestrictedAbstractBase",
]
