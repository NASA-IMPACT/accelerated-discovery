from __future__ import annotations

import inspect
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Type, cast

from loguru import logger
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationError,
    computed_field,
    create_model,
)

from akd.utils import get_model_fields, to_snake_case

from .errors import HumanInputRequired, SchemaValidationError
from .streaming import StreamingMixin
from .structures import RunContext
from .utils import AsyncRunMixin


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
    """Simple text-based output schema for unstructured content.

    Use this schema when your agent produces free-form text output without
    structured fields, such as for conversational responses, summaries,
    or any scenario where the output is just text content.

    This is the text-specific implementation of OutputSchema. For other modalities,
    use corresponding schemas like ImageOutput, DocumentOutput, etc. (when available).

    Example:
        class ChatAgent(LiteLLMInstructorBaseAgent[TextInput, TextOutput]):
            '''Simple conversational agent.'''
            input_schema = TextInput
            output_schema = TextOutput

        result = await agent.arun(TextInput(content="Tell me a joke"))
        print(result.content)  # The agent's text response
    """

    __response_field__: str | None = "content"

    content: str = Field(description="The text content response")


def _make_config_property(field_name: str):
    """Create a property that references a config field.

    This factory function creates properties at class definition time that delegate
    to self.config.field_name, maintaining reference semantics between agent.x and
    agent.config.x. Includes fallback to instance __dict__ for pre-init access.

    Args:
        field_name: Name of the config field to create a property for

    Returns:
        property: A property descriptor with getter/setter
    """

    def getter(self):
        # If config doesn't exist yet, fall back to instance attribute
        if not hasattr(self, "config") or self.config is None:
            return self.__dict__.get(field_name)
        return getattr(self.config, field_name)

    def setter(self, value):
        # If config doesn't exist yet, set as instance attribute
        if not hasattr(self, "config") or self.config is None:
            self.__dict__[field_name] = value
        else:
            setattr(self.config, field_name, value)

    return property(getter, setter)


def _make_computed_property(field_name: str):
    """Create a read-only property for a computed config field.

    Computed fields (decorated with @computed_field) are read-only and
    dynamically calculated from other config values.

    Args:
        field_name: Name of the computed field

    Returns:
        property: A read-only property descriptor
    """

    def getter(self):
        # If config doesn't exist yet, fall back to instance attribute
        if not hasattr(self, "config") or self.config is None:
            return self.__dict__.get(field_name)
        return getattr(self.config, field_name)

    return property(getter)


class AbstractBaseMeta(ABCMeta):
    """Metaclass that validates required schema attributes and creates config properties."""

    @staticmethod
    def _create_config_properties(target_class, dct):
        """Create properties for config fields at class definition time.

        This creates properties that reference self.config.field_name, maintaining
        reference semantics between agent.x and agent.config.x.

        Args:
            target_class: The class being created
            dct: The class dictionary from __new__
        """
        if not hasattr(target_class, "config_schema") or target_class.config_schema is None:
            return

        # Create properties for regular model fields
        if hasattr(target_class.config_schema, "model_fields"):
            for field_name in target_class.config_schema.model_fields.keys():
                # Skip if explicitly defined in this class's dict
                if field_name in dct:
                    continue
                # Skip if already a property (from parent or exposed params)
                if isinstance(getattr(target_class, field_name, None), property):
                    continue
                # Create property that references self.config.field_name
                setattr(target_class, field_name, _make_config_property(field_name))

        # Create read-only properties for computed fields
        if hasattr(target_class.config_schema, "model_computed_fields"):
            for field_name in target_class.config_schema.model_computed_fields.keys():
                if field_name in dct:
                    continue
                if isinstance(getattr(target_class, field_name, None), property):
                    continue
                setattr(target_class, field_name, _make_computed_property(field_name))

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        # Create config properties at class definition time for ALL classes
        # This must happen before early return to ensure base classes get properties too
        AbstractBaseMeta._create_config_properties(cls, dct)

        # Skip schema validation for base classes
        if name in [
            "AbstractBase",
            "UnrestrictedAbstractBase",
            "BaseAgent",
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
                if not isinstance(cls.output_schema, type) or not issubclass(
                    cls.output_schema,
                    (OutputSchema, BaseModel),
                ):
                    raise TypeError(
                        f"{name}.output_schema must be a subclass of OutputSchema",
                    )

        return cls


class AbstractBase[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](StreamingMixin, AsyncRunMixin, ABC, metaclass=AbstractBaseMeta):
    """
    Abstract base class for agents and tools that interact with a language model.
    This class provides the basic structure for an agent or tool that can handle
    asynchronous operations, manage memory, and utilize a language model
    for generating responses based on user input.
    """

    input_schema: Type[InSchema]
    output_schema: Type[OutSchema]

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
        config = config or (self.config_schema() if self.config_schema else None) or BaseConfig()
        self.config = config
        self._kwargs = kwargs
        self._post_init()
        self.debug = debug or getattr(config, "debug", False)

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

        self.description = (getattr(self, "description", None) or self.__class__.__doc__ or "").strip()

        # Add input/output schema info to description if io_hints is True
        if getattr(self, "io_hints", True):
            _in_schema = self._input_schema_info
            if _in_schema:
                self.description += f"\n\nINPUT FIELD DESCRIPTIONS:\n{_in_schema}"
            _out_schema = self._output_schema_info
            if _out_schema:
                self.description += f"\n\nOUTPUT FIELD DESCRIPTIONS:\n{_out_schema}"

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

        fields = get_model_fields(self.output_schema, skip_no_description=False)
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
        if not isinstance(params, self.input_schema):
            if isinstance(params, dict):
                try:
                    params = self.input_schema(**params)
                except ValidationError as e:
                    raise SchemaValidationError(f"Invalid input parameters: {e}") from e
            else:
                raise TypeError(
                    f"params must be an instance of {self.input_schema.__name__}",
                )
        return params

    def _validate_output(self, output: Any) -> OutSchema:
        """Validate output against schema."""
        if not isinstance(output, self.output_schema):
            raise TypeError(
                f"Output must be an instance of {self.output_schema.__name__}",
            )
        return output

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
](StreamingMixin, AsyncRunMixin, ABC, metaclass=AbstractBaseMeta):
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
