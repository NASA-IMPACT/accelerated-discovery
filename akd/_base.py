from __future__ import annotations

import inspect
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Type, cast

from loguru import logger
from pydantic import BaseModel, ValidationError, create_model

from akd.errors import SchemaValidationError
from akd.utils import AsyncRunMixin, LangchainToolMixin


class BaseConfig(BaseModel):
    """
    Base configuration class for agents and tools.
    This class can be extended to define specific configurations for agents or tools.
    """

    model_config = {
        "extra": "forbid",  # Disallow extra fields
    }

    debug: bool = False  # Debug mode flag


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


class AbstractBaseMeta(ABCMeta):
    """Metaclass that validates required schema attributes."""

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        # Skip validation for the base class itself
        if name in [
            "AbstractBase",
            "UnrestrictedAbstractBase",
            "BaseAgent",
            "LangBaseAgent",
            "InstructorBaseAgent",
            "BaseTool",
        ]:
            return cls

        # Check if this class inherits from AbstractBase
        if any(isinstance(base, AbstractBaseMeta) for base in bases):
            # Validate input_schema
            if "input_schema" not in dct and not any(
                hasattr(base, "input_schema") for base in bases
            ):
                raise TypeError(f"{name} must define 'input_schema' class attribute")

            # Validate output_schema
            if "output_schema" not in dct and not any(
                hasattr(base, "output_schema") for base in bases
            ):
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
](AsyncRunMixin, LangchainToolMixin, ABC, metaclass=AbstractBaseMeta):
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
        config = (
            config
            or (self.config_schema() if self.config_schema else None)
            or BaseConfig()
        )
        self.config = config
        self._kwargs = kwargs
        self._post_init()
        self.debug = getattr(config, "debug", False) or debug

    def _post_init(self) -> None:
        """
        Post-initialization hook to perform any additional setup after
        the instance has been initialized.
        This can be overridden by subclasses for custom behavior.
        """
        self.__set_attrs_from_config()
        for key, value in self._kwargs.items():
            setattr(self, key, value)

    def __set_attrs_from_config(self):
        if self.config is None:
            return
        for attr, value in self.config.model_dump().items():
            setattr(self, attr, value)

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

        config = (
            cls.config_schema(**config_dict)
            if cls.config_schema and config_dict
            else None
        )
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

    def _truncate_for_debug(self, obj: Any, max_length: int = 250) -> str:
        """Truncate object representation for debug logging."""
        obj_str = str(obj)
        if len(obj_str) <= max_length:
            return obj_str
        return obj_str[:max_length] + "..."

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
                f"Running {self.__class__.__name__} with params: {self._truncate_for_debug(params)}",
            )
        output = None
        try:
            output = await self._arun(params, **kwargs)
            output = self._validate_output(output)
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
](AsyncRunMixin, LangchainToolMixin, ABC):
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
        """
        self.__set_attrs_from_config()
        for key, value in self._kwargs.items():
            setattr(self, key, value)

    def __set_attrs_from_config(self):
        if self.config is None:
            return
        for attr, value in self.config.model_dump().items():
            setattr(self, attr, value)

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

        config = (
            cls.config_schema(**config_dict)
            if cls.config_schema and config_dict
            else None
        )
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

    def _truncate_for_debug(self, obj: Any, max_length: int = 250) -> str:
        """Truncate object representation for debug logging."""
        obj_str = str(obj)
        if len(obj_str) <= max_length:
            return obj_str
        return obj_str[:max_length] + "..."

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
                f"Running {self.__class__.__name__} with params: {self._truncate_for_debug(params)}",
            )
        output = None
        try:
            output = await self._arun(params, **kwargs)
            output = self._validate_output(output)
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
