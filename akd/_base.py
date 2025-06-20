import inspect
from abc import ABC, ABCMeta, abstractmethod
from typing import Type

from loguru import logger
from pydantic import BaseModel

from akd.utils import AsyncRunMixin, LangchainToolMixin


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
        if name == "AbstractBase":
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

    input_schema: Type[InputSchema]
    output_schema: Type[OutputSchema]

    def __init__(self, *args, debug: bool = False, **kwargs) -> None:
        """
        Initializes the BaseAgent with a language model client and memory.

        Args:
            debug (bool): If True, enables debug mode for additional logging.
        """
        self.debug = debug

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

        if self.input_schema and not isinstance(params, self.input_schema):
            raise TypeError(
                f"params must be an instance of {self.input_schema.__name__}",
            )
        if self.debug:
            logger.debug(
                f"Running {self.__class__.__name__} with params: {params.model_dump()}",
            )
        output = None
        try:
            output = await self._arun(params, **kwargs)
            if not isinstance(output, self.output_schema):
                raise TypeError(
                    f"Output must be an instance of {self.output_schema.__name__}",
                )
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
