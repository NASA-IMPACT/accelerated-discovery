from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, cast

import instructor
import openai
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from akd._base import AbstractBase, BaseConfig, InputSchema, OutputSchema
from akd.configs.project import CONFIG
from akd.configs.prompts import DEFAULT_SYSTEM_PROMPT


class BaseAgentConfig(BaseConfig):
    """Configuration class for LangBaseAgent."""

    api_key: Optional[str] = None
    model_name: Optional[str] = None
    temperature: float = 0.0
    system_prompt: Optional[str] = None


class BaseAgent[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](AbstractBase):
    """
    Base class for chat agents that interact with a language model.

    This class provides the basic structure for an agent that can handle
    asynchronous operations, manage memory, and utilize a language model
    for generating responses based on user input.
    """

    config_schema = BaseAgentConfig

    @property
    def memory(self) -> Any:
        """
        Returns the memory of the agent, implemented by subclasses.
        This property should return the memory structure used by the agent,
        which typically includes past messages or interactions.
        Raises:
            NotImplementedError: If the property is not implemented in a subclass.
        Args:
            None

        Returns:
            list[Any | BaseModel]: The memory of the agent.
        """
        raise NotImplementedError("Attribute 'memory' not implemented.")

    def reset_memory(self) -> None:
        pass

    @abstractmethod
    async def get_response_async(
        self,
        *args,
        **kwargs,
    ) -> OutputSchema:
        """
        Obtains a response from the language model asynchronously.

        Args:
            response_model (Optional[OutputSchema]):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            OutputSchema: The response from the language model.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class LangBaseAgent[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](BaseAgent):
    """Base class for LangChain-based chat agents.
    This class provides a foundation for agents that use LangChain's
    ChatOpenAI client for generating responses based on user input.
    It includes configuration options for the language model and memory management.

    Note:
        The object attributes (like `api_key`, `model_name` etc.) are dynamically set from the config.
    """

    def __init__(
        self,
        config: BaseAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        config = config or BaseAgentConfig(
            api_key=CONFIG.model_config_settings.api_keys.openai,
            model_name=CONFIG.model_config_settings.model_name,
            temperature=0.0,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            debug=debug,
        )

        super().__init__(config=config, debug=debug)

        # Create the OpenAI client
        self.client = ChatOpenAI(
            api_key=self.api_key or CONFIG.model_config_settings.api_keys.openai,  # type: ignore
            model=self.model_name or CONFIG.model_config_settings.model_name,  # type: ignore
            temperature=self.temperature,  # type: ignore
        )

        # Initialize memory
        self._memory = ChatMessageHistory()

        # Create system prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                {
                    "role": "system",
                    "content": self.system_prompt or DEFAULT_SYSTEM_PROMPT,
                },
                MessagesPlaceholder(variable_name="memory"),
            ],
        )

    @property
    def memory(self) -> ChatMessageHistory:
        return self._memory

    def reset_memory(self) -> None:
        """
        Resets the memory of the agent.
        This method clears the chat message history, effectively resetting the agent's memory.
        """
        self.memory.clear()

    async def get_response_async(
        self,
        response_model: type[OutputSchema] | None = None,
    ) -> OutSchema:
        """
        Obtains a response from the language model asynchronously.

        Args:
            response_model (Type[BaseModel], optional):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            Type[BaseModel]: The response from the language model.
        """
        response_model = response_model or self.output_schema
        structured_client = self.client.with_structured_output(response_model)

        # Format messages using the prompt template
        formatted_messages = self.prompt_template.format_messages(
            memory=self.memory.messages
        )

        response = await structured_client.ainvoke(formatted_messages)

        return cast(OutSchema, response)

    async def _arun(
        self,
        params: InSchema,
        **kwargs,
    ) -> OutSchema:
        """
        Runs the chat agent with the given user input asynchronously.

        Args:
            user_input (Optional[InputSchema]):
                The input from the user.
                If not provided, skips adding to memory.

        Returns:
            OutputSchema: The response from the chat agent.
        """

        if params:
            self.memory.add_user_message(params.model_dump_json(exclude={"type"}))

        response = await self.get_response_async(
            response_model=self.output_schema,
        )

        self.memory.add_ai_message(response.model_dump_json(exclude={"type"}))

        return response


class InstructorBaseAgent[
    InSchema: BaseModel,
    OutSchema: BaseModel,
](BaseAgent):
    """Base class for instructor-based chat agents.
    Note:
        The object attributes (like `api_key`, `model_name` etc.) are dynamically set from the config.
    """

    def __init__(
        self,
        config: BaseAgentConfig | None = None,
        debug: bool = False,
    ) -> None:
        config = config or BaseAgentConfig(
            api_key=CONFIG.model_config_settings.api_keys.openai,
            model_name=CONFIG.model_config_settings.model_name,
            temperature=0.0,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            debug=debug,
        )

        super().__init__(config=config, debug=debug)

        # Create the OpenAI client
        self.client = instructor.from_openai(
            openai.AsyncOpenAI(
                api_key=CONFIG.model_config_settings.api_keys.openai,
            ),
        )

        # Initialize memory
        self._memory = []

    @property
    def memory(self) -> list[dict[str, str]]:
        return self._memory

    def reset_memory(self) -> None:
        """
        Resets the memory of the agent.
        This method clears the chat message history, effectively resetting the agent's memory.
        """
        self.memory.clear()

    def _create_instructor_compatible_model(self, response_model: type[OutputSchema]):
        """Create a model that's compatible with instructor but avoids IOSchema validation."""
        from pydantic import create_model

        # Get the fields from the original model
        fields = {}
        for field_name, field_info in response_model.model_fields.items():
            fields[field_name] = (field_info.annotation, field_info)

        # Create a new model that inherits from BaseModel directly (not IOSchema)
        # This avoids the docstring validation issue
        instructor_model = create_model(
            response_model.__name__,
            __base__=BaseModel,
            **fields,
        )

        # Copy over the docstring and other metadata
        instructor_model.__doc__ = response_model.__doc__

        return instructor_model

    async def get_response_async(
        self,
        response_model: type[OutputSchema] | None = None,
    ) -> OutSchema:
        """
        Obtains a response from the language model asynchronously.

        Args:
            response_model (Type[BaseModel], optional):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            Type[BaseModel]: The response from the language model.
        """
        response_model = response_model or self.output_schema
        instructor_model = self._create_instructor_compatible_model(response_model)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
        ] + self.memory

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            response_model=instructor_model,
        )

        response_data = response.model_dump()
        response = response_model(**response_data)
        return cast(OutSchema, response)

    async def _arun(
        self,
        params: InSchema,
        **kwargs,
    ) -> OutSchema:
        """
        Runs the chat agent with the given user input asynchronously.

        Args:
            user_input (Optional[InputSchema]):
                The input from the user.
                If not provided, skips adding to memory.

        Returns:
            OutputSchema: The response from the chat agent.
        """

        if params:
            self.memory.append(
                dict(
                    role="user",
                    content=params.model_dump_json(exclude={"type"}),
                ),
            )

        response = await self.get_response_async(
            response_model=self.output_schema,
        )

        self.memory.append(
            dict(
                role="assistant",
                content=response.model_dump_json(exclude={"type"}),
            ),
        )

        return response
