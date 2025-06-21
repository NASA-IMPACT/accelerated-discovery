from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, cast

from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

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
        self.system_prompt = ChatPromptTemplate.from_messages(
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
        response = await structured_client.ainvoke(
            input=self.memory.messages,
        )

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
