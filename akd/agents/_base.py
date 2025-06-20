from __future__ import annotations

from typing import Optional

from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict

from akd._base import AbstractBase
from akd.configs.project import CONFIG
from akd.configs.prompts import DEFAULT_SYSTEM_PROMPT
from akd.structures import InputSchema, OutputSchema
from akd.utils import AsyncRunMixin, LangchainToolMixin


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

    @classmethod
    def from_config(cls, config: BaseModel) -> BaseAgent:
        """
        Create an instance of the agent from a Pydantic configuration model.

        Args:
            config (BaseModel): Pydantic model containing initialization
                parameters for the agent.

        Returns:
            Self: An instance of the agent class.
        """
        # Convert Pydantic model to dict and use as kwargs
        config_dict = config.model_dump()

        # Extract debug if it exists
        debug = config_dict.pop("debug", False)

        # Pass remaining config as kwargs
        return cls(debug=debug, **config_dict)

    # def __set_attrs_from_config(self):
    #     for attr, value in self.config.model_dump().items():
    #         setattr(self, attr, value)


class LangBaseAgent[
    InputSchema: BaseModel,
    OutputSchema: BaseModel,
](
    AsyncRunMixin,
    LangchainToolMixin,
):
    def __init__(
        self,
        config: Optional[ConfigDict] = None,
        debug: bool = False,
    ) -> None:
        client = ChatOpenAI(
            api_key=CONFIG.model_config_settings.api_keys.openai,
            model=CONFIG.model_config_settings.model_name,
            temperature=0.0,
        )
        self.config = config or ConfigDict(
            client=client,
            extra="allow",
            system_prompt=ChatPromptTemplate.from_messages(
                [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    MessagesPlaceholder(variable_name="memory"),
                ],
            ),
        )
        self.debug = debug
        self.client = client
        self.memory = ChatMessageHistory()
        self.system_prompt = self.config["system_prompt"]

    async def get_response_async(
        self,
        response_model=None,
    ) -> BaseModel:
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

        return response

    async def arun(
        self,
        user_input: Optional[InputSchema] = None,
    ) -> OutputSchema:
        """
        Runs the chat agent with the given user input asynchronously.

        Args:
            user_input (Optional[InputSchema]):
                The input from the user.
                If not provided, skips adding to memory.

        Returns:
            OutputSchema: The response from the chat agent.
        """

        if user_input:
            self.memory.add_user_message(user_input.model_dump_json(exclude={"type"}))

        response = await self.get_response_async(
            response_model=self.output_schema,
        )

        self.memory.add_ai_message(response.model_dump_json(exclude={"type"}))

        return response
