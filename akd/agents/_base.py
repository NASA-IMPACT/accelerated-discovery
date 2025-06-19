from typing import Optional

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, ConfigDict
from akd.configs.project import CONFIG

from ..utils import AsyncRunMixin, LangchainToolMixin


class InputSchema(BaseModel):
    """Base schema for tool input."""
    pass

class OutputSchema(BaseModel):
    """Base schema for tool output."""
    pass



class BaseAgent[
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
        client= ChatOpenAI(
            api_key=CONFIG.model_config_settings.api_keys.openai,
            model=CONFIG.model_config_settings.model_name,
            temperature=0.0,
            )
        self.config = config or ConfigDict(
            client=client,
            extra="allow",
        )
        self.debug = debug
        self.client = client
        self.memory = ChatMessageHistory()
        self.system_prompt = ChatPromptTemplate.from_messages(
                    [
                        {"role": "system", "content": """\
        IDENTITY and PURPOSE
        This is a conversation with a helpful and friendly AI assistant.

        OUTPUT INSTRUCTIONS
        - Always respond using the proper JSON schema.
        - Always use the available additional information and context to enhance the response.
        """},
                        MessagesPlaceholder(variable_name="memory"),
                    ]
                )


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
            input=self.memory.messages
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
