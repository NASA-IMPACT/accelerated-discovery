from typing import Optional

import openai
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings

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
        config = config or ConfigDict(
            client=ChatOpenAI(
                    api_key=CONFIG.model_config_settings.api_keys.openai,
                    model=CONFIG.model_config_settings.model_name,
                    temperature=0.0,
            ),
            extra='allow'
        )
        self.debug = debug
        self.config = config
        self.client = config['client']
        # super().__init__(config)


    async def get_response_async(
        self,
        response_model=None,
    ) -> BaseModel:
        """
        Obtains a response from the language model synchronously.

        Args:
            response_model (Type[BaseModel], optional):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            Type[BaseModel]: The response from the language model.
        """
    
        response_model = response_model or self.output_schema
        client = self.client.with_structured_output(response_model)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt_generator.generate_prompt(),
            },
        ] + self.memory.get_history()

        response = await client.ainvoke(
            messages=messages
        )

        return response

    async def arun(
        self,
        user_input: Optional[InputSchema] = None,
    ) -> OutputSchema:
        """
        Runs the chat agent with the given user input synchronously.

        Args:
            user_input (Optional[InputSchema]):
                The input from the user.
                If not provided, skips adding to memory.

        Returns:
            OutputSchema: The response from the chat agent.
        """
        pass
    
        # TODO: Replace with node specific code, or LangGraph equivalent

        # if user_input:
        #     self.memory.initialize_turn()
        #     self.current_user_input = user_input
        #     self.memory.add_message("user", user_input)

        # if self._is_async_client:
        #     response = await self.get_response_async(
        #         response_model=self.output_schema,
        #     )
        # else:
        #     logger.warning(
        #         "Using synchronous client for async run.",
        #     )
        #     response = self.get_response(
        #         response_model=self.output_schema,
        #     )
        # self.memory.add_message("assistant", response)

        # return response
