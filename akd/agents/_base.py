from __future__ import annotations

from abc import abstractmethod
from typing import Any, cast

import instructor
import openai
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import AnyUrl, BaseModel, Field

from akd._base import AbstractBase, BaseConfig, InputSchema, OutputSchema
from akd.configs.project import CONFIG
from akd.configs.prompts import DEFAULT_SYSTEM_PROMPT


class BaseAgentConfig(BaseConfig):
    """Configuration class for LangBaseAgent."""

    base_url: AnyUrl | None = Field(default=CONFIG.model_config_settings.base_url)
    api_key: str | None = Field(default=CONFIG.model_config_settings.api_keys.openai)
    model_name: str | None = Field(default=CONFIG.model_config_settings.model_name)
    temperature: float = 0.0
    system_prompt: str | None = Field(default=DEFAULT_SYSTEM_PROMPT)
    llm_timeout: float | None = Field(
        default=45.0,
        description="Timeout in seconds for individual LLM API calls. Set to None to disable. Timeouts trigger retry with exponential backoff.",
    )


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
        super().__init__(config=config, debug=debug)

        # Create the OpenAI client
        self.client = ChatOpenAI(
            api_key=self.api_key,
            model=self.model_name,
            base_url=str(self.base_url),
            temperature=self.temperature,  # type: ignore
        )

        # Initialize memory
        self._memory = ChatMessageHistory()

        # Create system prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                {
                    "role": "system",
                    "content": self.system_prompt,
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
        structured_client = self.client.with_structured_output(
            response_model,
            method="function_calling",
        )

        # Format messages using the prompt template
        formatted_messages = self.prompt_template.format_messages(
            memory=self.memory.messages,
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
    InSchema: InputSchema,
    OutSchema: OutputSchema,
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
        super().__init__(config=config, debug=debug)

        # Create the OpenAI client
        self.client = instructor.from_openai(
            openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=str(self.base_url),
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
        import asyncio
        import time

        response_model = response_model or self.output_schema
        instructor_model = self._create_instructor_compatible_model(response_model)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
        ] + self.memory

        # GPT-5 series only supports temperature=1, override if needed
        temperature = (
            1.0
            if self.model_name and self.model_name.startswith("gpt-5")
            else self.temperature
        )

        # Debug logging with detailed metrics
        if self.debug:
            import json

            task = asyncio.current_task()
            task_name = task.get_name() if task else "unknown"
            from loguru import logger

            # Calculate prompt size (approximate token count)
            prompt_text = json.dumps(messages)
            prompt_chars = len(prompt_text)
            prompt_tokens_est = prompt_chars // 4  # Rough estimate: 1 token ≈ 4 chars

            logger.debug(
                f"[{task_name}] LLM call starting: model={self.model_name}, schema={response_model.__name__}, memory_msgs={len(self.memory)}, prompt_tokens~{prompt_tokens_est}",
            )

        start_time = time.time()
        api_call_start = None

        # Create timeout warning task
        async def log_slow_request():
            """Log warnings if request takes too long."""
            thresholds = [5, 10, 30, 60, 120, 300]  # seconds
            for threshold in thresholds:
                await asyncio.sleep(threshold)
                elapsed = time.time() - start_time
                if self.debug:
                    from loguru import logger

                    logger.warning(
                        f"[{task_name}] ⏱️  LLM call still running after {elapsed:.0f}s (threshold: {threshold}s)",
                    )

        # Start timeout warning task (will be cancelled when request completes)
        timeout_task = asyncio.create_task(log_slow_request())

        try:
            api_call_start = time.time()

            # Log the actual API call details
            if self.debug:
                from loguru import logger

                logger.debug(
                    f"[{task_name}] 🌐 Making OpenAI API call: base_url={self.base_url}, model={self.model_name}",
                )

            # Get timeout from config
            timeout = getattr(self.config, "llm_timeout", 45.0)

            # Use rate limiter to prevent OpenAI throttling of concurrent requests
            from akd.agents.data_search.utils.rate_limiter import get_rate_limiter

            rate_limiter = get_rate_limiter(max_concurrent_requests=2)

            if self.debug:
                from loguru import logger

                available = rate_limiter.available_slots()
                timeout_str = f"{timeout}s" if timeout is not None else "∞"
                logger.debug(
                    f"[{task_name}] 🚦 Rate limiter: {available}/{rate_limiter.max_concurrent} slots available, timeout={timeout_str}",
                )

            # Define API call as async function for timeout wrapping
            async def make_api_call():
                async with rate_limiter:
                    return await self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        temperature=temperature,
                        response_model=instructor_model,
                    )

            # Apply timeout if configured
            if timeout is not None:
                try:
                    response = await asyncio.wait_for(make_api_call(), timeout=timeout)
                except asyncio.TimeoutError:
                    # Re-raise with clear marker for retry logic
                    elapsed = time.time() - start_time
                    raise RuntimeError(
                        f"LLM timeout: Call exceeded {timeout}s limit (elapsed: {elapsed:.1f}s, model: {self.model_name})",
                    )
            else:
                response = await make_api_call()

            # Cancel timeout warning task
            timeout_task.cancel()

            api_call_duration = time.time() - api_call_start

            if self.debug:
                from loguru import logger

                # Get response metadata if available
                response_tokens = (
                    getattr(response, "usage", {}).get("completion_tokens", "N/A")
                    if hasattr(response, "usage")
                    else "N/A"
                )
                request_id = (
                    getattr(response, "id", "N/A") if hasattr(response, "id") else "N/A"
                )
                total_duration = time.time() - start_time

                # Log completion with metadata
                logger.debug(
                    f"[{task_name}] LLM call completed: api_time={api_call_duration:.2f}s, total={total_duration:.2f}s, response_tokens~{response_tokens}",
                )

                # Log request ID for support tickets
                if request_id != "N/A":
                    logger.debug(f"[{task_name}] OpenAI request_id: {request_id}")

                # Warn if request was unusually slow
                if api_call_duration > 10:
                    logger.warning(
                        f"[{task_name}] ⚠️  Unusually slow API call: {api_call_duration:.1f}s (>10s threshold)",
                    )

        except asyncio.CancelledError:
            # Timeout warning task cancelled, this is expected
            timeout_task.cancel()
            raise
        except Exception as e:
            # Cancel timeout warning task
            timeout_task.cancel()

            duration = time.time() - start_time
            if self.debug:
                from loguru import logger

                error_type = type(e).__name__
                error_msg = str(e)

                # Check for rate limit errors
                if (
                    "rate_limit" in error_msg.lower()
                    or "rate limit" in error_msg.lower()
                ):
                    logger.error(
                        f"[{task_name}] 🚫 RATE LIMIT ERROR after {duration:.2f}s: {error_msg}",
                    )
                elif "timeout" in error_msg.lower():
                    logger.error(
                        f"[{task_name}] ⏱️  TIMEOUT ERROR after {duration:.2f}s: {error_msg}",
                    )
                elif "connection" in error_msg.lower():
                    logger.error(
                        f"[{task_name}] 🔌 CONNECTION ERROR after {duration:.2f}s: {error_msg}",
                    )
                else:
                    logger.error(
                        f"[{task_name}] LLM call failed after {duration:.2f}s: {error_type}: {error_msg}",
                    )
            raise

        # Parse and validate response
        parse_start = time.time()
        response_data = response.model_dump()

        # Log validation and parsing time
        if self.debug:
            from loguru import logger

            parse_duration = time.time() - parse_start
            response_size = len(str(response_data))
            logger.debug(
                f"[{task_name}] Response parsed: parse_time={parse_duration:.3f}s, response_size={response_size}b, fields={list(response_data.keys())}",
            )

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
