from __future__ import annotations

import json
import uuid
from abc import abstractmethod
from typing import Any, Literal, cast

import instructor
import openai
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import AnyUrl, BaseModel, Field

from akd._base import AbstractBase, BaseConfig, InputSchema, OutputSchema
from akd.configs.project import CONFIG
from akd.configs.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    TOOL_CALL_EXAMPLES,
    TOOL_CALL_FORMAT_INSTRUCTIONS,
    TOOL_CALL_INSTRUCTIONS,
)
from akd.structures import ToolCall


class BaseAgentConfig(BaseConfig):
    """Configuration class for LangBaseAgent."""

    base_url: AnyUrl | None = Field(default=CONFIG.model_config_settings.base_url)
    api_key: str | None = Field(default=CONFIG.model_config_settings.api_keys.openai)
    model_name: str | None = Field(default=CONFIG.model_config_settings.model_name)
    temperature: float = 0.0
    system_prompt: str | None = Field(default=DEFAULT_SYSTEM_PROMPT)


class ToolCallingAgentConfig(BaseAgentConfig):
    """Configuration class for InstructorBaseAgentWithToolCalling."""

    max_iterations: int = Field(
        default=5,
        description="Maximum number of tool calling iterations before forcing final response",
    )
    min_iterations: int = Field(
        default=1,
        description="Minimum number of tool calls before allowing final response",
    )
    enforce_min_tool_calls: bool = Field(
        default=True,
        description="Whether to enforce minimum tool calls before allowing final response",
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
            temperature=self.temperature,
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


class FinalResponse[T: OutputSchema](BaseModel):
    """Represents the final response from the agent."""

    response: T = Field(..., description="The final response")


class AgentAction[T: OutputSchema](BaseModel):
    """Union type for agent actions - either tool call or final response."""

    action_type: Literal["tool_call", "final_response"] = Field(
        ...,
        description="Type of action: 'tool_call' or 'final_response'",
    )
    tool_call: ToolCall | None = Field(
        None,
        description="Tool call if action_type is 'tool_call'",
    )
    final_response: T | None = Field(
        None,
        description="Final response if action_type is 'final_response'",
    )


class InstructorBaseAgentWithToolCalling[
    InSchema: InputSchema,
    OutSchema: OutputSchema,
](InstructorBaseAgent[InSchema, OutSchema]):
    """InstructorBaseAgent with tool calling capabilities.

    This agent can execute tools during conversation by leveraging instructor's
    function calling mechanism. Tools are registered and made available to the LLM,
    which can choose to call them or provide final responses.

    Note: This is an abstract base class. Subclasses must define:
    - input_schema: Class attribute pointing to InputSchema subclass
    - output_schema: Class attribute pointing to OutputSchema subclass
    """

    config_schema = ToolCallingAgentConfig

    def __init__(
        self,
        config: ToolCallingAgentConfig | None = None,
        debug: bool = False,
        tools: list | None = None,
    ) -> None:
        super().__init__(config=config, debug=debug)

        # Store available tools
        self._tools: dict[str, Any] = {}
        if tools:
            self.bind_tools(tools)

    def bind_tools(self, tools: list) -> None:
        """Bind tools to this agent for use during conversations.

        Args:
            tools: List of BaseTool instances to make available
        """
        for tool in tools:
            tool_name = tool.__class__.__name__
            self._tools[tool_name] = tool

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool with given input parameters.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool execution result
        """
        if tool_name not in self._tools:
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {self.get_available_tools()}",
            )

        tool = self._tools[tool_name]

        # Convert dict input to tool's input schema
        tool_input_schema = tool.input_schema(**tool_input)

        # Execute the tool
        result = await tool.arun(tool_input_schema)

        # Return as dict for easier handling, ensuring AnyUrl objects are converted to strings
        if hasattr(result, "model_dump"):
            return result.model_dump(
                mode="json",
            )  # mode='json' ensures proper serialization
        else:
            return result

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:12]}"

    def _add_tool_call_to_memory(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict,
    ) -> None:
        """Add a tool call to memory using OpenAI format."""
        tool_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_input),
                    },
                },
            ],
        }
        self.memory.append(tool_call_message)

    def _add_tool_result_to_memory(
        self,
        tool_call_id: str,
        tool_result: dict,
        success: bool = True,
    ) -> None:
        """Add a tool result to memory using OpenAI format."""
        if success:
            content = json.dumps(tool_result)
        else:
            content = json.dumps({"error": str(tool_result), "success": False})

        tool_result_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        self.memory.append(tool_result_message)

    def get_tool_call_history(self) -> list[dict]:
        """Get history of tool calls from memory."""
        tool_history = []
        for message in self.memory:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    tool_history.append(
                        {
                            "tool_call_id": tool_call["id"],
                            "tool_name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"]),
                            "type": "call",
                        },
                    )
            elif message.get("role") == "tool":
                try:
                    result = json.loads(message["content"])
                except json.JSONDecodeError:
                    result = message["content"]

                tool_history.append(
                    {
                        "tool_call_id": message["tool_call_id"],
                        "result": result,
                        "type": "result",
                    },
                )
        return tool_history

    def _create_tool_description(self) -> str:
        """Create detailed tool descriptions with parameter schemas for the system prompt."""
        if not self._tools:
            return "No tools are available."

        tool_descriptions = []
        for tool_name, tool in self._tools.items():
            # Get FULL description (not truncated)
            description = getattr(
                tool,
                "description",
                tool.__class__.__doc__ or f"Tool: {tool_name}",
            )
            if description:
                description = description.strip()  # Clean whitespace but keep full text

            # Get input schema details
            input_schema = tool.input_schema
            schema_info = input_schema.model_json_schema()

            # Format parameter information with types and descriptions
            params = []
            properties = schema_info.get("properties", {})
            required_fields = schema_info.get("required", [])

            for field_name, field_info in properties.items():
                field_type = field_info.get("type", "unknown")
                field_desc = field_info.get("description", "")
                is_required = field_name in required_fields

                # Handle array types properly
                if field_type == "array":
                    items_info = field_info.get("items", {})
                    items_type = items_info.get("type", "unknown")
                    field_type = f"array of {items_type}"

                param_str = f"    - {field_name}: {field_type}"
                if not is_required:
                    param_str += " (optional)"
                if field_desc:
                    param_str += f" - {field_desc}"
                params.append(param_str)

            # Build complete tool description
            tool_desc = f"- {tool_name}: {description}\n  Parameters:"
            if params:
                tool_desc += "\n" + "\n".join(params)
            else:
                tool_desc += " None required"

            tool_descriptions.append(tool_desc)

        return "Available tools:\n" + "\n\n".join(tool_descriptions)

    @staticmethod
    def _create_tool_calling_prompt(base_prompt: str, tool_descriptions: str) -> str:
        """Create enhanced prompt with tool calling capabilities."""
        return f"""{base_prompt}

{tool_descriptions}

{TOOL_CALL_FORMAT_INSTRUCTIONS}

{TOOL_CALL_EXAMPLES}

{TOOL_CALL_INSTRUCTIONS}"""

    async def get_response_async(
        self,
        response_model: type[OutputSchema] | None = None,
        max_iterations: int | None = None,
    ) -> OutSchema:
        """
        Obtains a response from the language model with tool calling support.

        Args:
            response_model: The schema for the response data. If not set, self.output_schema is used.
            max_iterations: Maximum number of tool calling iterations. Uses config default if not provided.

        Returns:
            The final response from the language model.
        """
        response_model = response_model or self.output_schema
        max_iterations = max_iterations or self.max_iterations
        tool_calls_made = 0

        # Create enhanced system prompt with tool information
        original_system_prompt = self.system_prompt
        if self._tools:
            tool_descriptions = self._create_tool_description()
            enhanced_prompt = self._create_tool_calling_prompt(
                original_system_prompt,
                tool_descriptions,
            )
        else:
            enhanced_prompt = f"{original_system_prompt}\n\nNo tools are available. Provide your final response with action_type='final_response'."

        # Create the action model for this specific response type
        action_model = type(
            f"AgentAction_{response_model.__name__}",
            (BaseModel,),
            {
                "action_type": Field(
                    ...,
                    description="Type of action: 'tool_call' or 'final_response'",
                ),
                "tool_call": Field(
                    None,
                    description="Tool call if action_type is 'tool_call'",
                ),
                "final_response": Field(
                    None,
                    description="Final response if action_type is 'final_response'",
                ),
                "__annotations__": {
                    "action_type": str,
                    "tool_call": ToolCall | None,
                    "final_response": response_model | None,
                },
            },
        )

        # Create instructor-compatible model
        instructor_model = self._create_instructor_compatible_model(action_model)

        # Tool execution loop
        for iteration in range(max_iterations):
            # If we're on the last iteration, force a final response
            current_prompt = enhanced_prompt
            if iteration == max_iterations - 1:
                current_prompt += "\n\nIMPORTANT: This is your final chance to respond. You MUST provide action_type='final_response' with your best answer based on the information you have."

            messages = [
                {
                    "role": "system",
                    "content": current_prompt,
                },
            ] + self.memory

            # Get action from LLM
            action_response = await self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                response_model=instructor_model,
            )

            action_data = action_response.model_dump()

            if self.debug:
                logger.debug(
                    f"Iteration {iteration + 1}: Action type = {action_data.get('action_type')}",
                )

            if action_data["action_type"] == "final_response":
                # Check if we've met minimum iteration requirements
                if (
                    self.enforce_min_tool_calls
                    and tool_calls_made < self.min_iterations
                    and iteration < max_iterations - 1
                ):
                    if self.debug:
                        logger.debug(
                            f"Enforcing minimum tool calls: only {tool_calls_made} tool calls made, "
                            f"minimum is {self.min_iterations}",
                        )

                    # Add message encouraging more thorough investigation
                    self.memory.append(
                        {
                            "role": "user",
                            "content": "Please use available tools to gather more information before providing your final answer. Consider what additional tool calls might be helpful.",
                        },
                    )
                    continue

                # Return the final response
                final_response_data = action_data["final_response"]
                response = response_model(**final_response_data)
                return cast(OutSchema, response)

            elif action_data["action_type"] == "tool_call":
                # Execute the tool
                tool_call_data = action_data.get("tool_call")
                if not tool_call_data:
                    raise ValueError(
                        "tool_call field is required when action_type is 'tool_call'",
                    )

                tool_name = tool_call_data.get("tool_name")
                if not tool_name:
                    raise ValueError("tool_name is required in tool_call")

                tool_input = tool_call_data.get("tool_input", {})

                # Validate tool exists
                if tool_name not in self._tools:
                    available_tools = list(self._tools.keys())
                    raise ValueError(
                        f"Tool '{tool_name}' not found. Available tools: {available_tools}",
                    )

                # Generate unique tool call ID
                tool_call_id = self._generate_tool_call_id()

                # Add tool call to memory in OpenAI format
                self._add_tool_call_to_memory(tool_call_id, tool_name, tool_input)
                tool_calls_made += 1

                try:
                    # Execute the tool
                    tool_result = await self._execute_tool(tool_name, tool_input)

                    if self.debug:
                        logger.debug(
                            f"Tool {tool_name} executed successfully: {tool_result}",
                        )

                    # Add successful tool result to memory
                    self._add_tool_result_to_memory(
                        tool_call_id,
                        tool_result,
                        success=True,
                    )

                except Exception as e:
                    error_msg = f"Tool {tool_name} failed: {str(e)}"
                    if self.debug:
                        logger.debug(error_msg)

                    # Add failed tool result to memory
                    self._add_tool_result_to_memory(
                        tool_call_id,
                        error_msg,
                        success=False,
                    )
            else:
                raise ValueError(f"Unknown action type: {action_data['action_type']}")

        # If we reach max iterations, force a final response
        raise RuntimeError(
            f"Max iterations ({max_iterations}) reached without final response",
        )
