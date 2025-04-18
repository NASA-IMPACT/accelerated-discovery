import inspect
from typing import Any, Callable, Coroutine, Dict, Optional, Union

from pydantic import BaseModel, create_model

from ..structures import CallableSpec
from ..utils import AsyncRunMixin
from ._base import BaseIOSchema, BaseTool


def tool_wrapper(func: Union[Callable[..., Any], Coroutine]) -> Any:
    """
    Converts any function or coroutine into a type of BaseTool.
    The input params are automatically converted to pydantic schema


    Example usage:
        ```python

        from akd.tools.utils import tool_wrapper

        @tool_wrapper
        def add(a: int, b: int) -> int:
            return a + b

        print(add(1, 2))
        # await add.arun(1, 2)

        tool = add.to_langchain_structured_tool()
        await too.ainvoke(input=dict(x=1, y=2))
        ```
    """

    def get_fields() -> dict:
        sig = inspect.signature(func)
        fields = {}
        for name, param in sig.parameters.items():
            field_type = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else Any
            )
            default = (
                param.default if param.default is not inspect.Parameter.empty else ...
            )
            fields[name] = (field_type, default)
        return fields

    def to_camel_case(snake_str):
        components = snake_str.split("_")
        return "".join(x.title() for x in components)

    fields = get_fields()
    return_type = func.__annotations__.get("return", Any)
    func_name = to_camel_case(func.__name__)

    input_schema = create_model(
        f"{func_name}InputSchema",
        **fields,
        __base__=BaseIOSchema,
        __doc__=f"Input Schema for {func_name}",
    )
    output_schema = create_model(
        f"{func_name}OutputSchema",
        result=(return_type, ...),
        __base__=BaseIOSchema,
        __doc__=f"Output Schema for {func_name}",
    )

    class FunctionTool(BaseTool):
        def __init__(self) -> None:
            super().__init__()
            self._func = func

        def __call__(self, *args, **kwargs) -> Any:
            """
            Synchronous execution using the wrapped function directly.
            """
            return self._func(*args, **kwargs)

        def _is_input_schema_instance(self, args, kwargs):
            """
            Check if a single argument is provided and it's already an instance of the input schema.
            """
            return (
                len(args) == 1
                and not kwargs
                and isinstance(args[0], self.__class__.input_schema)
            )

        def _convert_positional_to_keyword_args(self, args):
            """
            Convert positional arguments to keyword arguments based on the function signature.
            """
            if not args:
                return {}

            sig = inspect.signature(self._func)
            param_names = list(sig.parameters.keys())

            if len(args) > len(param_names):
                raise ValueError(
                    f"Too many positional arguments for function {func.__name__}",
                )

            return {param_names[i]: args[i] for i in range(len(args))}

        def _construct_input_schema(self, kwargs):
            """
            Create an input schema instance from keyword arguments.
            """
            try:
                return self.__class__.input_schema(**kwargs)
            except Exception as e:
                raise ValueError(f"Error constructing input schema: {e}")

        def _create_params(self, *args, **kwargs):
            """
            Create and validate input parameters for the function.
            Returns a validated input schema instance.
            """
            # Check if input is already a schema instance
            if self._is_input_schema_instance(args, kwargs):
                return args[0]

            # Convert positional args to kwargs if needed
            if args and not kwargs:
                kwargs = self._convert_positional_to_keyword_args(args)

            # Construct the input schema from kwargs
            return self._construct_input_schema(kwargs)

        async def arun(self, *args, **kwargs) -> Any:
            """
            Asynchronous execution that supports both sync and async underlying functions.
            It constructs the input model from the provided arguments, calls the function,
            and wraps the result in the dynamically-created output schema.
            """
            try:
                # Get validated parameters
                params = self._create_params(*args, **kwargs)

                # Execute the function with the parameters
                if inspect.iscoroutinefunction(self._func):
                    result = await self._func(**params.model_dump())
                else:
                    result = self._func(**params.model_dump())

                return self.__class__.output_schema(result=result)
            except Exception as e:
                raise ValueError(f"Error in function execution: {e}")

    # Assign the schemas as class attributes so they're available inside the class.
    FunctionTool.input_schema = input_schema
    FunctionTool.output_schema = output_schema

    # Create an instance of the tool.
    FunctionTool.__name__ = f"{func_name}Tool"
    tool_instance = FunctionTool()
    tool_instance.__name__ = f"{func_name}Tool"
    tool_instance.__doc__ = func.__doc__
    return tool_instance


class ToolRunner(AsyncRunMixin):
    """
    Generic mapper that binds a state dict to a
    tool's input_schema and invokes it.

    Supports BaseTool instances or raw callables (wrapped automatically).

    If the spec is a Tuple, the 2nd element is the mapping of input keys.
    Eg:
        ```python
            @tool_wrapper
            async def count_words(text: str) -> int:
                return len(text.split())

            data=dict("key1" : 123, "key2": "hello world")
            await ToolRunner().arun((count_words, {"text": "key1"}), data)
        ```
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @staticmethod
    def get_tool(spec: Union[BaseTool, Callable]) -> BaseTool:
        # Wrap callables into BaseTool via tool_wrapper
        if isinstance(spec, BaseTool):
            return spec
        return tool_wrapper(spec)

    def map_to_schema(
        self,
        tool: BaseTool,
        data: Dict[str, Any],
        mapping: Optional[Dict[str, str]] = None,
    ) -> BaseModel:
        mapping = mapping or {}
        schema = tool.input_schema
        fields = list(schema.model_fields)
        kwargs: Dict[str, Any] = {}
        # single-field fallback
        if len(fields) == 1 and len(data) == 1:
            kwargs[fields[0]] = next(iter(data.values()))
        else:
            for param in fields:
                key = mapping.get(param, param)
                if key in data:
                    kwargs[param] = data[key]
        return schema(**kwargs)

    async def arun(self, spec: CallableSpec, data: Dict[str, Any]) -> Any:
        # Unpack spec
        if isinstance(spec, tuple):
            tool, mapping = spec
        else:
            tool, mapping = spec, {}
        tool = self.get_tool(tool)
        inp = self.map_to_schema(tool, data, mapping)
        out = await tool.arun(inp)
        return getattr(out, "result", out)
