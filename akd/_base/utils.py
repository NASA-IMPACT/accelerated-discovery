import asyncio
from abc import abstractmethod
from typing import Any, Callable, get_type_hints, overload

from pydantic import BaseModel, Field


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class AsyncRunMixin:
    """
    Mixin for adding interface to run
    async methods in a sync context.
    """

    @abstractmethod
    async def arun(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses should implement this method")

    # async def ainvoke(self, *args, **kwargs) -> Any:
    #     return await self.arun(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        """
        Runs the async method in a sync context.
        """
        if not hasattr(self, "arun"):
            raise AttributeError("Method 'arun' not implemented in the class")
        try:
            # Check if there's a running event loop
            loop = get_event_loop()
            # If we're already in an event loop, we need to use create_task and wait for it
            if loop and loop.is_running():
                # This creates a new task in the current event loop
                future = asyncio.ensure_future(self.arun(*args, **kwargs))
                return loop.run_until_complete(future)
            else:
                return asyncio.run(self.arun(*args, **kwargs))
        except RuntimeError:
            # No running event loop, create a new one
            return asyncio.run(self.arun(*args, **kwargs))


### Exposed Param Decorator and Metadata ###


class ExposedParam(BaseModel):
    """Metadata for exposed paramters."""

    description: str = Field(default="", description="Human readable description")
    extra: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Additional metadata for the parameter",
    )


@overload
def exposed_param[F: Callable](_func: F, /) -> property: ...


@overload
def exposed_param[F: Callable](
    *,
    description: str = "",
    **kwargs,
) -> Callable[[F], property]: ...


def exposed_param[F: Callable](
    _func: F | None = None,
    /,
    *,
    description: str = "",
    **kwargs,
) -> property | Callable[[F], property]:
    """
    Decorator to mark a property as exposed to external systems (UI, backend, API, etc.).

    This decorator combines @property functionality with parameter exposure metadata,
    making the parameter accessible and configurable from external systems.

    Can be used with or without parentheses:
        @exposed_param
        @exposed_param(description="...", extra_field="...")

    Args:
        _func: The function being decorated (positional-only, internal use)
        description: Human readable description of the parameter
        **kwargs: Additional metadata stored in 'extra' field (e.g., constraints, hints)
    """

    def decorator(func: F) -> property:
        # Priority: explicit description > function docstring > prettified function name
        final_description = description or (func.__doc__ or "").strip() or func.__name__.replace("_", " ").title()

        func._exposed_meta = ExposedParam(  # type: ignore[attr-defined]
            description=final_description,
            extra=kwargs or {},
        )
        return property(func)  # type: ignore[return-value]

    if _func is not None:
        return decorator(_func)

    return decorator


class ParamExposureMixin:
    """Mixin to add parameter exposure capabilities to agents."""

    def get_exposed_params(self, include_values: bool = False) -> dict[str, dict[str, Any]]:
        """
        Get metadata about exposed parameters.

        Args:
            include_values: If True, include current runtime values in the output

        Returns:
            Dict mapping parameter name to metadata containing:
                - description: Human readable description
                - type: Python type name (from type hints or runtime value)
                - editable: Whether the parameter has a setter
                - extra: Additional metadata (if present)
                - current_value: Current value (if include_values=True)
        """

        def __get_type_from_hints(attr: property) -> str:
            """Extract type from property type hints. Priority: fget return > fset param."""
            # Try fget return type
            try:
                hints = get_type_hints(attr.fget)
                if "return" in hints:
                    return getattr(hints["return"], "__name__", str(hints["return"]))
            except Exception:
                pass

            # Try fset parameter type
            if attr.fset:
                try:
                    hints = get_type_hints(attr.fset)
                    params = [k for k in hints.keys() if k != "return"]
                    if params:
                        return getattr(hints[params[0]], "__name__", str(hints[params[0]]))
                except Exception:
                    pass

            return "unknown"

        exposed = {}
        for attr_name in dir(self):
            attr = getattr(type(self), attr_name, None)
            if not (isinstance(attr, property) and hasattr(attr.fget, "_exposed_meta")):
                continue

            meta = attr.fget._exposed_meta

            # Get type from hints, fallback to runtime value
            param_type = __get_type_from_hints(attr)
            current_value = None

            if param_type == "unknown":
                try:
                    current_value = getattr(self, attr_name)
                    param_type = type(current_value).__name__
                except Exception:
                    param_type = "unknown"

            # Get current value if requested and not already fetched
            if include_values and current_value is None:
                try:
                    current_value = getattr(self, attr_name)
                except Exception:
                    current_value = None

            exposed[attr_name] = {
                "description": meta.description,
                "type": param_type,
                "editable": attr.fset is not None,
            }

            if meta.extra:
                exposed[attr_name]["extra"] = meta.extra

            if include_values:
                exposed[attr_name]["current_value"] = current_value

        return exposed
