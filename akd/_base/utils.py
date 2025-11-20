import asyncio
import types
from abc import abstractmethod
from typing import Any, Callable, get_args, get_type_hints, overload

from loguru import logger
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
    """Metadata for exposed parameters."""

    name: str = Field(
        description="Parameter name",
    )
    description: str = Field(
        default="",
        description="Human readable description",
    )
    extra: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Additional metadata for the parameter",
    )


class ExposedParamRuntimeInfo(ExposedParam):
    """Complete runtime information about an exposed parameter.

    Inherits name, description, and extra from ExposedParam.
    Adds runtime-specific fields like type, editability, and current value.
    """

    type_: str = Field(
        description="Parameter type as string (e.g., 'str', 'int', 'float')",
    )
    type_source: str = Field(
        description="Source of type inference: 'fget' (getter annotation), 'fset' (setter annotation), 'runtime' (inferred from value), or 'none' (unknown)",
    )
    editable: bool = Field(
        description="Whether the parameter has a setter and can be modified",
    )
    current_value: Any | None = Field(
        default=None,
        description="Current runtime value of the parameter (only populated when include_values=True)",
    )


class ValidatedProperty(property):
    """Property subclass that automatically validates setter arguments against type hints.

    When a setter is added via the .setter() method, it's automatically wrapped
    with type validation logic that checks the value against type annotations.
    """

    def setter(self, fset):
        """Override setter to add automatic type validation."""
        original_fset = fset

        def validated_setter(instance, value):
            # Get expected type from getter or setter type hints
            expected_type = None

            # Try getter return type first
            try:
                hints = get_type_hints(self.fget)
                if "return" in hints:
                    expected_type = hints["return"]
            except Exception:
                pass

            # Try setter parameter type if getter didn't have type
            if expected_type is None:
                try:
                    hints = get_type_hints(original_fset)
                    params = [k for k in hints.keys() if k != "return"]
                    if params:
                        expected_type = hints[params[0]]
                except Exception:
                    pass

            # Validate type if we found one
            if expected_type is not None:
                # Get origin type for generics (List[int] -> list)
                origin = getattr(expected_type, "__origin__", None)

                # Check if it's a Union type (typing.Union or Python 3.10+ int | str)
                is_union = origin is not None or isinstance(expected_type, types.UnionType)

                if is_union:
                    # Handle Union types (e.g., Union[int, str] or int | str)
                    type_args = get_args(expected_type)
                    if type_args and not isinstance(value, type_args):
                        param_name = getattr(self.fget, "_exposed_meta", None)
                        name = param_name.name if param_name else "parameter"
                        type_names = ", ".join(t.__name__ if hasattr(t, "__name__") else str(t) for t in type_args)
                        raise TypeError(
                            f"Parameter '{name}' expects one of [{type_names}], got {type(value).__name__}",
                        )
                else:
                    # Simple type check
                    if not isinstance(value, expected_type):
                        param_name = getattr(self.fget, "_exposed_meta", None)
                        name = param_name.name if param_name else "parameter"
                        type_name = getattr(expected_type, "__name__", str(expected_type))
                        raise TypeError(
                            f"Parameter '{name}' expects {type_name}, got {type(value).__name__}",
                        )

            # Call original setter
            return original_fset(instance, value)

        return super().setter(validated_setter)


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
            name=func.__name__,
            description=final_description,
            extra=kwargs or {},
        )
        return ValidatedProperty(func)  # type: ignore[return-value]

    if _func is not None:
        return decorator(_func)

    return decorator


class ParamExposureMixin:
    """Mixin to add parameter exposure capabilities to agents."""

    def get_exposed_params(self, include_values: bool = False) -> list[ExposedParamRuntimeInfo]:
        """
        Get metadata about exposed parameters.

        Args:
            include_values: If True, include current runtime values in the output

        Returns:
            List of ExposedParamRuntimeInfo objects, each containing:
                - name: Parameter name
                - description: Human readable description
                - type_: Python type name (from type hints or runtime value)
                - type_source: Source of type inference
                - editable: Whether the parameter has a setter
                - extra: Additional metadata from decorator
                - current_value: Current value (if include_values=True)
        """

        def _get_type_from_hints(attr: property) -> tuple[str, str]:
            """Extract type and source from property type hints. Priority: fget return > fset param."""
            # Try fget return type
            try:
                hints = get_type_hints(attr.fget)
                if "return" in hints:
                    return getattr(hints["return"], "__name__", str(hints["return"])), "fget"
            except Exception as e:
                logger.warning(f"Failed to get type hints from fget (getter) for {attr}: {e}")

            # Try fset parameter type
            if attr.fset:
                try:
                    hints = get_type_hints(attr.fset)
                    params = [k for k in hints.keys() if k != "return"]
                    if params:
                        return getattr(hints[params[0]], "__name__", str(hints[params[0]])), "fset"
                except Exception as e:
                    logger.warning(f"Failed to get type hints from fset (setter) for {attr}: {e}")

            return "unknown", "none"

        exposed = []
        for attr_name in dir(self):
            attr = getattr(type(self), attr_name, None)
            if not (isinstance(attr, property) and hasattr(attr.fget, "_exposed_meta")):
                continue

            meta = attr.fget._exposed_meta

            # Get type from hints, fallback to runtime value
            param_type, type_source = _get_type_from_hints(attr)
            current_value = None

            if param_type == "unknown":
                try:
                    current_value = getattr(self, attr_name)
                    param_type = type(current_value).__name__
                    type_source = "runtime"
                except Exception:
                    param_type = "unknown"

            # Get current value if requested and not already fetched
            if include_values and current_value is None:
                try:
                    current_value = getattr(self, attr_name)
                except Exception:
                    current_value = None

            exposed.append(
                ExposedParamRuntimeInfo(
                    name=attr_name,
                    description=meta.description,
                    type_=param_type,
                    type_source=type_source,
                    editable=attr.fset is not None,
                    extra=meta.extra,
                    current_value=current_value if include_values else None,
                ),
            )

        return exposed
