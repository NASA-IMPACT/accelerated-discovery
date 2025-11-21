import types
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import (
    Any,
    Callable,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from loguru import logger
from pydantic import BaseModel


def get_type_from_property(prop: property) -> tuple[Any | None, str]:
    """Extract type annotation from property.

    Tries getter return type first, then setter parameter type.

    Args:
        prop: Property to extract type from

    Returns:
        Tuple of (type_hint, source) where:
        - type_hint: The actual type annotation object (e.g., int, str, Union[int, str])
        - source: 'fget' (from getter), 'fset' (from setter), or 'none' (not found)
    """
    # Try getter return type first
    try:
        hints = get_type_hints(prop.fget)
        if "return" in hints:
            return hints["return"], "fget"
    except Exception as e:
        logger.warning(f"Failed to get type hints from fget (getter) for {prop}: {e}")

    # Try setter parameter type
    if prop.fset:
        try:
            hints = get_type_hints(prop.fset)
            params = [k for k in hints.keys() if k != "return"]
            if params:
                return hints[params[0]], "fset"
        except Exception as e:
            logger.warning(f"Failed to get type hints from fset (setter) for {prop}: {e}")

    return None, "none"


@dataclass
class ExposedParam:
    """Metadata for exposed parameters.

    This model supports both decorator-based (@exposed_param) and annotation-based
    (Annotated[T, Exposed()]) parameter exposure patterns.

    Attributes:
        name: Parameter name
        description: Human readable description
        extra: Additional metadata for the parameter
        expose: Whether to create a property for external control
        persistent: Whether this is persistent state (True) or transient data (False)
        guardrails: List of guardrail validators to run when value is set
    """

    name: str = ""
    description: str = ""
    extra: dict[str, Any] = dc_field(default_factory=dict)
    expose: bool = True
    persistent: bool = True
    guardrails: list[Any] = dc_field(default_factory=list)


@dataclass
class ExposedParamRuntimeInfo(ExposedParam):
    """Complete runtime information about an exposed parameter.

    Inherits name, description, and extra from ExposedParam.
    Adds runtime-specific fields like type, editability, and current value.

    Attributes:
        type_: Parameter type as string (e.g., 'str', 'int', 'float')
        type_source: Source of type inference: 'fget' (getter annotation), 'fset' (setter annotation), 'runtime' (inferred from value), or 'none' (unknown)
        editable: Whether the parameter has a setter and can be modified
        current_value: Current runtime value of the parameter (only populated when include_values=True)
    """

    type_: str = "unknown"
    type_source: str = "none"
    editable: bool = False
    current_value: Any | None = None


# Helper functions for Annotated-based exposure


def Exposed(
    name: str | None = None,
    description: str | None = None,
    guardrails: list | None = None,
    **extra,
) -> ExposedParam:
    """Create metadata for persistent, exposed config parameters.

    Use this in Annotated type hints to mark config fields that should be:
    - Exposed as properties on the class
    - Externally controllable (via UI, API, etc.)
    - Validated with optional guardrails

    Args:
        name: Explicit parameter name (defaults to field name if not provided)
        description: Human readable description of the parameter
        guardrails: List of guardrail validators to run when value is set
        **extra: Additional metadata stored in the 'extra' field

    Returns:
        ExposedParam instance configured for persistent, exposed parameters

    Example:
        ```python
        class MyConfig(BaseModel):
            temperature: Annotated[float, Exposed(
                description="LLM temperature",
                guardrails=[RangeGuardrail(0.0, 2.0)]
            )] = 0.7
        ```
    """
    return ExposedParam(
        name=name or "",
        description=description or "",
        expose=True,
        persistent=True,
        guardrails=guardrails or [],
        extra=extra,
    )


def Validated(
    name: str | None = None,
    description: str | None = None,
    guardrails: list | None = None,
    **extra,
) -> ExposedParam:
    """Create metadata for transient, validated parameters (input/output schemas).

    Use this for input/output schema fields that should be:
    - Validated with guardrails during schema validation
    - NOT exposed as instance properties (transient data)

    Args:
        name: Explicit parameter name (defaults to field name if not provided)
        description: Human readable description of the parameter
        guardrails: List of guardrail validators to run during validation
        **extra: Additional metadata stored in the 'extra' field

    Returns:
        ExposedParam instance configured for transient, validated parameters

    Example:
        ```python
        class MyInput(BaseModel):
            query: Annotated[str, Validated(
                description="Search query",
                guardrails=[LengthGuardrail(max=500)]
            )]
        ```
    """
    return ExposedParam(
        name=name or "",
        description=description or "",
        expose=False,
        persistent=False,
        guardrails=guardrails or [],
        extra=extra,
    )


def ReadOnly(name: str | None = None, description: str | None = None, **extra) -> ExposedParam:
    """Create metadata for read-only exposed parameters.

    Use this for config fields that should be:
    - Exposed as properties (readable)
    - NOT editable (no setter created)

    Args:
        name: Explicit parameter name (defaults to field name if not provided)
        description: Human readable description of the parameter
        **extra: Additional metadata stored in the 'extra' field

    Returns:
        ExposedParam instance configured for read-only parameters

    Example:
        ```python
        class MyConfig(BaseModel):
            version: Annotated[str, ReadOnly(description="Version info")] = "1.0.0"
        ```
    """
    extra_with_editable = {"editable": False, **extra}
    return ExposedParam(
        name=name or "",
        description=description or "",
        expose=True,
        persistent=True,
        guardrails=[],
        extra=extra_with_editable,
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
            # Get expected type from property getter first
            expected_type, _ = get_type_from_property(self)

            # If no type from getter, try the new setter being added
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


# Helper functions for Annotated-based parameter scanning and property creation


def _extract_exposed_params_from_schema(schema_class: type, debug: bool = False) -> dict[str, tuple[Any, ExposedParam]]:
    """Extract Annotated[T, ExposedParam] fields from a schema class.

    Args:
        schema_class: A class (typically Pydantic BaseModel) to scan for annotated fields

    Returns:
        Dict mapping field names to (type_hint, ExposedParam) tuples
    """
    result = {}
    try:
        # Get type hints with Annotated metadata
        hints = get_type_hints(schema_class, include_extras=True)
        if debug:
            logger.debug(f"Scanning {schema_class.__name__}: found {len(hints)} type hints")

        # Try to get Pydantic v2 field info if available
        pydantic_fields = {}
        try:
            if issubclass(schema_class, BaseModel):
                pydantic_fields = schema_class.model_fields
        except (TypeError, AttributeError):
            pass

        for field_name, type_hint in hints.items():
            # Check if this has args (Annotated types have args)
            args = get_args(type_hint)
            if debug:
                logger.debug(f"  Field {field_name}: type={type_hint}, args={args}")

            if args:
                # Look for ExposedParam in the metadata (args after the first one)
                for i, arg in enumerate(args[1:], 1):  # Skip first arg (the actual type)
                    if debug:
                        logger.debug(f"    Arg[{i}]: {arg} (type: {type(arg).__name__})")
                    if isinstance(arg, ExposedParam):
                        if debug:
                            logger.debug(f"    ✓ Found ExposedParam for {field_name}")

                        # If description is empty, try to get it from Pydantic Field
                        if not arg.description and field_name in pydantic_fields:
                            field_info = pydantic_fields[field_name]
                            pydantic_description = getattr(field_info, "description", None)
                            if pydantic_description:
                                # Create new ExposedParam with Pydantic description
                                arg = ExposedParam(
                                    name=arg.name,
                                    description=pydantic_description,
                                    extra=arg.extra,
                                    expose=arg.expose,
                                    persistent=arg.persistent,
                                    guardrails=arg.guardrails,
                                )

                        result[field_name] = (args[0], arg)  # (actual_type, metadata)
                        break
    except Exception as e:
        logger.warning(f"Could not extract type hints from {schema_class}: {e}")

    return result


def _scan_annotated_fields_recursive(
    cls: type,
    prefix: str = "",
    current_depth: int = 0,
    max_depth: int = 5,
    visited: set[type] | None = None,
    debug: bool = False,
) -> dict[str, tuple[list[str], ExposedParam, Any]]:
    """Recursively scan class for Annotated[T, ExposedParam] fields.

    Scans all type-annotated attributes and recursively descends into
    nested types to find Annotated fields with ExposedParam metadata.

    Args:
        cls: Class to scan
        prefix: Current path prefix (for nested components)
        current_depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops
        visited: Set of already-visited types to prevent circular references

    Returns:
        Dict mapping flattened names to (path_parts, ExposedParam, type_hint) tuples
        Example: {
            "config_temperature": (
                ["config", "temperature"],
                ExposedParam(...),
                float
            )
        }
    """
    if visited is None:
        visited = set()

    if current_depth >= max_depth or cls in visited:
        return {}

    visited.add(cls)
    result = {}

    # First, extract Annotated fields directly from this class
    direct_fields = _extract_exposed_params_from_schema(cls, debug=debug)
    for field_name, (field_type, metadata) in direct_fields.items():
        # Build the path (just the field name for direct fields)
        if prefix:
            path_parts = prefix.split("_") + [field_name]
            flat_name = "_".join(path_parts)
        else:
            path_parts = [field_name]
            flat_name = field_name

        # Store with name filled in if not set
        if not metadata.name:
            metadata = ExposedParam(
                name=flat_name,
                description=metadata.description,
                extra=metadata.extra,
                expose=metadata.expose,
                persistent=metadata.persistent,
                guardrails=metadata.guardrails,
            )

        result[flat_name] = (path_parts, metadata, field_type)

    # Then scan type-annotated attributes for nested objects
    try:
        hints = get_type_hints(cls, include_extras=True)

        for attr_name, type_hint in hints.items():
            # Skip private attributes
            if attr_name.startswith("_"):
                continue

            # Check if the type itself has Annotated fields (e.g., Pydantic models)
            origin = get_origin(type_hint)
            actual_type = type_hint

            # Unwrap Annotated if present
            if origin is not None:
                args = get_args(type_hint)
                if args:
                    actual_type = args[0]

            # If the attribute's type is a class, scan it for Annotated fields
            if isinstance(actual_type, type):
                exposed_fields = _extract_exposed_params_from_schema(actual_type, debug=debug)

                for field_name, (field_type, metadata) in exposed_fields.items():
                    # Build the path
                    if prefix:
                        path_parts = prefix.split("_") + [attr_name, field_name]
                    else:
                        path_parts = [attr_name, field_name]

                    # Create flattened name
                    flat_name = "_".join(path_parts)

                    # Store with name filled in if not set
                    if not metadata.name:
                        metadata = ExposedParam(
                            name=flat_name,
                            description=metadata.description,
                            extra=metadata.extra,
                            expose=metadata.expose,
                            persistent=metadata.persistent,
                            guardrails=metadata.guardrails,
                        )

                    result[flat_name] = (path_parts, metadata, field_type)

                # Recursively scan the nested type
                new_prefix = f"{prefix}_{attr_name}" if prefix else attr_name
                nested_result = _scan_annotated_fields_recursive(
                    actual_type,
                    prefix=new_prefix,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    visited=visited,
                )
                result.update(nested_result)

    except Exception as e:
        logger.debug(f"Could not scan annotations for {cls}: {e}")

    return result


def _create_guarded_property(
    cls: type,
    flat_name: str,
    path_parts: list[str],
    metadata: ExposedParam,
    type_hint: Any,
) -> None:
    """Create a property with getter/setter on the class that traverses a nested path.

    Args:
        cls: Class to add the property to
        flat_name: Flattened property name (e.g., "component_config_temperature")
        path_parts: Path to traverse (e.g., ["component", "config", "temperature"])
        metadata: ExposedParam metadata
        type_hint: Expected type for validation
    """

    def make_getter(path: list[str]):
        def getter(self):
            obj = self
            try:
                for part in path:
                    obj = getattr(obj, part)
                return obj
            except AttributeError as e:
                logger.warning(f"Could not access {'.'.join(path)}: {e}")
                return None

        return getter

    def make_setter(path: list[str], guardrails: list):
        def setter(self, value):
            # Run guardrails first
            for guardrail in guardrails:
                if hasattr(guardrail, "validate"):
                    guardrail.validate(value)
                elif callable(guardrail):
                    guardrail(value)

            # Navigate to parent object and set the final attribute
            obj = self
            try:
                for part in path[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, path[-1], value)
            except AttributeError as e:
                logger.warning(f"Could not set {'.'.join(path)}: {e}")

        return setter

    # Create the property
    getter = make_getter(path_parts)

    # Check if it should be read-only
    is_read_only = metadata.extra.get("editable") is False

    if is_read_only:
        # Read-only property
        prop = property(getter)
    else:
        # Read-write property with guardrails
        setter = make_setter(path_parts, metadata.guardrails)
        prop = ValidatedProperty(getter)
        prop = prop.setter(setter)

    # Attach metadata
    prop.fget._exposed_meta = metadata  # type: ignore[attr-defined]

    # Set the property on the class
    setattr(cls, flat_name, prop)


class ParamExposureMixin:
    """Mixin to add parameter exposure capabilities to agents.

    This mixin provides two ways to expose parameters:
    1. Decorator-based: Use @exposed_param on properties
    2. Annotation-based: Use Annotated[T, Exposed()] in schema fields

    The annotation-based approach automatically creates properties for nested
    component fields at class creation time via __init_subclass__.
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically discover and create properties for Annotated fields.

        This hook is called when a class inherits from ParamExposureMixin.
        It scans the class for Annotated[T, ExposedParam] fields and creates
        properties with getters/setters that traverse nested paths.
        """
        super().__init_subclass__(**kwargs)

        # Scan for annotated fields
        discovered = _scan_annotated_fields_recursive(cls)

        # Create properties for fields that should be exposed
        for flat_name, (path_parts, metadata, type_hint) in discovered.items():
            # Only create properties for persistent, exposed fields
            if metadata.expose and metadata.persistent:
                # Check if property already exists (avoid overwriting @exposed_param)
                if not hasattr(cls, flat_name):
                    _create_guarded_property(cls, flat_name, path_parts, metadata, type_hint)
                else:
                    logger.debug(
                        f"Skipping auto-creation of property '{flat_name}' on {cls.__name__} - already exists",
                    )

    def get_exposed_params(
        self,
        include_values: bool = False,
        filter_by: Literal["exposed", "persistent", "all"] = "exposed",
    ) -> list[ExposedParamRuntimeInfo]:
        """
        Get metadata about exposed parameters from both decorator and annotation-based sources.

        This method discovers parameters exposed via:
        1. @exposed_param decorator on properties
        2. Annotated[T, Exposed()] fields in schemas (auto-created properties)

        Args:
            include_values: If True, include current runtime values in the output
            filter_by: Filter which parameters to return:
                - "exposed": Only externally controllable params (expose=True, default)
                - "persistent": Only persistent state params (persistent=True)
                - "all": All parameters with ExposedParam metadata

        Returns:
            List of ExposedParamRuntimeInfo objects, each containing:
                - name: Parameter name
                - description: Human readable description
                - type_: Python type name (from type hints or runtime value)
                - type_source: Source of type inference
                - editable: Whether the parameter has a setter
                - extra: Additional metadata (including expose, persistent, guardrails)
                - current_value: Current value (if include_values=True)
        """
        exposed = []
        for attr_name in dir(self):
            attr = getattr(type(self), attr_name, None)
            if not (isinstance(attr, property) and hasattr(attr.fget, "_exposed_meta")):
                continue

            meta = attr.fget._exposed_meta

            # Apply filter
            if filter_by == "exposed" and not meta.expose:
                continue
            elif filter_by == "persistent" and not meta.persistent:
                continue
            # "all" includes everything

            # Get type from hints using shared utility
            type_hint, type_source = get_type_from_property(attr)
            current_value = None

            # Convert type hint to string name
            if type_hint is not None:
                param_type = getattr(type_hint, "__name__", str(type_hint))
            else:
                param_type = "unknown"

            # Fallback to runtime value if no type hint
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
