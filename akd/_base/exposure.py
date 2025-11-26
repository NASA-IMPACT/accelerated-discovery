import sys
import types
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import StrEnum
from typing import (
    Any,
    Callable,
    ForwardRef,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from loguru import logger
from pydantic import BaseModel

_EXPOSED_META_VAR_NAME = "_exposed_meta"


class ExposedParamTypeSource(StrEnum):
    FGET = "fget"
    FSET = "fset"
    ANNOTATED = "annotated"
    RUNTIME = "runtime"
    NONE = "none"


def get_type_from_property(prop: property) -> tuple[Any | None, ExposedParamTypeSource]:
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
            return hints["return"], ExposedParamTypeSource.FGET
    except Exception as e:
        logger.warning(f"Failed to get type hints from fget (getter) for {prop}: {e}")

    # Try setter parameter type
    if prop.fset:
        try:
            hints = get_type_hints(prop.fset)
            params = [k for k in hints.keys() if k != "return"]
            if params:
                return hints[params[0]], ExposedParamTypeSource.FSET
        except Exception as e:
            logger.warning(f"Failed to get type hints from fset (setter) for {prop}: {e}")

    return None, ExposedParamTypeSource.NONE


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
        type_hint: Type hint captured from annotations (e.g., str, int, list[str])
    """

    name: str = ""
    description: str = ""
    expose: bool = True
    persistent: bool = True
    type_hint: Any = None
    type_source: ExposedParamTypeSource | None = None
    extra: dict[str, Any] = dc_field(default_factory=dict)


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
    editable: bool = False
    current_value: Any | None = None


# Helper functions for Annotated-based exposure


def Exposed(
    name: str | None = None,
    description: str | None = None,
    expose: bool = True,
    **extra,
) -> ExposedParam:
    """Create metadata for persistent, exposed config parameters.

    Use this in Annotated type hints to mark config fields that should be:
    - Exposed as properties on the class
    - Externally controllable (via UI, API, etc.)

    Args:
        name: Explicit parameter name (defaults to field name if not provided)
        description: Human readable description of the parameter
        **extra: Additional metadata stored in the 'extra' field

    Returns:
        ExposedParam instance configured for persistent, exposed parameters

    Example:
        ```python
        class MyConfig(BaseModel):
            temperature: Annotated[float, Exposed(
                description="LLM temperature",
            )] = 0.7
        ```
    """
    return ExposedParam(
        name=name or "",
        description=description or "",
        expose=expose,
        persistent=True,
        extra=extra,
    )


def Validated(
    name: str | None = None,
    description: str | None = None,
    **extra,
) -> ExposedParam:
    """Create metadata for transient, validated parameters (input/output schemas).

    Use this for input/output schema fields that should be:
    - NOT exposed as instance properties (transient data)

    Args:
        name: Explicit parameter name (defaults to field name if not provided)
        description: Human readable description of the parameter
        **extra: Additional metadata stored in the 'extra' field

    Returns:
        ExposedParam instance configured for transient, validated parameters

    Example:
        ```python
        class MyInput(BaseModel):
            query: Annotated[str, Validated(
                description="Search query",
            )]
        ```
    """
    return ExposedParam(
        name=name or "",
        description=description or "",
        expose=False,
        persistent=False,
        extra=extra,
    )


def ReadOnly(name: str | None = None, description: str | None = None, expose: bool = True, **extra) -> ExposedParam:
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
        expose=expose,
        persistent=True,
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
                        param_name = getattr(self.fget, _EXPOSED_META_VAR_NAME, None)
                        name = param_name.name if param_name else "parameter"
                        type_names = ", ".join(t.__name__ if hasattr(t, "__name__") else str(t) for t in type_args)
                        raise TypeError(
                            f"Parameter '{name}' expects one of [{type_names}], got {type(value).__name__}",
                        )
                else:
                    # Simple type check
                    if not isinstance(value, expected_type):
                        param_name = getattr(self.fget, _EXPOSED_META_VAR_NAME, None)
                        name = param_name.name if param_name else "parameter"
                        type_name = getattr(expected_type, "__name__", str(expected_type))
                        raise TypeError(
                            f"Parameter '{name}' expects {type_name}, got {type(value).__name__}",
                        )

            # Call original setter
            return original_fset(instance, value)

        # preserve original annotations
        validated_setter.__annotations__ = original_fset.__annotations__.copy()
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
    name: str | None = None,
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

        # Extract type hint from getter return annotation
        type_hint = None
        try:
            hints = get_type_hints(func)
            if "return" in hints:
                type_hint = hints["return"]
        except Exception:
            pass

        setattr(
            func,
            _EXPOSED_META_VAR_NAME,
            ExposedParam(  # type: ignore[attr-defined]
                name=name or func.__name__,
                description=final_description,
                extra=kwargs or {},
                type_hint=type_hint,
                type_source=ExposedParamTypeSource.FGET if type_hint is not None else None,
            ),
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
            # Skip generic type parameters (InSchema, OutSchema, etc.)
            if field_name in ("InSchema", "OutSchema"):
                continue

            # Check if this has args (Annotated types have args)
            args = get_args(type_hint)
            if debug:
                logger.debug(f"  Field {field_name}: type={type_hint}, args={args}")

            if args:
                # Look for ExposedParam in the metadata (args after the first one)
                for i, arg in enumerate(args[1:], 1):  # Skip first arg (the actual type)
                    if debug:
                        logger.debug(f"\tArg[{i}]: {arg} (type: {type(arg).__name__})")
                    if isinstance(arg, ExposedParam):
                        if debug:
                            logger.debug(f"\tFound ExposedParam for {field_name}")

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
                                    type_hint=args[0],  # Capture type hint
                                    type_source=ExposedParamTypeSource.ANNOTATED,
                                )

                        # Ensure type_hint is set even if we didn't recreate the object
                        if arg.type_hint is None:
                            arg.type_hint = args[0]
                            arg.type_source = ExposedParamTypeSource.ANNOTATED

                        result[field_name] = (args[0], arg)  # (actual_type, metadata)
                        break
    except NameError as e:
        # Skip classes with unresolved generic type parameters
        if debug:
            logger.debug(f"Skipping {schema_class.__name__} due to generic types: {e}")
    except Exception as e:
        if debug:
            logger.debug(f"Could not extract type hints from {schema_class}: {e}")

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
                type_hint=field_type,  # Capture type hint
                type_source=ExposedParamTypeSource.ANNOTATED,
            )

        result[flat_name] = (path_parts, metadata, field_type)

    # Build dict of all attributes to scan (from type hints + class attributes)
    attrs_to_scan = {}

    # Try to get type-annotated attributes
    try:
        hints = get_type_hints(cls, include_extras=True)
        # Filter out generic type parameters
        attrs_to_scan = {k: v for k, v in hints.items() if k not in ("InSchema", "OutSchema")}
        if debug:
            logger.debug(f"Found {len(attrs_to_scan)} type hints in {cls.__name__}")
    except NameError as e:
        # Skip classes with unresolved generic type parameters
        if debug:
            logger.debug(f"Skipping type hints for {cls.__name__} due to generic types: {e}")
    except Exception as e:
        if debug:
            logger.debug(f"Could not get type hints for {cls.__name__}: {e}")

    # Check for schema class attributes and add them to scan
    # AbstractBase.__init__ creates self.config from config_schema
    if hasattr(cls, "config_schema"):
        config_schema = getattr(cls, "config_schema")
        if isinstance(config_schema, type):
            attrs_to_scan["config"] = config_schema
            if debug:
                logger.debug(f"Adding config_schema as 'config' attribute: {config_schema.__name__}")

    # Now scan all discovered attributes for nested objects
    for attr_name, type_hint in attrs_to_scan.items():
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
                        type_hint=field_type,  # Capture type hint
                        type_source=ExposedParamTypeSource.ANNOTATED,
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
                debug=debug,
            )
            result.update(nested_result)

    return result


def _extract_exposed_from_annotations(
    source_class: type,
    skip_fields: set[str] | None = None,
) -> dict[str, ExposedParam]:
    """Extract Exposed() markers from class annotations for registry storage.

    This function extracts ExposedParam metadata from Annotated type hints.
    Works for both config schemas (BaseModel) and regular classes.

    Args:
        source_class: Class to extract from (can be BaseModel or any class)
        skip_fields: Optional set of field names to skip (e.g., schema attributes)

    Returns:
        Dict mapping field_name -> ExposedParam with type_hint and description resolved
    """
    exposed_fields = {}
    skip_fields = skip_fields or set()

    try:
        # Use __annotations__ directly to get only THIS class's annotations (not inherited)
        # This avoids issues with base class type annotations that we don't need
        raw_annotations = getattr(source_class, "__annotations__", {})

        if not raw_annotations:
            return exposed_fields

        # Get module namespace for resolving forward references in THIS class's annotations
        module = sys.modules.get(source_class.__module__)
        globalns = vars(module) if module else {}

        # Manually evaluate annotations with proper namespace
        hints = {}
        for name, annotation in raw_annotations.items():
            try:
                # Use eval_str=True in Python 3.10+ to handle stringified annotations
                if isinstance(annotation, str):
                    annotation = ForwardRef(annotation)
                if isinstance(annotation, ForwardRef):
                    annotation = annotation._evaluate(globalns, globalns, frozenset())
                hints[name] = annotation
            except:  # noqa: E722
                # If evaluation fails, use the raw annotation
                hints[name] = annotation

        for field_name, type_hint in hints.items():
            # Skip specified fields
            if field_name in skip_fields:
                continue

            # Check if this has args (Annotated types have args)
            origin = get_origin(type_hint)
            if origin is not None:
                args = get_args(type_hint)
                if not args:
                    continue

                # Look for ExposedParam in the metadata
                for metadata in args[1:]:  # Skip first arg (the actual type)
                    if isinstance(metadata, ExposedParam):
                        # Resolve description priority:
                        # 1. Exposed(description=...)
                        # 2. Field(description=...) - works for both BaseModel and class attributes
                        # 3. Prettified field name
                        final_description = metadata.description

                        if not final_description:
                            if hasattr(source_class, "model_fields"):
                                field_info = source_class.model_fields.get(field_name)
                                if field_info and field_info.description:
                                    final_description = field_info.description

                        if not final_description:
                            final_description = field_name.replace("_", " ").title()

                        # Extract actual type (first arg in Annotated[T, ...])
                        actual_type = args[0]

                        # Create enriched metadata with resolved description and type
                        enriched = ExposedParam(
                            name=metadata.name or field_name,
                            description=final_description,
                            extra=metadata.extra,
                            expose=metadata.expose,
                            persistent=metadata.persistent,
                            type_hint=actual_type,
                            type_source=ExposedParamTypeSource.ANNOTATED,
                        )

                        exposed_fields[field_name] = enriched
                        break

    except NameError as e:
        # Log classes with unresolved type references for debugging
        logger.debug(
            f"Could not extract exposed params from {source_class.__name__}: unresolved type reference - {e}",
        )
    except Exception as e:
        # Log other unexpected errors at debug level
        logger.debug(f"Could not extract exposed params from {source_class.__name__}: {e}")

    return exposed_fields


def _create_property(
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

    def make_setter(path: list[str]):
        def setter(self, value):
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
        # Read-write property
        setter = make_setter(path_parts)
        prop = ValidatedProperty(getter)
        prop = prop.setter(setter)

    # Attach metadata
    if metadata.type_hint is None:
        metadata.type_hint = type_hint
    setattr(prop.fget, _EXPOSED_META_VAR_NAME, metadata)

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

        Also builds the _exposure_registry for config-level fields.
        """
        super().__init_subclass__(**kwargs)

        # Initialize registry
        if not hasattr(cls, "_exposure_registry"):
            cls._exposure_registry = {}

        # NEW: Extract and store metadata from config_schema
        if hasattr(cls, "config_schema") and cls.config_schema is not None:
            config_exposed = _extract_exposed_from_annotations(cls.config_schema)
            # Add config_ prefix to field names for registry keys (full path naming)
            for field_name, metadata in config_exposed.items():
                registry_key = f"config_{field_name}"
                cls._exposure_registry[registry_key] = metadata

        # NEW: Extract and store class-level annotated fields
        # Skip schema attributes to avoid conflicts
        skip_fields = {"InSchema", "OutSchema", "input_schema", "output_schema", "config_schema"}
        class_exposed = _extract_exposed_from_annotations(cls, skip_fields=skip_fields)
        # No prefix for class-level fields
        cls._exposure_registry.update(class_exposed)

        # Scan for annotated fields (including nested components)
        discovered = _scan_annotated_fields_recursive(cls)

        # Create properties for fields that should be exposed
        for flat_name, (path_parts, metadata, type_hint) in discovered.items():
            # Only create properties for persistent, exposed fields
            if metadata.expose and metadata.persistent:
                # Check if it's already a property with _exposed_meta (from @exposed_param decorator)
                existing_attr = getattr(cls, flat_name, None)
                is_exposed_property = isinstance(existing_attr, property) and hasattr(
                    existing_attr.fget,
                    _EXPOSED_META_VAR_NAME,
                )

                if not is_exposed_property:
                    # Create/overwrite with property (overwrites class attributes)
                    _create_property(cls, flat_name, path_parts, metadata, type_hint)
                else:
                    logger.debug(
                        f"Skipping auto-creation of property '{flat_name}' on {cls.__name__} - already exposed via decorator",
                    )

    def get_exposed_params(
        self,
        include_values: bool = False,
        filter_by: Literal["exposed", "persistent", "all"] = "exposed",
    ) -> list[ExposedParamRuntimeInfo]:
        """
        Get metadata about exposed parameters from decorator, annotation, and registry sources.

        This method discovers parameters exposed via:
        1. @exposed_param decorator on properties (highest priority)
        2. Annotated[T, Exposed()] fields in schemas (auto-created properties)
        3. Registry entries for config-level Annotated fields

        Priority resolution:
        - Decorator metadata (@exposed_param) takes precedence over registry
        - Registry provides metadata for fields that may not have properties yet

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
        exposed_names = set()

        # STEP 1: Collect from properties with @exposed_param decorator
        for attr_name in dir(self):
            attr = getattr(type(self), attr_name, None)
            if not (isinstance(attr, property) and hasattr(attr.fget, _EXPOSED_META_VAR_NAME)):
                continue

            meta = getattr(attr.fget, _EXPOSED_META_VAR_NAME)

            # Get type from metadata if available, otherwise try property inspection
            type_hint = getattr(meta, "type_hint", None)
            type_source = getattr(meta, "type_source", ExposedParamTypeSource.NONE)

            # If still None, try fget (legacy/fallback)
            if type_hint is None:
                t, s = get_type_from_property(attr)
                if t is not None:
                    type_hint = t
                    type_source = s

            current_value = None

            # Convert type hint to string name
            if type_hint is not None:
                param_type = getattr(type_hint, "__name__", str(type_hint))
                if type_source is ExposedParamTypeSource.NONE:
                    type_source = ExposedParamTypeSource.ANNOTATED
            else:
                param_type = "unknown"
                type_source = ExposedParamTypeSource.NONE

            # Fallback to runtime value if no type hint
            if param_type == "unknown":
                try:
                    current_value = getattr(self, attr_name)
                    param_type = type(current_value).__name__
                    type_source = ExposedParamTypeSource.RUNTIME
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
                    extra=meta.extra,
                    expose=meta.expose,
                    persistent=meta.persistent,
                    type_hint=getattr(meta, "type_hint", None),
                    type_=param_type,
                    type_source=type_source,
                    editable=attr.fset is not None,
                    current_value=current_value if include_values else None,
                ),
            )
            exposed_names.add(attr_name)

        # STEP 2: Consult registry for config-level fields
        registry = getattr(type(self), "_exposure_registry", {})
        for registry_key, metadata in registry.items():
            # Skip if already exposed via decorator (decorator has priority)
            if registry_key in exposed_names:
                continue

            # Apply filter
            if filter_by == "exposed" and not metadata.expose:
                continue
            elif filter_by == "persistent" and not metadata.persistent:
                continue

            # Check if a property was created for this registry entry
            attr = getattr(type(self), registry_key, None)
            if isinstance(attr, property):
                # Property exists, get type from it
                type_hint, type_source = get_type_from_property(attr)
                param_type = getattr(type_hint, "__name__", str(type_hint)) if type_hint else "unknown"
                editable = attr.fset is not None
            else:
                # No property - use type from registry metadata
                editable = metadata.extra.get("editable", True)

                # Use type_hint from registry (captured at compile time)
                if metadata.type_hint is not None:
                    param_type = getattr(metadata.type_hint, "__name__", str(metadata.type_hint))
                    type_source = ExposedParamTypeSource.ANNOTATED
                else:
                    # Fallback to runtime value if type not in metadata
                    param_type = "unknown"
                    type_source = ExposedParamTypeSource.NONE
                    try:
                        current_value = getattr(self, registry_key)
                        param_type = type(current_value).__name__
                        type_source = ExposedParamTypeSource.RUNTIME
                    except AttributeError:
                        pass

            # Get current value if requested
            current_value = None
            if include_values:
                try:
                    current_value = getattr(self, registry_key)
                except AttributeError:
                    pass

            exposed.append(
                ExposedParamRuntimeInfo(
                    name=metadata.name or registry_key,
                    description=metadata.description,
                    extra=metadata.extra,
                    expose=metadata.expose,
                    persistent=metadata.persistent,
                    type_hint=metadata.type_hint,
                    type_=param_type,
                    type_source=type_source,
                    editable=editable,
                    current_value=current_value if include_values else None,
                ),
            )
            exposed_names.add(registry_key)

        # STEP 3: Scan for direct Annotated fields (not in registry or properties)
        discovered = _scan_annotated_fields_recursive(type(self))
        for flat_name, (path_parts, metadata, type_hint) in discovered.items():
            if flat_name in exposed_names:
                continue  # Already added as property or from registry

            # Apply filter
            if filter_by == "exposed" and not metadata.expose:
                continue
            elif filter_by == "persistent" and not metadata.persistent:
                continue

            # Get current value
            current_value = None
            if include_values:
                try:
                    current_value = getattr(self, flat_name)
                except AttributeError:
                    pass

            # Get type name
            param_type = getattr(type_hint, "__name__", str(type_hint))
            is_editable = metadata.extra.get("editable", True)

            exposed.append(
                ExposedParamRuntimeInfo(
                    name=metadata.name or flat_name,
                    description=metadata.description,
                    extra=metadata.extra,
                    expose=metadata.expose,
                    persistent=metadata.persistent,
                    type_hint=type_hint,
                    type_=param_type,
                    type_source=ExposedParamTypeSource.ANNOTATED,
                    editable=is_editable,
                    current_value=current_value,
                ),
            )

        return exposed
