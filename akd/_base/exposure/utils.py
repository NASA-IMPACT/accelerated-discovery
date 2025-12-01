"""Helper utilities for parameter exposure system."""

import sys
from typing import Any, ForwardRef, get_args, get_origin, get_type_hints

from loguru import logger
from pydantic import BaseModel

from .structures import (
    _EXPOSED_META_VAR_NAME,
    ExposedParam,
    ExposedParamRuntimeInfo,
    ExposedParamTypeSource,
)


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


def _extract_annotated_fields(
    source_class: type,
    skip_fields: set[str] | None = None,
    debug: bool = False,
) -> dict[str, ExposedParam]:
    """Extract Annotated[T, ExposedParam] fields from a class.

    Unified method that combines the functionality of the previous
    _extract_exposed_params_from_schema() and _extract_exposed_from_annotations().

    Uses __annotations__ directly to avoid NameError issues with generic type parameters,
    with proper handling of:
    - Forward references (manual resolution)
    - Pydantic Field description fallback
    - Optional field filtering

    Args:
        source_class: Class to extract from (BaseModel, config schema, or any class)
        skip_fields: Optional set of field names to skip
        debug: Enable debug logging

    Returns:
        Dict mapping field names to ExposedParam (with type_hint populated)
    """
    result = {}
    skip_fields = skip_fields or set()

    try:
        # Use __annotations__ directly to avoid NameError with generic type parameters
        raw_annotations = getattr(source_class, "__annotations__", {})

        if not raw_annotations:
            if debug:
                logger.debug(f"{source_class.__name__} has no annotations")
            return result

        if debug:
            logger.debug(f"Scanning {source_class.__name__}: found {len(raw_annotations)} annotations")

        # Get module namespace for resolving forward references
        module = sys.modules.get(source_class.__module__)
        globalns = vars(module) if module else {}

        # Manually evaluate annotations with proper namespace
        hints = {}
        for name, annotation in raw_annotations.items():
            try:
                # Handle stringified annotations and forward references
                if isinstance(annotation, str):
                    annotation = ForwardRef(annotation)
                if isinstance(annotation, ForwardRef):
                    # Python 3.11+ has recursive_guard with default=frozenset()
                    annotation = annotation._evaluate(globalns, globalns)
                hints[name] = annotation
            except Exception:
                # If evaluation fails, use the raw annotation
                hints[name] = annotation

        # Try to get Pydantic v2 field info if available for description fallback
        pydantic_fields = {}
        try:
            if issubclass(source_class, BaseModel):
                pydantic_fields = source_class.model_fields
        except (TypeError, AttributeError):
            pass

        for field_name, type_hint in hints.items():
            # Skip specified fields
            if field_name in skip_fields:
                if debug:
                    logger.debug(f"  Skipping {field_name} (in skip_fields)")
                continue

            # Check if this has args (Annotated types have args)
            origin = get_origin(type_hint)
            args = get_args(type_hint)

            if debug:
                logger.debug(f"  Field {field_name}: type={type_hint}, origin={origin}, args={args}")

            if args:
                # Look for ExposedParam in the metadata (args after the first one)
                for i, metadata in enumerate(args[1:], 1):  # Skip first arg (the actual type)
                    if debug:
                        logger.debug(f"\tArg[{i}]: {metadata} (type: {type(metadata).__name__})")

                    if isinstance(metadata, ExposedParam):
                        if debug:
                            logger.debug(f"\tFound ExposedParam for {field_name}")

                        # Resolve description priority:
                        # 1. Explicit Exposed(description=...)
                        # 2. Pydantic Field(description=...)
                        # 3. Prettified field name
                        final_description = metadata.description

                        if not final_description and field_name in pydantic_fields:
                            field_info = pydantic_fields[field_name]
                            pydantic_description = getattr(field_info, "description", None)
                            if pydantic_description:
                                final_description = pydantic_description

                        if not final_description:
                            final_description = field_name.replace("_", " ").title()

                        # Extract actual type (first arg in Annotated[T, ...])
                        actual_type = args[0]

                        # Create enriched ExposedParam with resolved description and type
                        enriched = ExposedParam(
                            name=metadata.name or field_name,
                            description=final_description,
                            extra=metadata.extra,
                            expose=metadata.expose,
                            persistent=metadata.persistent,
                            type_hint=actual_type,
                            type_source=ExposedParamTypeSource.ANNOTATED,
                        )

                        result[field_name] = enriched
                        break

    except NameError as e:
        # Log classes with unresolved type references for debugging
        if debug:
            logger.debug(
                f"Could not extract exposed params from {source_class.__name__}: unresolved type reference - {e}",
            )
    except Exception as e:
        # Log other unexpected errors at debug level
        if debug:
            logger.debug(f"Could not extract exposed params from {source_class}: {e}")

    return result


def _scan_annotated_fields_recursive(
    cls: type,
    prefix: str = "",
    current_depth: int = 0,
    max_depth: int = 5,
    visited: set[type] | None = None,
    debug: bool = False,
) -> dict[str, tuple[list[str], ExposedParam]]:
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
        Dict mapping flattened names to (path_parts, ExposedParam) tuples.
        The type_hint is accessible via ExposedParam.type_hint.
        Example: {
            "config_temperature": (
                ["config", "temperature"],
                ExposedParam(..., type_hint=float),
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
    direct_fields = _extract_annotated_fields(cls, debug=debug)
    for field_name, metadata in direct_fields.items():
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
                type_hint=metadata.type_hint,  # Type hint already populated
                type_source=ExposedParamTypeSource.ANNOTATED,
            )

        result[flat_name] = (path_parts, metadata)

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
        # Fall back to __annotations__ when generic type parameters can't be resolved
        # This ensures we still discover regular type-annotated attributes like `component: SomeClass`
        if debug:
            logger.debug(
                f"get_type_hints() failed for {cls.__name__} due to generic types: {e}. "
                f"Falling back to __annotations__",
            )
        raw_annotations = getattr(cls, "__annotations__", {})
        # Filter out generic type parameters and schema fields
        attrs_to_scan = {
            k: v
            for k, v in raw_annotations.items()
            if k not in ("InSchema", "OutSchema", "input_schema", "output_schema", "config_schema")
        }
        if debug:
            logger.debug(f"Found {len(attrs_to_scan)} annotations in {cls.__name__} via __annotations__")
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
            exposed_fields = _extract_annotated_fields(actual_type, debug=debug)

            for field_name, metadata in exposed_fields.items():
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
                        type_hint=metadata.type_hint,  # Type hint already populated
                        type_source=ExposedParamTypeSource.ANNOTATED,
                    )

                result[flat_name] = (path_parts, metadata)

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


def _create_property(
    cls: type,
    flat_name: str,
    path_parts: list[str],
    metadata: ExposedParam,
) -> None:
    """Create a property with getter/setter on the class that traverses a nested path.

    Args:
        cls: Class to add the property to
        flat_name: Flattened property name (e.g., "component_config_temperature")
        path_parts: Path to traverse (e.g., ["component", "config", "temperature"])
        metadata: ExposedParam metadata (with type_hint already populated)
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

                # If obj is still self, use object.__setattr__ to avoid recursion
                # (happens when path has only one element)
                if obj is self:
                    object.__setattr__(obj, path[-1], value)
                else:
                    setattr(obj, path[-1], value)
            except AttributeError as e:
                logger.warning(f"Could not set {'.'.join(path)}: {e}")

        return setter

    # Import ValidatedProperty here to avoid circular imports
    from .core import ValidatedProperty

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

    # Attach metadata (type_hint already populated in metadata)
    setattr(prop.fget, _EXPOSED_META_VAR_NAME, metadata)

    # Set the property on the class
    setattr(cls, flat_name, prop)


class _ExposureHelper:
    """Helper class for type extraction and runtime scanning in the exposure system.

    This class provides static methods for:
    - Extracting type information from properties and metadata
    - Recursively scanning instances for exposed parameters
    """

    @staticmethod
    def get_type_info_from_property(prop) -> tuple[Any | None, str, ExposedParamTypeSource]:
        """Extract type information from a property object.

        Args:
            prop: Property to extract type from

        Returns:
            Tuple of (type_hint, type_name_str, type_source)
        """
        # Try to get type from property's stored metadata first
        if hasattr(prop.fget, _EXPOSED_META_VAR_NAME):
            meta = getattr(prop.fget, _EXPOSED_META_VAR_NAME)
            type_hint = getattr(meta, "type_hint", None)
            type_source = getattr(meta, "type_source", ExposedParamTypeSource.NONE)

            if type_hint is not None:
                type_name = getattr(type_hint, "__name__", str(type_hint))
                if type_source is ExposedParamTypeSource.NONE:
                    type_source = ExposedParamTypeSource.ANNOTATED
                return type_hint, type_name, type_source

        # Fall back to inspecting property annotations
        type_hint, type_source = get_type_from_property(prop)
        if type_hint is not None:
            type_name = getattr(type_hint, "__name__", str(type_hint))
            return type_hint, type_name, type_source

        return None, "unknown", ExposedParamTypeSource.NONE

    @staticmethod
    def get_type_info_from_metadata(metadata: ExposedParam) -> tuple[Any | None, str, ExposedParamTypeSource]:
        """Extract type information from ExposedParam metadata.

        Args:
            metadata: ExposedParam metadata object

        Returns:
            Tuple of (type_hint, type_name_str, type_source)
        """
        if metadata.type_hint is not None:
            type_name = getattr(metadata.type_hint, "__name__", str(metadata.type_hint))
            return metadata.type_hint, type_name, ExposedParamTypeSource.ANNOTATED

        return None, "unknown", ExposedParamTypeSource.NONE

    @staticmethod
    def get_type_info_from_value(value: Any) -> tuple[Any | None, str, ExposedParamTypeSource]:
        """Infer type information from runtime value as fallback.

        Args:
            value: Runtime value to infer type from

        Returns:
            Tuple of (type_hint, type_name_str, type_source)
        """
        if value is not None:
            value_type = type(value)
            return value_type, value_type.__name__, ExposedParamTypeSource.RUNTIME
        return None, "unknown", ExposedParamTypeSource.NONE

    @staticmethod
    def scan_instance_recursively(
        instance: Any,
        include_values: bool,
        already_exposed: set[str],
        prefix: str,
        current_depth: int,
        max_depth: int,
        visited: set[int],
    ) -> list[ExposedParamRuntimeInfo]:
        """Recursively scan instance for exposed components.

        Args:
            instance: Object instance to scan
            include_values: Whether to include current values
            already_exposed: Set of already-discovered param names
            prefix: Path prefix for nested components
            current_depth: Current recursion depth
            max_depth: Maximum recursion depth
            visited: Set of visited object IDs (circular ref prevention)

        Returns:
            List of discovered ExposedParamRuntimeInfo objects
        """
        # Base case: max depth reached
        if current_depth >= max_depth:
            return []

        exposed = []

        # Scan instance attributes
        for attr_name in dir(instance):
            # Skip private, dunder, and already-exposed attributes
            if attr_name.startswith("_") or attr_name in already_exposed:
                continue

            # Skip if it's a class attribute or method
            if hasattr(type(instance), attr_name):
                cls_attr = getattr(type(instance), attr_name)
                # Skip properties, methods, classmethods, staticmethods
                if isinstance(cls_attr, (property, type(lambda: None), classmethod, staticmethod)):
                    continue

            # Get the instance attribute
            try:
                attr_value = getattr(instance, attr_name)
            except Exception:
                continue

            # Skip None, primitives, and built-in types
            if attr_value is None or isinstance(attr_value, (str, int, float, bool, list, dict, tuple, set)):
                continue

            # Circular reference check
            attr_id = id(attr_value)
            if attr_id in visited:
                continue
            visited.add(attr_id)

            # Check if this object has a class with Annotated fields
            attr_type = type(attr_value)
            component_fields = _extract_annotated_fields(attr_type)

            # Process exposed fields if any
            for field_name, metadata in component_fields.items():
                # Build flattened name with prefix
                flat_name = f"{prefix}_{attr_name}_{field_name}" if prefix else f"{attr_name}_{field_name}"

                # Skip if already exposed
                if flat_name in already_exposed:
                    continue

                # Get field value
                try:
                    field_value = getattr(attr_value, field_name)
                except Exception:
                    field_value = None

                # Get type info
                type_hint = metadata.type_hint
                if type_hint is not None:
                    param_type = getattr(type_hint, "__name__", str(type_hint))
                    type_source = ExposedParamTypeSource.ANNOTATED
                elif field_value is not None:
                    type_hint = type(field_value)
                    param_type = type_hint.__name__
                    type_source = ExposedParamTypeSource.RUNTIME
                else:
                    type_hint = None
                    param_type = "unknown"
                    type_source = ExposedParamTypeSource.NONE

                exposed.append(
                    ExposedParamRuntimeInfo(
                        name=flat_name,
                        description=metadata.description,
                        extra=metadata.extra,
                        expose=metadata.expose,
                        persistent=metadata.persistent,
                        type_hint=type_hint,
                        type_=param_type,
                        type_source=type_source,
                        editable=False,  # Runtime components don't have auto-generated setters
                        current_value=field_value if include_values else None,
                    ),
                )
                already_exposed.add(flat_name)

            # Recurse into component's instance attributes
            new_prefix = f"{prefix}_{attr_name}" if prefix else attr_name
            nested = _ExposureHelper.scan_instance_recursively(
                attr_value,
                include_values,
                already_exposed,
                new_prefix,
                current_depth + 1,
                max_depth,
                visited,
            )
            exposed.extend(nested)

        return exposed
