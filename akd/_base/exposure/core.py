import types
from dataclasses import replace as dc_replace
from typing import Any, Callable, get_args, get_type_hints, overload

from loguru import logger

from akd.utils import rgetattr, rsetattr

from .structures import (
    _EXPOSED_META_VAR_NAME,
    ExposedParam,
    ExposedParamRuntimeInfo,
    ExposedParamTypeSource,
)
from .utils import (
    _create_property,
    _ExposureHelper,
    _extract_annotated_fields,
    _scan_annotated_fields_recursive,
    get_type_from_property,
)

# Property validation and decorator


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
            config_exposed = _extract_annotated_fields(cls.config_schema)
            # Add config_ prefix to field names for registry keys (full path naming)
            for field_name, metadata in config_exposed.items():
                registry_key = f"config_{field_name}"
                # Update metadata.name to match registry_key for consistency
                cls._exposure_registry[registry_key] = dc_replace(metadata, name=registry_key)

        # NEW: Extract and store class-level annotated fields
        # Skip schema attributes to avoid conflicts
        skip_fields = {"InSchema", "OutSchema", "input_schema", "output_schema", "config_schema"}
        class_exposed = _extract_annotated_fields(cls, skip_fields=skip_fields)
        # No prefix for class-level fields
        cls._exposure_registry.update(class_exposed)

        # Scan for annotated fields (including nested components)
        discovered = _scan_annotated_fields_recursive(cls)

        # Create properties for fields that should be exposed
        for flat_name, (path_parts, metadata) in discovered.items():
            # Only create properties for persistent, exposed fields
            if metadata.expose and metadata.persistent:
                # Skip class-level attributes (single-element paths) - they work fine as-is
                # Only create properties for nested paths that need traversal (e.g., config.temperature)
                if len(path_parts) == 1:
                    continue

                # Check if it's already a property with _exposed_meta (from @exposed_param decorator)
                existing_attr = getattr(cls, flat_name, None)
                is_exposed_property = isinstance(existing_attr, property) and hasattr(
                    existing_attr.fget,
                    _EXPOSED_META_VAR_NAME,
                )

                if not is_exposed_property:
                    # Create property for nested path (e.g., config_temperature -> config.temperature)
                    _create_property(cls, flat_name, path_parts, metadata)
                else:
                    logger.debug(
                        f"Skipping auto-creation of property '{flat_name}' on {cls.__name__} - already exposed via decorator",
                    )

    def _collect_exposed_from_decorated_properties(
        self,
        include_values: bool = False,
    ) -> list[ExposedParamRuntimeInfo]:
        """Collect exposed params from @exposed_param decorated properties.

        Args:
            include_values: Whether to include current runtime values

        Returns:
            List of ExposedParamRuntimeInfo objects
        """
        exposed = []

        for attr_name in dir(self):
            attr = getattr(type(self), attr_name, None)
            if not (isinstance(attr, property) and hasattr(attr.fget, _EXPOSED_META_VAR_NAME)):
                continue

            meta = getattr(attr.fget, _EXPOSED_META_VAR_NAME)

            # Get type info from property using helper
            type_hint, param_type, type_source = _ExposureHelper.get_type_info_from_property(attr)

            # Fallback to runtime value if type is still unknown
            current_value = None
            if param_type == "unknown":
                current_value = getattr(self, attr_name, None)
                if current_value is not None:
                    type_hint, param_type, type_source = _ExposureHelper.get_type_info_from_value(current_value)

            # Get current value if requested and not already fetched
            if include_values and current_value is None:
                current_value = getattr(self, attr_name, None)

            exposed.append(
                ExposedParamRuntimeInfo(
                    name=attr_name,
                    description=meta.description,
                    extra=meta.extra,
                    expose=meta.expose,
                    persistent=meta.persistent,
                    type_hint=type_hint,
                    type_=param_type,
                    type_source=type_source,
                    editable=attr.fset is not None,
                    current_value=current_value if include_values else None,
                ),
            )

        return exposed

    def _collect_exposed_from_registry(
        self,
        exposed_names: set[str] | None = None,
        include_values: bool = False,
    ) -> list[ExposedParamRuntimeInfo]:
        """Collect exposed params from registry (config-level fields).

        Args:
            exposed_names: Optional set of already-collected param names to skip (for deduplication)
            include_values: Whether to include current runtime values

        Returns:
            List of ExposedParamRuntimeInfo objects
        """
        if exposed_names is None:
            exposed_names = set()

        exposed = []
        registry = getattr(type(self), "_exposure_registry", {})

        for registry_key, metadata in registry.items():
            # Skip if already exposed via decorator (decorator has priority)
            if registry_key in exposed_names:
                continue

            # Check if a property was created for this registry entry
            attr = getattr(type(self), registry_key, None)
            if isinstance(attr, property):
                # Property exists, get type from it
                type_hint, param_type, type_source = _ExposureHelper.get_type_info_from_property(attr)
                editable = attr.fset is not None
            else:
                # No property - use type from registry metadata
                editable = metadata.extra.get("editable", True)
                type_hint, param_type, type_source = _ExposureHelper.get_type_info_from_metadata(metadata)

                # Fallback to runtime value if type still unknown
                if param_type == "unknown":
                    current_value = getattr(self, registry_key, None)
                    if current_value is not None:
                        type_hint, param_type, type_source = _ExposureHelper.get_type_info_from_value(current_value)

            # Get current value if requested
            current_value = None
            if include_values:
                current_value = getattr(self, registry_key, None)

            exposed.append(
                ExposedParamRuntimeInfo(
                    name=metadata.name or registry_key,
                    description=metadata.description,
                    extra=metadata.extra,
                    expose=metadata.expose,
                    persistent=metadata.persistent,
                    type_hint=type_hint,
                    type_=param_type,
                    type_source=type_source,
                    editable=editable,
                    current_value=current_value if include_values else None,
                ),
            )

        return exposed

    def _collect_exposed_from_runtime(
        self,
        include_values: bool = False,
        already_exposed: set[str] | None = None,
        max_depth: int = 2,
    ) -> list[ExposedParamRuntimeInfo]:
        """Scan instance attributes at runtime for components with exposed fields.

        Recursively discovers nested runtime components up to max_depth.

        Args:
            include_values: Whether to include current runtime values
            already_exposed: Set of already-exposed param names to skip
            max_depth: Maximum recursion depth (default: 2)

        Returns:
            List of ExposedParamRuntimeInfo objects for runtime-discovered component fields
        """
        if already_exposed is None:
            already_exposed = set()

        return _ExposureHelper.scan_instance_recursively(
            instance=self,
            include_values=include_values,
            already_exposed=already_exposed,
            prefix="",
            current_depth=0,
            max_depth=max_depth,
            visited=set(),
        )

    def get_exposed_params(
        self,
        include_values: bool = False,
        scan_runtime: bool = True,
        runtime_max_depth: int = 3,
    ) -> list[ExposedParamRuntimeInfo]:
        """
        Get metadata about exposed parameters from decorator and registry sources.

        This method discovers parameters exposed via:
        1. @exposed_param decorator on properties (highest priority)
        2. Registry entries for Annotated fields (populated at class creation time)
        3. Runtime component scanning (if include_runtime_components=True)

        Priority resolution:
        - Decorator metadata (@exposed_param) takes precedence over registry
        - Registry contains all Annotated[T, Exposed()] fields discovered at class creation
        - Runtime scanning finds components created in __init__ without class-level declarations

        Args:
            include_values: If True, include current runtime values in the output
            include_runtime_components: If True, scan instance attributes for components
                with exposed fields (useful for components not declared at class level)
            runtime_max_depth: Maximum recursion depth for runtime component scanning (default: 3)

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
        # STEP 1: Collect from @exposed_param decorated properties
        exposed = self._collect_exposed_from_decorated_properties(include_values)
        exposed_names = {p.name for p in exposed}

        # STEP 2: Collect from registry (contains all Annotated fields)
        exposed.extend(self._collect_exposed_from_registry(exposed_names, include_values))
        exposed_names = {p.name for p in exposed}

        # STEP 3: Optional runtime component scanning (with recursion)
        if scan_runtime:
            exposed.extend(
                self._collect_exposed_from_runtime(include_values, exposed_names, runtime_max_depth),
            )

        return exposed

    def __getitem__(self, key: str) -> Any:
        """Get exposed parameter value using dict-like syntax.

        Args:
            key: Dot-separated path to the parameter (e.g., 'component.config.temperature')

        Returns:
            The value at the specified path

        Raises:
            AttributeError: If the path doesn't exist

        Example:
            >>> value = agent['component.config.temperature']
        """
        return rgetattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set exposed parameter value using dict-like syntax.

        Args:
            key: Dot-separated path to the parameter (e.g., 'component.config.temperature')
            value: The value to set

        Raises:
            AttributeError: If the path doesn't exist

        Example:
            >>> agent['component.config.temperature'] = 0.8
        """
        rsetattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get exposed parameter value with optional default.

        Dict-like interface that returns a default value if the key doesn't exist,
        rather than raising an exception.

        Args:
            key: Dot-separated path to the parameter (e.g., 'component.config.temperature')
            default: Value to return if the path doesn't exist (default: None)

        Returns:
            The value at the specified path, or default if not found

        Example:
            >>> temp = agent.get('component.config.temperature', 0.7)
            >>> missing = agent.get('nonexistent.path', 'default_value')
        """
        try:
            return rgetattr(self, key)
        except AttributeError:
            return default
