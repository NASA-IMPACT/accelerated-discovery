# Parameter Exposure System

## Introduction

The Parameter Exposure System enables agents and tools to expose their internal parameters for external configuration, introspection, and UI integration. This system provides a unified interface for discovering, accessing, and modifying parameters at runtime, making it easy to build configurable agents that can be controlled through APIs, user interfaces, or orchestration systems.

### Why Expose Parameters?

- **UI Integration**: Build control panels and configuration interfaces for agents
- **API Configurability**: Allow external systems to configure agent behavior
- **Introspection**: Enable debugging, monitoring, and workflow visualization
- **Reproducibility**: Capture and restore agent configurations for experiments

### Three Exposure Strategies

The system supports three complementary strategies for exposing parameters, each designed for different use cases:

1. **Decorator-Based (`@exposed_param`)** - Most controlled, recommended for critical parameters
2. **Annotation-Based (`Annotated[T, Exposed()]`)** - Automatic, best for config schemas
3. **Pipe Operator (`value | Exposed()`)** - Dynamic, works in `__init__` for runtime components

All three strategies integrate seamlessly and can be used together in the same agent.

---

## Strategy 1: Decorator-Based (`@exposed_param`)

**Most Controlled - Recommended for Critical Parameters**

The decorator-based approach provides the highest level of control over parameter exposure. Use `@exposed_param` on property methods to explicitly mark parameters that should be exposed to external systems.

### Basic Example

```python
from akd._base import exposed_param
from akd.agents._base import InstructorBaseAgent

class QueryAgent(InstructorBaseAgent):
    def __init__(self, config=None):
        super().__init__(config=config)
        self._clarification_prompt = "Default clarification prompt"
        self._max_iterations = 5
        self._temperature_override = 0.7

    @exposed_param(description="System prompt for query clarification")
    def clarification_prompt(self) -> str:
        return self._clarification_prompt

    @clarification_prompt.setter
    def clarification_prompt(self, val: str) -> None:
        self._clarification_prompt = val

    @exposed_param(description="Maximum research iterations", min=1, max=20)
    def max_iterations(self) -> int:
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, val: int) -> None:
        self._max_iterations = val

    @exposed_param  # Auto-description from docstring
    def temperature_override(self) -> float:
        """Temperature override for this agent"""
        return self._temperature_override

    @temperature_override.setter
    def temperature_override(self, val: float) -> None:
        self._temperature_override = val

    @exposed_param(description="Read-only computed value")
    def computed_value(self) -> str:
        """This property has no setter, so it's read-only."""
        return f"iterations={self._max_iterations}, temp={self._temperature_override}"
```

### Features

- **Explicit Control**: Only properties with `@exposed_param` are exposed
- **Automatic Type Validation**: Setters automatically validate types via `ValidatedProperty`
- **Extra Metadata**: Support custom metadata like constraints (`min`, `max`, etc.)
- **Auto-Description**: Uses docstring if no explicit description provided
- **Read-Only Support**: Properties without setters are marked as non-editable

### When to Use

- Critical agent parameters that require validation
- Read-only computed values
- Parameters with complex constraints or business logic
- When you need explicit control over what gets exposed

### Pros & Cons

**Pros:**
- Explicit and self-documenting
- Type validation built-in
- Full control over exposed parameters
- Supports read-only properties

**Cons:**
- More verbose than other strategies
- Requires manual getter/setter implementation
- Cannot be used on class attributes directly

---

## Strategy 2: Annotation-Based (`Annotated[T, Exposed()]`)

**Static Class-Level - Best for Config Schemas**

The annotation-based approach uses Python's `Annotated` type hints to mark fields for automatic exposure. This is the cleanest approach for config schemas and data classes.

### Basic Example

```python
from typing import Annotated
from akd._base.exposure import Exposed
from akd.agents._base import BaseAgentConfig, LiteLLMInstructorBaseAgent

class DummyExposedConfigSchema(BaseAgentConfig):
    """Config schema with exposed parameters."""

    topic_steer: Annotated[str, Exposed(description="Topic steering parameter")] = "default"
    temperature: Annotated[float, Exposed(description="Temperature setting")] = 0.7

class DummyAgentWithExposedConfig(LiteLLMInstructorBaseAgent):
    input_schema = AgentTestInputSchema
    output_schema = AgentTestOutputSchema
    config_schema = DummyExposedConfigSchema
```

### Nested Configuration Example

```python
class ExposedNestedConfig(BaseAgentConfig):
    """Nested configuration with exposed parameters."""

    temperature: Annotated[float, Exposed(description="Temperature parameter")] = 0.7
    max_tokens: Annotated[int, Exposed(description="Maximum tokens")] = 100

class ExposedComplexComponent:
    """Complex component with nested configuration."""

    name: Annotated[str, Exposed(description="Component name")] = "component"

    def __init__(self):
        self.config = ExposedNestedConfig()
```

### Access Patterns for Config Fields

Config schema fields support three equivalent access patterns:

```python
agent = DummyAgentWithExposedConfig()

# Pattern 1: Direct config access
agent.config.topic_steer = "new_value"

# Pattern 2: Prefixed property (auto-created by __init_subclass__)
agent.config_topic_steer = "new_value"

# Pattern 3: Dict-like access (see section 6)
agent["config.topic_steer"] = "new_value"

# All three are equivalent and stay in sync
```

### How It Works

When a class inherits from `ParamExposureMixin` (via base agent classes):

1. `__init_subclass__` scans for `Annotated[T, Exposed()]` fields
2. Auto-creates properties for nested paths (e.g., `config_temperature` → `config.temperature`)
3. Stores metadata in the class `_exposure_registry`
4. Properties provide getter/setter that traverse the nested path automatically

### Features

- **Declarative**: Visible directly in class definition
- **Automatic Property Creation**: Nested paths get properties automatically
- **Type Hints Preserved**: Full type safety with mypy/pyright
- **No Manual Getters/Setters**: Framework handles traversal
- **Class Creation Time**: Discovered via `__init_subclass__`, no runtime overhead

### When to Use

- Config schemas and data classes
- Class-level component attributes
- Nested configuration objects
- When you want declarative, visible exposure

### Pros & Cons

**Pros:**
- Clean, declarative syntax
- Automatic property creation for nested paths
- Type hints fully preserved
- No boilerplate getters/setters

**Cons:**
- Class-level only (doesn't work in `__init__`)
- Requires `Annotated` syntax (Python 3.9+)

---

## Strategy 3: Pipe Operator (`value | Exposed()`)

**Runtime Instance-Level - For Dynamic Components**

The pipe operator approach enables runtime parameter exposure using Python's `__ror__` operator. This is the only strategy that works for values created in `__init__`, where Python erases type annotations.

### Basic Example

```python
from akd._base.exposure import Exposed

class PipeNestedComponent:
    """Nested component using pipe operator for instance-level exposure."""

    def __init__(self):
        self.nested_field = "deeply nested value" | Exposed(
            description="Nested field via pipe",
        )

class PipeMixedComponent:
    """Component mixing class-level annotations and pipe operator."""

    # Class-level annotation
    class_field: Annotated[str, Exposed(description="Class-level field")] = "class_value"

    def __init__(self):
        # Instance-level pipe operator
        self.instance_field = "instance_value" | Exposed(description="Instance field via pipe")
        self.nested = PipeNestedComponent()
```

### Type Preservation

The pipe operator creates a dynamic subclass that preserves type behavior:

```python
component = PipeMixedComponent()
value = component.instance_field

# isinstance() works (AnnotatedStr is subclass of str)
assert isinstance(value, str)  # ✓ True

# All string operations work normally
assert value.upper() == "INSTANCE_VALUE"  # ✓ Works
assert value.startswith("instance")       # ✓ Works

# type() check shows the annotated subclass
assert type(value).__name__ == "AnnotatedStr"
```

### How It Works

1. `value | Exposed(...)` calls `Exposed.__ror__(value)`
2. Creates dynamic subclass (e.g., `AnnotatedStr` for `str`)
3. Attaches `_exposed_metadata` attribute to the instance
4. Runtime scanner detects `_exposed_metadata` and collects parameters
5. LRU cache preserves metadata across scans

### Dynamic Subclass Creation

The system uses `@functools.cache` to create type-specific subclasses:

```python
"text" | Exposed(...)       # Creates AnnotatedStr subclass
42 | Exposed(...)           # Creates AnnotatedInt subclass
[1, 2, 3] | Exposed(...)    # Creates AnnotatedList subclass
```

Built-in types (str, int, float, list, dict, etc.) get special subclasses that preserve immutability and behavior. Custom types get direct `_exposed_metadata` injection.

### Features

- **Works in `__init__`**: Only strategy that works where annotations are erased
- **Clean Syntax**: Intuitive `value | Exposed(...)` expression
- **Type Preservation**: `isinstance()` checks still work
- **Runtime Discovery**: Found via `scan_runtime=True`
- **LRU Cached**: Metadata preserved across scans even if value changes

### When to Use

- Components created dynamically in `__init__`
- Runtime-constructed values
- Quick prototyping and experimentation
- When annotations aren't available

### Pros & Cons

**Pros:**
- Works in `__init__` where annotations don't
- Clean, readable syntax
- `isinstance()` checks preserved
- Flexible and dynamic

**Cons:**
- Type mutation (`type(x) == str` fails, but `isinstance(x, str)` works)
- Requires runtime discovery (`scan_runtime=True`)
- Slightly less discoverable than class-level annotations

---

## Getting Exposed Parameters

All exposed parameters can be retrieved using the `get_exposed_params()` method available on any agent that inherits from `ParamExposureMixin`.

### Basic Usage

```python
# Get all exposed params (decorator + annotated + runtime)
params = agent.get_exposed_params(
    include_values=True,      # Include current values
    scan_runtime=True,        # Discover pipe operator params
    runtime_max_depth=3,      # Max recursion depth for nested components
)

for param in params:
    print(f"Name: {param.name}")
    print(f"Description: {param.description}")
    print(f"Type: {param.type_}")
    print(f"Editable: {param.editable}")
    print(f"Current Value: {param.current_value}")
    print(f"Extra Metadata: {param.extra}")
```

### Discovery Priority

Parameters are discovered in three phases with the following priority:

1. **`@exposed_param` decorator** (highest priority)
   - Explicit properties with decorator
   - Most control, takes precedence

2. **Registry from `Annotated` fields** (class-level)
   - Config schema fields
   - Class-level component attributes
   - Discovered at class creation time

3. **Runtime scanning** (if `scan_runtime=True`)
   - Pipe operator exposed params
   - Nested components created in `__init__`
   - Recursively scans up to `runtime_max_depth`

If the same parameter is exposed via multiple strategies, decorator metadata takes precedence.

### ExposedParamRuntimeInfo Structure

Each parameter returns an `ExposedParamRuntimeInfo` object with:

```python
@dataclass
class ExposedParamRuntimeInfo:
    name: str                    # Parameter name (dot-separated path)
    description: str             # Human-readable description
    type_: str                   # Python type name ("str", "int", etc.)
    type_hint: Any              # Full type hint object
    type_source: str            # Source of type info (fget, fset, runtime)
    editable: bool              # Whether parameter has a setter
    expose: bool                # Whether to expose (default: True)
    persistent: bool            # Whether to persist (default: True)
    extra: dict                 # Additional metadata (min, max, etc.)
    current_value: Any | None   # Current value (if include_values=True)
```

### Parameters

- **`include_values`** (bool, default: False)
  - If True, includes current runtime values in `current_value` field
  - Use when you need to serialize current state

- **`scan_runtime`** (bool, default: True)
  - If True, recursively scans instance attributes for pipe operator params
  - Required to discover components created in `__init__`

- **`runtime_max_depth`** (int, default: 3)
  - Maximum recursion depth for runtime scanning
  - Prevents infinite loops in circular references

---

## Dict-Like Access Pattern

Agents provide a dict-like interface for accessing and modifying exposed parameters using dot-separated paths.

### Basic Operations

```python
# Get value (raises AttributeError if missing)
value = agent["component.config.temperature"]

# Set value
agent["component.config.temperature"] = 0.9

# Safe get with default (no exception)
value = agent.get("component.config.temperature", 0.7)
missing = agent.get("nonexistent.path", "default")
```

### Round-Trip Consistency

All access methods stay in sync:

```python
# Set via dict interface, get via direct access
agent["component.name"] = "updated"
assert agent.component.name == "updated"

# Set via direct access, get via dict interface
agent.component.config.temperature = 0.85
assert agent["component.config.temperature"] == 0.85

# All access patterns are consistent
temp1 = agent["component.config.temperature"]
temp2 = agent.component.config.temperature
temp3 = agent.get("component.config.temperature")
assert temp1 == temp2 == temp3
```

### Safe Access with `get()`

The `get()` method returns a default value instead of raising an exception:

```python
# Existing paths
temp = agent.get("component.config.temperature", 0.7)

# Non-existing paths don't raise
missing = agent.get("nonexistent.path")  # Returns None
fallback = agent.get("also.missing", "default")  # Returns "default"
```

### Implementation

The dict-like interface is implemented via three methods in `ParamExposureMixin`:

- **`__getitem__(key: str)`** - Get value, raise if missing
- **`__setitem__(key: str, value: Any)`** - Set value, raise if path invalid
- **`get(key: str, default: Any = None)`** - Get value or default

All three use `rgetattr()` and `rsetattr()` utilities internally.

---

## Utilities: `rgetattr()` and `rsetattr()`

The exposure system uses two recursive attribute access utilities from `akd.utils` for traversing nested paths.

### `rgetattr(obj, attr)` - Recursive Get Attribute

Gets a nested attribute using a dot-separated string:

```python
from akd.utils import rgetattr

# Instead of:
value = obj.component.config.temperature

# Use:
value = rgetattr(obj, "component.config.temperature")
```

**Implementation:**
- Uses `operator.attrgetter` for clean, efficient access
- Raises `AttributeError` if any attribute in path doesn't exist
- Used internally by `__getitem__` and `get()`

### `rsetattr(obj, attr, val)` - Recursive Set Attribute

Sets a nested attribute using a dot-separated string:

```python
from akd.utils import rsetattr

# Instead of:
obj.component.config.temperature = 0.9

# Use:
rsetattr(obj, "component.config.temperature", 0.9)
```

**Implementation:**
- Uses `functools.reduce` to traverse parent chain
- Sets attribute on final parent object
- Raises `AttributeError` if any intermediate path doesn't exist
- Used internally by `__setitem__`

### Example Usage

```python
from akd.utils import rgetattr, rsetattr

# Get nested values
model_name = rgetattr(agent, "config.model_name")
temperature = rgetattr(agent, "tool.rate_limiter.max_calls")

# Set nested values
rsetattr(agent, "config.temperature", 0.8)
rsetattr(agent, "component.nested.value", "new_value")

# These are equivalent to agent["..."] but can be used independently
```

---

## Recommendations

### When to Use Each Strategy

#### 1. Decorator (`@exposed_param`) - MOST CONTROLLED

**Use for:**
- Critical agent parameters that require validation
- Read-only computed values
- Parameters with complex constraints (min/max, regex, etc.)
- Business logic that needs to run on parameter changes

**Pros:**
- Explicit and self-documenting
- Type validation built-in
- Most control over exposure
- Supports read-only properties

**Cons:**
- More verbose than other strategies
- Requires manual getter/setter implementation

**Example:**
```python
@exposed_param(description="API rate limit", min=1, max=100)
def rate_limit(self) -> float:
    return self._rate_limit

@rate_limit.setter
def rate_limit(self, val: float) -> None:
    if val < 1 or val > 100:
        raise ValueError("Rate limit must be between 1 and 100")
    self._rate_limit = val
    self._reset_rate_limiter()  # Business logic on change
```

#### 2. Annotated (`Annotated[T, Exposed()]`) - AUTOMATIC

**Use for:**
- Config schemas and data classes
- Class-level component attributes
- Nested configuration objects
- When you want declarative, visible exposure

**Pros:**
- Clean, declarative syntax
- Automatic property creation for nested paths
- Type hints fully preserved
- No boilerplate code

**Cons:**
- Class-level only (doesn't work in `__init__`)

**Example:**
```python
class AgentConfig(BaseAgentConfig):
    temperature: Annotated[float, Exposed(description="LLM temperature")] = 0.7
    max_tokens: Annotated[int, Exposed(description="Max output tokens")] = 1000
```

#### 3. Pipe Operator (`| Exposed()`) - DYNAMIC

**Use for:**
- Components created in `__init__`
- Runtime-constructed values
- Quick prototyping
- When annotations aren't available

**Pros:**
- Works in `__init__` where annotations don't
- Clean, readable syntax
- `isinstance()` checks preserved

**Cons:**
- Type mutation (minor - `isinstance()` still works)
- Requires runtime discovery (`scan_runtime=True`)

**Example:**
```python
def __init__(self):
    self.rate_limiter = RateLimiter(5.0) | Exposed(
        description="API rate limiter component"
    )
```

### General Recommendation

**Start with the most appropriate strategy for your use case:**

1. **Agent-level parameters**: Use `@exposed_param`
   - Gives you explicit control and validation
   - Best for parameters that change agent behavior

2. **Config schemas**: Use `Annotated[T, Exposed()]`
   - Clean, declarative, visible in class definition
   - Automatic property creation for nested access

3. **Dynamic components**: Use `| Exposed()`
   - Only works in `__init__` where you need it
   - Quick and flexible for runtime composition

**Mix strategies freely** - all three work together seamlessly in the same agent.

---

## Complete Example

Here's a complete agent that demonstrates all three exposure strategies working together:

```python
from typing import Annotated
from akd._base import exposed_param, Exposed
from akd.agents._base import LiteLLMInstructorBaseAgent, BaseAgentConfig

class ResearchConfig(BaseAgentConfig):
    """Configuration for research agent."""

    # Strategy 2: Annotated (config schema)
    temperature: Annotated[float, Exposed(description="LLM temperature")] = 0.7
    max_iterations: Annotated[int, Exposed(description="Max iterations")] = 10
    model_name: Annotated[str, Exposed(description="LLM model")] = "gpt-4"

class ToolComponent:
    """Dynamic tool component created at runtime."""

    def __init__(self):
        # Strategy 3: Pipe operator (instance-level)
        self.rate_limit = 5.0 | Exposed(description="API rate limit (calls/sec)")
        self.enabled = True | Exposed(description="Whether tool is enabled")

class ResearchAgent(LiteLLMInstructorBaseAgent):
    """Research agent with all three exposure strategies."""

    config_schema = ResearchConfig

    def __init__(self, config=None):
        super().__init__(config)
        self._system_prompt = "You are a research assistant"
        self._verbose = False
        self.tool = ToolComponent()  # Runtime component

    # Strategy 1: Decorator (most controlled)
    @exposed_param(description="System prompt for the agent")
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, val: str) -> None:
        if not val.strip():
            raise ValueError("System prompt cannot be empty")
        self._system_prompt = val

    @exposed_param(description="Enable verbose logging")
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, val: bool) -> None:
        self._verbose = val

    @exposed_param(description="Agent status summary (read-only)")
    def status(self) -> str:
        """Read-only computed value."""
        return f"Model: {self.config.model_name}, Temp: {self.config.temperature}"

# Usage Examples
agent = ResearchAgent()

# 1. Get all exposed parameters
params = agent.get_exposed_params(include_values=True, scan_runtime=True)
for p in params:
    print(f"{p.name}: {p.current_value} ({p.type_})")

# Output:
# system_prompt: You are a research assistant (str)
# verbose: False (bool)
# status: Model: gpt-4, Temp: 0.7 (str)
# config_temperature: 0.7 (float)
# config_max_iterations: 10 (int)
# config_model_name: gpt-4 (str)
# tool.rate_limit: 5.0 (AnnotatedFloat)
# tool.enabled: True (AnnotatedBool)

# 2. Dict-like access works for ALL strategies
agent["config_temperature"] = 0.9          # Strategy 2 (Annotated)
agent["system_prompt"] = "New prompt"      # Strategy 1 (Decorator)
agent["tool.rate_limit"] = 10.0            # Strategy 3 (Pipe operator)

# 3. Direct access also works
agent.config.temperature = 0.8
agent.system_prompt = "Another prompt"
agent.tool.rate_limit = 7.5

# 4. Safe access with get()
temp = agent.get("config_temperature", 0.7)
missing = agent.get("nonexistent.param", "default")

# 5. All access patterns stay in sync
assert agent["config_temperature"] == agent.config.temperature == 0.8
assert agent["system_prompt"] == agent.system_prompt == "Another prompt"
```

### Output Inspection

```python
# Inspect all parameters
for param in agent.get_exposed_params():
    print(f"\nParameter: {param.name}")
    print(f"  Description: {param.description}")
    print(f"  Type: {param.type_}")
    print(f"  Editable: {param.editable}")
    print(f"  Source: {param.type_source}")
    if param.extra:
        print(f"  Extra: {param.extra}")

# Parameter: system_prompt
#   Description: System prompt for the agent
#   Type: str
#   Editable: True
#   Source: fget

# Parameter: verbose
#   Description: Enable verbose logging
#   Type: bool
#   Editable: True
#   Source: fget

# Parameter: status
#   Description: Agent status summary (read-only)
#   Type: str
#   Editable: False
#   Source: fget

# Parameter: config_temperature
#   Description: LLM temperature
#   Type: float
#   Editable: True
#   Source: annotation

# ... (more parameters)
```

---

## Summary

The Parameter Exposure System provides three complementary strategies for making agent parameters discoverable and configurable:

1. **`@exposed_param` decorator** - Explicit, validated, most control
2. **`Annotated[T, Exposed()]`** - Declarative, automatic, clean
3. **`value | Exposed()`** - Dynamic, works in `__init__`, flexible

All three strategies integrate seamlessly, provide a unified `get_exposed_params()` interface, and support dict-like access patterns for convenient parameter manipulation.

Choose the strategy that best fits your use case, or mix all three in the same agent for maximum flexibility.
