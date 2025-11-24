# AKD Base Classes: Configuration and Attribute Reference Guide

## Overview

The AKD framework provides two base classes for all agents and tools:
- **`AbstractBase`**: Strict base class with required schema validation
- **`UnrestrictedAbstractBase`**: Flexible base class without strict schema enforcement

Both classes implement **reference semantics** for config attributes, meaning changes to `agent.field` automatically update `agent.config.field` and vice versa.

## Attribute Binding

### How It Works

When you define a class that inherits from `AbstractBase`:

```python
from akd.agents import BaseAgent
from akd.agents._base import BaseAgentConfig

class MyAgentConfig(BaseAgentConfig):
    temperature: float = 0.7
    max_tokens: int = 1000

class MyAgent(BaseAgent):
    config_schema = MyAgentConfig
```

The `AbstractBaseMeta` metaclass **automatically creates properties** at class definition time for all fields in `config_schema`. These properties provide reference semantics:

```python
agent = MyAgent(config=MyAgentConfig(temperature=0.5))

# These are equivalent - both access the same underlying config field
agent.temperature = 0.9
agent.config.temperature = 0.9

# They always stay in sync
assert agent.temperature == agent.config.temperature  # Always True
```

### Why Reference Semantics?

Reference semantics ensure:
1. **Single source of truth**: Config object is the canonical source
2. **Automatic synchronization**: No need to manually sync values
3. **Consistent state**: Impossible for `agent.field` and `agent.config.field` to diverge
4. **Test isolation**: Each instance has independent config state

### Property Factory Functions

Two module-level factory functions create properties:

#### `_make_config_property(field_name)`

Creates a read-write property for regular config fields:

```python
def _make_config_property(field_name: str):
    def getter(self):
        if not hasattr(self, "config") or self.config is None:
            return self.__dict__.get(field_name)  # Fallback for pre-init
        return getattr(self.config, field_name)

    def setter(self, value):
        if not hasattr(self, "config") or self.config is None:
            self.__dict__[field_name] = value  # Fallback for pre-init
        else:
            setattr(self.config, field_name, value)

    return property(getter, setter)
```

**Features**:
- Delegates to `self.config.field_name` for both get and set
- Fallback to `__dict__` for pre-init access (before `__init__` is called)
- Maintains reference semantics

#### `_make_computed_property(field_name)`

Creates a read-only property for computed fields (decorated with `@computed_field`):

```python
def _make_computed_property(field_name: str):
    def getter(self):
        if not hasattr(self, "config") or self.config is None:
            return self.__dict__.get(field_name)
        return getattr(self.config, field_name)

    return property(getter)  # No setter
```

**Features**:
- Read-only (attempting to set raises AttributeError)
- Dynamically computed from other config values
- Fallback for pre-init access

### Metaclass Architecture

#### Why Metaclass?

Properties must be created at **class definition time**, not instance time:

```python
# ❌ WRONG: Creating properties in __init__ sets them on the class
class MyClass:
    def __init__(self):
        type(self).my_prop = property(...)  # Sets on class, not instance!

# ✅ CORRECT: Metaclass creates properties at class definition time
class MyMeta(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)
        cls.my_prop = property(...)  # Created once per class
        return cls
```

#### AbstractBaseMeta Implementation

```python
class AbstractBaseMeta(ABCMeta):
    @staticmethod
    def _create_config_properties(target_class, dct):
        """Create properties for config fields at class definition time."""
        if not hasattr(target_class, 'config_schema') or target_class.config_schema is None:
            return

        # Create properties for regular model fields
        if hasattr(target_class.config_schema, 'model_fields'):
            for field_name in target_class.config_schema.model_fields.keys():
                if field_name in dct or isinstance(getattr(target_class, field_name, None), property):
                    continue
                setattr(target_class, field_name, _make_config_property(field_name))

        # Create read-only properties for computed fields
        if hasattr(target_class.config_schema, 'model_computed_fields'):
            for field_name in target_class.config_schema.model_computed_fields.keys():
                if field_name in dct or isinstance(getattr(target_class, field_name, None), property):
                    continue
                setattr(target_class, field_name, _make_computed_property(field_name))

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        # Create config properties at class definition time for ALL classes
        # This must happen before early return to ensure base classes get properties too
        AbstractBaseMeta._create_config_properties(cls, dct)

        # Skip schema validation for base classes
        if name in ["AbstractBase", "UnrestrictedAbstractBase", "BaseAgent",
                    "InstructorBaseAgent", "LiteLLMInstructorBaseAgent", "BaseTool"]:
            return cls

        # ... schema validation code ...

        return cls
```

**Key Points**:
- `_create_config_properties()` is a `@staticmethod` for functional design
- Property creation happens **before** the early return for base classes
- This ensures even base classes like `InstructorBaseAgent` get properties
- Properties are only created if not already present (skip list check)

## Attribute Resolution Precedence

When you access an attribute like `agent.temperature`, the resolution order is:

### Standard Precedence

For most config attributes:

1. **kwargs passed to `__init__`** (highest priority)
   ```python
   agent = MyAgent(config=config, custom_param="value")
   # custom_param is set via kwargs in _post_init()
   ```

2. **Config object field value**
   ```python
   config = MyAgentConfig(temperature=0.8)
   agent = MyAgent(config=config)
   agent.temperature  # Returns 0.8 from config
   ```

3. **Config schema default value** (lowest priority)
   ```python
   class MyAgentConfig(BaseAgentConfig):
       temperature: float = 0.7  # Default

   agent = MyAgent(config=MyAgentConfig())
   agent.temperature  # Returns 0.7 (default)
   ```

### Special Case: `debug` Parameter

⚠️ **Note**: The `debug` parameter has different behavior in `AbstractBase`:

In `AbstractBase`, the `debug` init parameter takes precedence over the config field:

```python
config = MyAgentConfig(debug=False)
agent = MyAgent(config=config, debug=True)
agent.debug  # Returns True (init parameter wins)
```

**Precedence order for `debug` in AbstractBase**:
1. `debug` init parameter (highest)
2. `config.debug` field value
3. Config schema default

**Note**: `UnrestrictedAbstractBase` has different `debug` precedence where the config field takes priority over the init parameter.

### Precedence Summary Table

| Attribute Type | Init Param | kwargs | Config Field | Config Default | Class Attr |
|----------------|-----------|---------|--------------|----------------|------------|
| `debug` (AbstractBase) | **1 (wins)** | N/A | 2 | 3 | ❌ Blocks |
| Other attrs | N/A | **1 (wins)** | 2 | 3 | ❌ Blocks |

## Class-Level Attributes Warning

⚠️ **Critical Warning**: Defining class-level attributes **blocks property creation** and breaks reference semantics.

### What NOT To Do

```python
class BrokenAgent(BaseAgent):
    temperature: float = 0.0  # ❌ DON'T DO THIS!
    config_schema = BaseAgentConfig

config = BaseAgentConfig(temperature=0.9)
agent = BrokenAgent(config=config)

# Property was NOT created because 'temperature' exists in class dict
agent.temperature  # Returns 0.0, NOT 0.9! 💥
agent.config.temperature  # Returns 0.9 ✅

# Reference semantics are broken:
agent.temperature = 0.5
assert agent.temperature == agent.config.temperature  # ❌ FAILS!
```

### Why This Happens

The metaclass checks if an attribute is already defined in the class dictionary before creating a property:

```python
# From AbstractBaseMeta._create_config_properties()
for field_name in target_class.config_schema.model_fields.keys():
    if field_name in dct:  # Skip if already defined
        continue
    # Only create property if not already present
    setattr(target_class, field_name, _make_config_property(field_name))
```

If you define `temperature: float = 0.0` in your class body, it's in `dct`, so the property is skipped.

### The Correct Way

Let the config schema define all fields:

```python
class MyAgentConfig(BaseAgentConfig):
    temperature: float = 0.0  # ✅ Define in config schema

class MyAgent(BaseAgent):
    config_schema = MyAgentConfig  # ✅ No class-level attributes

config = MyAgentConfig(temperature=0.9)
agent = MyAgent(config=config)

agent.temperature  # Returns 0.9 ✅
agent.config.temperature  # Returns 0.9 ✅
# Reference semantics work correctly
```

## Common Patterns and Examples

### Pattern 1: Standard Config Override

```python
from akd.agents import BaseAgent, BaseAgentConfig

# Define custom config with defaults
class MyAgentConfig(BaseAgentConfig):
    temperature: float = 0.7
    max_tokens: int = 1000

# Create agent with default config
agent1 = MyAgent(config=MyAgentConfig())
agent1.temperature  # 0.7 (default)

# Create agent with custom config
agent2 = MyAgent(config=MyAgentConfig(temperature=0.9))
agent2.temperature  # 0.9 (overridden)

# Modify at runtime
agent2.temperature = 0.5
agent2.config.temperature  # 0.5 (stays in sync)
```

### Pattern 2: kwargs Override

```python
# Pass additional kwargs
agent = MyAgent(
    config=MyAgentConfig(temperature=0.7),
    custom_param="value",
    another_param=123
)

# kwargs are set in _post_init()
agent.custom_param  # "value"
agent.another_param  # 123
```

### Pattern 3: Debug Override

```python
# Init parameter takes precedence in AbstractBase
config = MyAgentConfig(debug=False)
agent = MyAgent(config=config, debug=True)
agent.debug  # True (init parameter wins)
agent.config.debug  # False (config unchanged)
```

### Pattern 4: Independent Instances

```python
# Best practice: Create separate config instances
agent1 = MyAgent(config=MyAgentConfig(temperature=0.7))
agent2 = MyAgent(config=MyAgentConfig(temperature=0.7))

agent1.temperature = 0.9
agent2.temperature  # 0.7 (independent)

# ⚠️ Avoid sharing config objects
shared_config = MyAgentConfig(temperature=0.7)
agent1 = MyAgent(config=shared_config)
agent2 = MyAgent(config=shared_config)

agent1.temperature = 0.9
agent2.temperature  # 0.9 (both reference same config!)
```

### Pattern 5: Computed Fields

```python
from pydantic import computed_field

class MyAgentConfig(BaseAgentConfig):
    temperature: float = 0.7

    @computed_field
    @property
    def double_temperature(self) -> float:
        return self.temperature * 2

agent = MyAgent(config=MyAgentConfig())
agent.double_temperature  # 1.4 (read-only)

agent.temperature = 0.5
agent.double_temperature  # 1.0 (automatically recomputed)

agent.double_temperature = 2.0  # ❌ AttributeError: can't set attribute
```

## Testing Considerations

### Test Isolation

Properties created at class definition time ensure test isolation:

```python
def test_one():
    agent = MyAgent(config=MyAgentConfig(temperature=0.7))
    agent.temperature = 0.9
    assert agent.temperature == 0.9

def test_two():
    agent = MyAgent(config=MyAgentConfig(temperature=0.7))
    # This agent's temperature is NOT affected by test_one
    assert agent.temperature == 0.7  # ✅ Passes
```

### Testing Reference Semantics

```python
def test_reference_semantics():
    config = MyAgentConfig(temperature=0.7)
    agent = MyAgent(config=config)

    # Test bidirectional sync
    agent.temperature = 0.9
    assert agent.config.temperature == 0.9

    agent.config.temperature = 0.5
    assert agent.temperature == 0.5

    # Test that they reference the same value
    assert agent.temperature == agent.config.temperature
```

### Testing Class-Level Attributes (Negative Test)

```python
def test_class_attr_blocks_property():
    """Verify that class-level attributes block property creation."""
    class BrokenAgent(BaseAgent):
        temperature: float = 0.0  # Class-level attribute
        config_schema = BaseAgentConfig

    config = BaseAgentConfig(temperature=0.9)
    agent = BrokenAgent(config=config)

    # Property was not created, class attribute wins
    assert agent.temperature == 0.0  # NOT 0.9
    assert agent.config.temperature == 0.9

    # Reference semantics are broken
    agent.temperature = 0.5
    assert agent.temperature == 0.5
    assert agent.config.temperature == 0.9  # NOT in sync!
```

## Migration Guide

If you have existing code with class-level attributes, migrate to config schemas:

### Before (Broken)
```python
class MyAgent(BaseAgent):
    temperature: float = 0.7  # ❌ Class-level
    max_tokens: int = 1000    # ❌ Class-level
    config_schema = BaseAgentConfig
```

### After (Correct)
```python
class MyAgentConfig(BaseAgentConfig):
    temperature: float = 0.7  # ✅ In config schema
    max_tokens: int = 1000    # ✅ In config schema

class MyAgent(BaseAgent):
    config_schema = MyAgentConfig  # ✅ No class-level attributes
```

## Best Practices

### DO ✅

- Define all fields in config schemas, not as class attributes
- Create separate config instances for each agent instance
- Use config object for all configuration management
- Let the metaclass create properties automatically
- Test reference semantics in your custom agents

### DON'T ❌

- Define class-level attributes that shadow config fields
- Share config objects between multiple instances
- Access config attributes before `super().__init__()`
- Override `_create_config_properties()` without understanding the implications
- Rely on the `debug` init parameter in AbstractBase (prefer setting it in config for consistency)

## Key Takeaways

1. **Properties are created at class definition time** by the metaclass
2. **Reference semantics** keep `agent.field` and `agent.config.field` in sync
3. **Class-level attributes block property creation** and break reference semantics
4. **Precedence order**: kwargs > config field > config default (except `debug` in AbstractBase)
5. **Each instance should have its own config object** for proper isolation
