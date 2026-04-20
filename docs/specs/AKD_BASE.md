# AKD Base Classes: Configuration and Attribute Reference Guide

## Overview

The AKD framework provides a single base class for all agents and tools:
- **`AbstractBase`**: concrete base with schema validation, streaming, sync `run()`, and config/metadata binding

Both `BaseAgent` and `BaseTool` descend from `AbstractBase`. Agents and tools inherit **reference semantics** for config attributes — changes to `agent.field` automatically update `agent.config.field` and vice versa.

Framework adapters that inherit a third-party class directly (e.g. `class PydanticAIAgent(pydantic_ai.Agent)`) can get the same reference semantics by mixing in `ConfigBindingMixin`. They can also declare structural conformance via the `AKDExecutable` / `AKDTool` protocols (see `akd/_base/protocols.py`).

## Attribute Binding

### How It Works

When you define a class that inherits from `AbstractBase` (directly or via `BaseAgent` / `BaseTool`):

```python
from akd.agents import BaseAgent
from akd.agents._base import BaseAgentConfig

class MyAgentConfig(BaseAgentConfig):
    temperature: float = 0.7
    max_tokens: int = 1000

class MyAgent(BaseAgent):
    config_schema = MyAgentConfig
```

`ConfigBindingMixin.__init_subclass__` (inherited by `AbstractBase`) **automatically creates properties** at class definition time for every field in `config_schema`. These properties provide reference semantics:

```python
agent = MyAgent(config=MyAgentConfig(temperature=0.5))

# Equivalent — both access the same underlying config field
agent.temperature = 0.9
agent.config.temperature = 0.9

# They always stay in sync
assert agent.temperature == agent.config.temperature  # Always True
```

### Why Reference Semantics?

1. **Single source of truth** — the config object is the canonical source
2. **Automatic synchronization** — no manual sync between `agent.field` and `agent.config.field`
3. **Consistent state** — impossible for the two to diverge
4. **Test isolation** — each instance has independent config state

### Collision-Aware Binding

`ConfigBindingMixin` skips creating a property when one of these is true:
- The field name is already set on the class body
- A framework parent class (e.g. `pydantic_ai.Agent`) already declares the attribute

This means mixing `ConfigBindingMixin` with a framework class that owns attributes like `retries` or `name` is safe — the framework attribute wins and isn't shadowed.

## Mechanism

### `__init_subclass__` (not metaclass)

Config property creation happens via `ConfigBindingMixin.__init_subclass__`, triggered automatically when any subclass is created:

```python
class ConfigBindingMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        config_cls = getattr(cls, "config_schema", None)
        if config_cls is None or not issubclass(config_cls, BaseModel):
            return

        for field_name in config_cls.model_fields:
            if field_name in cls.__dict__:
                continue
            existing = getattr(cls, field_name, None)
            if existing is not None and not isinstance(existing, property):
                continue
            setattr(cls, field_name, _make_config_property(field_name))
        # ...same for model_computed_fields, creating read-only properties
```

No metaclass is involved. This keeps `AbstractBase`'s MRO clean (`AbstractBase → Generic → ConfigBindingMixin → ABC → object`), avoiding the metaclass composition issues that prevent clean multi-inheritance with framework classes.

### Metadata Binding

`ConfigBindingMixin._bind_metadata()` runs at `__init__` time and populates:
- `self.name` — from `config.name` or `to_snake_case(cls.__name__)`
- `self.description` — from `config.description` or the class docstring, plus IO field hints appended if `config.io_hints=True` (default)

Union output schemas are handled uniformly — each branch is described with its docstring and fields under it.

### Schema Validation

`AbstractBase.__init__` validates that `input_schema` and `output_schema` are proper types at **instantiation time**, not class creation time. That means:
- Abstract or intermediate classes (e.g. `BaseAgent`, `AKDAgent`) can be defined without concrete schemas
- They cannot be instantiated directly — `TypeError` is raised
- Concrete subclasses that set valid schemas instantiate fine

No hardcoded class-name skip list is needed — the check naturally allows intermediate bases to exist without being usable.

## Attribute Resolution Precedence

When you access `agent.temperature`, the resolution order is:

### Standard Precedence

1. **kwargs passed to `__init__`** (highest priority)
   ```python
   agent = MyAgent(config=config, custom_param="value")
   # custom_param is set via kwargs during _post_init()
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
       temperature: float = 0.7

   agent = MyAgent(config=MyAgentConfig())
   agent.temperature  # Returns 0.7 (default)
   ```

### Special Case: `debug` Parameter

The `debug` init parameter takes precedence over the config field:

```python
config = MyAgentConfig(debug=False)
agent = MyAgent(config=config, debug=True)
agent.debug  # Returns True (init parameter wins)
```

**Precedence order for `debug`**:
1. `debug` init parameter (highest)
2. `config.debug` field value
3. Config schema default

### Precedence Summary Table

| Attribute Type | Init Param | kwargs | Config Field | Config Default | Class Attr |
|----------------|-----------|---------|--------------|----------------|------------|
| `debug` | **1 (wins)** | N/A | 2 | 3 | ❌ Blocks |
| Other attrs | N/A | **1 (wins)** | 2 | 3 | ❌ Blocks |

## Class-Level Attributes Warning

⚠️ Defining class-level attributes **blocks property creation** and breaks reference semantics.

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

`ConfigBindingMixin.__init_subclass__` skips a field if it's already defined at the class level:

```python
for field_name in config_cls.model_fields:
    if field_name in cls.__dict__:
        continue  # Skip — something already defined it
    # ...create property
```

This collision check is intentional — it protects framework-owned attributes (e.g. `pai.Agent.retries`) from being shadowed. But it also means user-declared class-level attributes block the property.

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
```

## Common Patterns and Examples

### Pattern 1: Standard Config Override

```python
from akd.agents import BaseAgent, BaseAgentConfig

class MyAgentConfig(BaseAgentConfig):
    temperature: float = 0.7
    max_tokens: int = 1000

# Default config
agent1 = MyAgent(config=MyAgentConfig())
agent1.temperature  # 0.7

# Custom config
agent2 = MyAgent(config=MyAgentConfig(temperature=0.9))
agent2.temperature  # 0.9

# Runtime mutation stays in sync
agent2.temperature = 0.5
agent2.config.temperature  # 0.5
```

### Pattern 2: kwargs Override

```python
agent = MyAgent(
    config=MyAgentConfig(temperature=0.7),
    custom_param="value",
)
agent.custom_param  # "value" — set via kwargs in _post_init()
```

### Pattern 3: Debug Override

```python
config = MyAgentConfig(debug=False)
agent = MyAgent(config=config, debug=True)
agent.debug         # True (init parameter wins)
agent.config.debug  # False (config unchanged)
```

### Pattern 4: Independent Instances

```python
# Best practice — separate config instances
agent1 = MyAgent(config=MyAgentConfig(temperature=0.7))
agent2 = MyAgent(config=MyAgentConfig(temperature=0.7))

agent1.temperature = 0.9
agent2.temperature  # 0.7 (independent)

# ⚠️ Avoid sharing config objects
shared = MyAgentConfig(temperature=0.7)
agent1 = MyAgent(config=shared)
agent2 = MyAgent(config=shared)

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
agent.double_temperature  # 1.0 (recomputed)

agent.double_temperature = 2.0  # ❌ AttributeError: can't set attribute
```

## Framework-Adapter Usage

Adapters that inherit a third-party framework class (e.g. `pydantic_ai.Agent`) can get the same binding by mixing in `ConfigBindingMixin`:

```python
from akd._base import ConfigBindingMixin
import pydantic_ai

class PydanticAIAgent(ConfigBindingMixin, pydantic_ai.Agent):
    config_schema = MyConfig
    input_schema = MyIn
    output_schema = MyOut

    async def arun(self, params, run_context=None, **kw): ...
    async def astream(self, params, run_context=None, **kw): ...
```

Since pai's `Agent.__init__` sets some attributes (`retries`, `name`, `model`) directly on the instance, `ConfigBindingMixin` detects the collision and doesn't shadow them with properties. Config fields that don't collide (e.g. `temperature`, `custom_setting`) get the usual `agent.x` → `agent.config.x` treatment.

`isinstance(agent, AKDExecutable)` passes without explicit Protocol inheritance — structural conformance does the work.

## Testing

### Test Isolation

Properties created at class definition time mean per-instance config mutation is safe:

```python
def test_one():
    agent = MyAgent(config=MyAgentConfig(temperature=0.7))
    agent.temperature = 0.9
    assert agent.temperature == 0.9

def test_two():
    agent = MyAgent(config=MyAgentConfig(temperature=0.7))
    # Not affected by test_one
    assert agent.temperature == 0.7  # ✅ passes
```

### Testing Reference Semantics

```python
def test_reference_semantics():
    config = MyAgentConfig(temperature=0.7)
    agent = MyAgent(config=config)

    agent.temperature = 0.9
    assert agent.config.temperature == 0.9

    agent.config.temperature = 0.5
    assert agent.temperature == 0.5

    assert agent.temperature == agent.config.temperature
```

### Testing Class-Level Attributes (Negative Test)

```python
def test_class_attr_blocks_property():
    class BrokenAgent(BaseAgent):
        temperature: float = 0.0  # class-level attribute blocks property
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

If you have existing code with class-level attributes that should be config fields, move them to the config schema:

### Before
```python
class MyAgent(BaseAgent):
    temperature: float = 0.7  # ❌ class-level
    max_tokens: int = 1000    # ❌ class-level
    config_schema = BaseAgentConfig
```

### After
```python
class MyAgentConfig(BaseAgentConfig):
    temperature: float = 0.7  # ✅ in config schema
    max_tokens: int = 1000

class MyAgent(BaseAgent):
    config_schema = MyAgentConfig  # ✅ no class-level attributes
```

## Best Practices

### DO ✅

- Define all fields in config schemas, not as class attributes
- Create separate config instances for each agent instance
- Use config object for all configuration management
- Let `ConfigBindingMixin` create properties automatically
- Test reference semantics in your custom agents

### DON'T ❌

- Define class-level attributes that shadow config fields
- Share config objects between multiple instances
- Access config attributes before `super().__init__()`
- Rely on the `debug` init parameter for long-term state (prefer setting it in config for consistency)

## Key Takeaways

1. **Properties are created at class definition time** by `ConfigBindingMixin.__init_subclass__`
2. **Reference semantics** keep `agent.field` and `agent.config.field` in sync
3. **Class-level attributes block property creation** and break reference semantics
4. **Schema validation fires at instantiation**, not class creation — abstract classes can be defined without schemas
5. **Precedence order**: kwargs > config field > config default (except `debug`, where init parameter wins)
6. **Each instance should have its own config object** for proper isolation
7. **Framework adapters** can mix in `ConfigBindingMixin` directly and satisfy `AKDExecutable` structurally without inheriting `BaseAgent`
