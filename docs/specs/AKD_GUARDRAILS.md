# AKD Guardrails System

A comprehensive guide to the `akd.guardrails` system for implementing safety checks, risk detection, and content validation in the Accelerated Knowledge Discovery framework.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Risk Categories](#risk-categories)
4. [GuardrailProtocol & IO Schemas](#guardrailprotocol--io-schemas)
5. [Providers](#providers)
   - [GraniteGuardianTool](#graniteguardiantool)
   - [MultiRiskGraniteGuardianTool](#multiriskgraniteguardiantool)
   - [RiskAgent](#riskagent)
6. [Composing Guardrails](#composing-guardrails)
7. [Decorators](#decorators)
8. [Exception Handling](#exception-handling)
9. [Creating Custom Guardrails](#creating-custom-guardrails)
10. [Examples](#examples)

---

## Overview

The AKD Guardrails system provides a unified, composable framework for validating inputs and outputs of agents and tools. It enables:

- **Risk Detection**: Identify harmful, biased, or unsafe content
- **Scientific Integrity**: Detect hallucinations, attribution issues, and overgeneralization
- **Content Validation**: Ensure outputs meet quality and safety standards
- **Flexible Composition**: Combine multiple guardrails with AND/OR/sequential logic
- **Non-invasive Integration**: Apply guardrails via decorators without modifying agent code

### Core Principles

1. **Protocol-Based Design**: Any class implementing `check()` and `acheck()` can be a guardrail
2. **Composability**: Guardrails can be combined using operators (`&`, `|`, `>>`)
3. **Transparency**: All results include detected risks, per-risk details, and provider metadata
4. **Extensibility**: Easy to create custom guardrails and risk categories

---

## Architecture

```
                    GuardrailProtocol
                          |
        +-----------------+-----------------+
        |                 |                 |
 GraniteGuardianTool  RiskAgent    CompositeGuardrail
        |                 |                 |
        +-----------------+-----------------+
                          |
                  GuardrailOperatorMixin
                    (&, |, >> operators)
                          |
        +-----------------+-----------------+
        |                                   |
   @guardrail                        apply_guardrails()
   (decorator)                       (runtime wrapper)
```

### Module Structure

```
akd/guardrails/
├── __init__.py                 # Public API exports
├── _base.py                    # Protocol, IO schemas, mixins
├── decorators.py               # @guardrail, apply_guardrails()
├── utils.py                    # Text extraction utilities
├── categories/
│   ├── _base.py                # RiskCategory, RiskMetadata
│   ├── granite.py              # GraniteRiskCategory, GraniteHarmCategory
│   ├── atlas.py                # AtlasRiskCategory, ScienceRiskCategory (YAML)
│   ├── risk_atlas_data.yaml    # IBM Risk Atlas categories
│   └── science_lit_risks.yaml  # Scientific discovery risks
└── providers/
    ├── granite_guardian.py     # GraniteGuardianTool, MultiRiskGraniteGuardianTool
    ├── risk_agent.py           # RiskAgent (DAG-based evaluation)
    └── composite.py            # CompositeGuardrail
```

---

## Risk Categories

Risk categories are typed enums with inline metadata. All categories inherit from `RiskCategory`, a `StrEnum` subclass.

### RiskMetadata

Each risk category member carries metadata:

```python
from akd.guardrails import RiskMetadata

metadata = RiskMetadata(
    description="Human-readable description of the risk",
    severity="low" | "normal" | "high",
    name="Display name (optional)",
    extra={"custom_key": "value"}  # Extensible metadata
)
```

### Creating Risk Categories

```python
from akd.guardrails.categories._base import RiskCategory, RiskMetadata

class MyRiskCategory(RiskCategory):
    """Custom risk categories."""

    SPAM = ("spam", RiskMetadata(description="Spam content", severity="low"))
    HARMFUL = ("harmful", RiskMetadata(description="Harmful content", severity="high"))
    BIAS = ("bias", RiskMetadata(description="Biased content"))  # Uses defaults
```

### Available Risk Categories

#### GraniteRiskCategory

Input categories for Granite Guardian single-risk mode:

| Category | Value | Description | Severity |
|----------|-------|-------------|----------|
| `HARM` | `harm` | General harmful content | high |
| `SOCIAL_BIAS` | `social_bias` | Socially biased content | normal |
| `PROFANITY` | `profanity` | Profane language | normal |
| `SEXUAL_CONTENT` | `sexual_content` | Sexual content | high |
| `UNETHICAL_BEHAVIOR` | `unethical_behavior` | Unethical behavior | normal |
| `VIOLENCE` | `violence` | Violence-related content | high |
| `JAILBREAK` | `jailbreak` | Jailbreak attempts | high |
| `GROUNDEDNESS` | `groundedness` | Response grounded in context | normal |
| `RELEVANCE` | `relevance` | Content relevance to query | normal |
| `ANSWER_RELEVANCE` | `answer_relevance` | Answer relevance to question | normal |

```python
from akd.guardrails import GraniteRiskCategory

# Access category
risk = GraniteRiskCategory.HARM
print(risk.value)        # "harm"
print(risk.description)  # "General harmful content"
print(risk.severity)     # "high"
```

#### GraniteHarmCategory

Output categories from Granite Guardian multi-harm model (Title Case values):

| Category | Value | Maps To |
|----------|-------|---------|
| `SOCIAL_BIAS` | `Social Bias` | `social_bias` |
| `JAILBREAKING` | `Jailbreaking` | `jailbreak` |
| `VIOLENCE` | `Violence` | `violence` |
| `PROFANITY` | `Profanity` | `profanity` |
| `SEXUAL_CONTENT` | `Sexual Content` | `sexual_content` |
| `UNETHICAL_BEHAVIOR` | `Unethical Behavior` | `unethical_behavior` |
| `HARMFUL` | `Harmful` | `harm` |
| `NOT_HARMFUL_PROMPT` | `Not harmful prompt` | - |
| `NOT_HARMFUL_RESPONSE` | `Not harmful response` | - |

#### ScienceRiskCategory

YAML-loaded categories for scientific discovery workflows. Key categories include:

| Category | Description |
|----------|-------------|
| `HALLUCINATION_IDENTIFICATION` | Detecting factually fabricated content |
| `ATTRIBUTION` | Ensuring proper source attribution |
| `CONSISTENCY` | Internal coherence within responses |
| `UNCERTAINTY_IDENTIFICATION` | Signaling low confidence |
| `OVERGENERALIZATION` | Removing essential qualifiers |
| `STATIC_KNOWLEDGE` | Failure to incorporate newer findings |
| `OUTDATED_CONFIDENCE` | Presenting outdated info with false certainty |
| `POSITIVITY_BIAS` | Overly favorable presentation of findings |
| `MULTIDISCIPLINARY_FAILURE` | Failure to integrate cross-domain knowledge |

```python
from akd.guardrails import ScienceRiskCategory

# Check if loaded (graceful degradation if YAML missing)
if ScienceRiskCategory:
    risk = ScienceRiskCategory.HALLUCINATION_IDENTIFICATION
    print(risk.description)
```

#### AtlasRiskCategory

YAML-loaded categories from IBM's AI Risk Atlas. Categories are dynamically built at module import time.

---

## GuardrailProtocol & IO Schemas

### GuardrailProtocol

The core interface all guardrail providers must implement:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class GuardrailProtocol(Protocol):
    """Protocol for guardrail implementations."""

    def check(self, params: GuardrailInput) -> GuardrailOutput:
        """Synchronous guardrail check."""
        ...

    async def acheck(self, params: GuardrailInput) -> GuardrailOutput:
        """Asynchronous guardrail check."""
        ...
```

Any class implementing these methods can be used as a guardrail and composed with operators.

### GuardrailInput

Unified input for all guardrail providers:

```python
from akd.guardrails import GuardrailInput, GraniteRiskCategory

# Simple usage
input = GuardrailInput(
    content="Text to check for risks",
    context="Optional context (prior conversation, RAG docs)",
    risk_categories=[GraniteRiskCategory.HARM, GraniteRiskCategory.JAILBREAK],
)

# Multi-turn conversation
input = GuardrailInput.from_multi_turn(
    inputs=["What is X?", "Tell me more about X"],      # User messages
    outputs=["X is...", "X is also known for..."],      # Model responses
    risk_categories=[ScienceRiskCategory.HALLUCINATION_IDENTIFICATION],
    additional_context="Agent: Scientific Literature Search Agent",
)
```

**Fields:**
- `content: str` - Content to check for risks
- `context: str | None` - Optional context (prior conversation, RAG docs)
- `risk_categories: Sequence[RiskCategory]` - Risk categories to check (empty = provider defaults)

### GuardrailOutput

Unified output from guardrail checks:

```python
from akd.guardrails import GuardrailOutput

# Output structure
output = GuardrailOutput(
    detected_risks=[GraniteRiskCategory.HARM],
    risk_results={
        GraniteRiskCategory.HARM: {
            "is_risky": True,
            "confidence": 0.95,
            "raw_response": "yes",
        }
    },
    provider="GraniteGuardianTool",
    extra={"model": "granite3-guardian:8b"},
)

# Computed properties
print(output.passed)    # False (has detected risks)
print(output.summary)   # "Risks detected: harm"
```

**Fields:**
- `detected_risks: Sequence[RiskCategory]` - Risk categories that were detected
- `risk_results: dict[RiskCategory, dict]` - Per-risk evaluation details
- `provider: str | None` - Name of the guardrail provider
- `extra: dict[str, Any]` - Provider-specific data

**Computed Properties:**
- `passed: bool` - `True` if no risks detected
- `summary: str` - Human-readable summary

### GuardrailOperatorMixin

Mixin that adds composition operators to guardrail implementations:

```python
class GuardrailOperatorMixin:
    """Adds &, |, >> operators to guardrails."""

    def __and__(self, other) -> CompositeGuardrail:
        """g1 & g2 -> ALL mode (both must pass)"""

    def __or__(self, other) -> CompositeGuardrail:
        """g1 | g2 -> ANY mode (at least one passes)"""

    def __rshift__(self, other) -> CompositeGuardrail:
        """g1 >> g2 -> FAIL_FAST mode (sequential, stop on first fail)"""
```

### RiskCategoryValidationMixin

Validates that input risk categories match the provider's supported types:

```python
class RiskCategoryValidationMixin:
    """Validates category types against config declaration."""

    def _validate_category_types(self, categories: Sequence[RiskCategory]) -> None:
        """Raises TypeError if categories don't match supported type."""
```

Set `validate_categories=False` in config to disable validation.

---

## Providers

### GraniteGuardianTool

Single-risk detection using IBM Granite Guardian models via Ollama.

```python
from akd.guardrails.providers import GraniteGuardianTool
from akd.guardrails.providers.granite_guardian import (
    GraniteGuardianToolConfig,
    GuardianModelID,
)

# Basic usage
tool = GraniteGuardianTool()
output = await tool.acheck(GuardrailInput(content="How do I hack a system?"))
print(output.passed)  # False

# With custom config
config = GraniteGuardianToolConfig(
    ollama_base_url="http://localhost:11434",
    model=GuardianModelID.GUARDIAN_8B,
    max_concurrency=3,
    timeout=60.0,
    risk_categories=[
        GraniteRiskCategory.HARM,
        GraniteRiskCategory.JAILBREAK,
    ],
)
tool = GraniteGuardianTool(config=config)
```

**Configuration Options:**
- `ollama_base_url: HttpUrl` - Ollama server URL (default: `http://localhost:11434`)
- `model: GuardianModelID` - Model to use (default: `GUARDIAN_8B`)
- `max_concurrency: int` - Max concurrent requests (default: 3)
- `timeout: float` - HTTP timeout in seconds (default: 60.0)
- `validate_categories: bool` - Validate category types (default: True)
- `risk_categories: list[GraniteRiskCategory]` - Categories to check (default: all)

**Supported Models:**
- `GuardianModelID.GUARDIAN_2B` - `granite3-guardian:2b`
- `GuardianModelID.GUARDIAN_8B` - `granite3-guardian:8b`
- `GuardianModelID.GUARDIAN_3_3_8B` - `ibm/granite3.3-guardian:8b`
- `GuardianModelID.GUARDIAN_3_2_5B_MULTI_HARM` - Multi-harm model

### MultiRiskGraniteGuardianTool

Detects ALL risk categories in a single model call:

```python
from akd.guardrails.providers import MultiRiskGraniteGuardianTool
from akd.guardrails.providers.granite_guardian import MultiRiskGraniteGuardianToolConfig

config = MultiRiskGraniteGuardianToolConfig(
    risk_categories=[
        GraniteHarmCategory.VIOLENCE,
        GraniteHarmCategory.HARMFUL,
    ],
)
tool = MultiRiskGraniteGuardianTool(config=config)

output = await tool.acheck(GuardrailInput(content="..."))
print(output.detected_risks)  # List of GraniteHarmCategory
```

**Key Differences from GraniteGuardianTool:**
- Uses multi-harm model for single-call detection
- Returns `GraniteHarmCategory` (Title Case) instead of `GraniteRiskCategory`
- More efficient for checking multiple risks

### RiskAgent

LLM-based guardrail using DeepEval's DAGMetric for hierarchical evaluation:

```python
from akd.guardrails.providers import RiskAgent
from akd.guardrails.providers.risk_agent import RiskAgentConfig

# Basic usage with default science risks
agent = RiskAgent()
output = await agent.acheck(GuardrailInput(
    content="The capital of France is Berlin.",
    context="What is the capital of France?",
))
print(output.passed)        # False
print(output.detected_risks)  # [ScienceRiskCategory.HALLUCINATION_IDENTIFICATION, ...]

# With custom config
config = RiskAgentConfig(
    pass_threshold=0.85,
    risk_categories=[
        ScienceRiskCategory.HALLUCINATION_IDENTIFICATION,
        ScienceRiskCategory.ATTRIBUTION,
    ],
    risk_weights={
        "hallucination_identification": 1.0,
        "attribution": 0.8,
    },
    include_dag_metric=True,  # Include full DAGMetric in output
)
agent = RiskAgent(config=config, debug=True)
```

**How It Works:**

1. **Criteria Generation**: For each risk category, generates evaluation criteria with importance levels (HIGH, MEDIUM, LOW)
2. **DAG Construction**: Builds hierarchical evaluation graph
   - HIGH criteria: All must pass
   - MEDIUM criteria: At least half must pass
   - LOW criteria: Tiebreaker only
3. **Evaluation**: Measures content against DAG metric
4. **Scoring**: Returns score 0-1 based on weighted pass ratio

**Configuration Options:**
- `pass_threshold: float` - Score threshold for passing (default: 0.9)
- `risk_categories: list[RiskCategory]` - Categories to evaluate (default: 5 science risks)
- `risk_weights: dict[str, float] | None` - Per-risk weight overrides
- `include_dag_metric: bool` - Include full DAGMetric in output.extra (default: False)
- `validate_categories: bool` - Validate category types (default: True)

**Output Extra Fields:**
- `score: float` - Normalized score (0-1)
- `raw_score: float` - DAGMetric score (0-10)
- `reason: str` - DAGMetric reasoning
- `verbose_logs: str` - Detailed evaluation logs
- `dag_metric: DAGMetric` - Full metric object (if `include_dag_metric=True`)

---

## Composing Guardrails

### CompositeGuardrail

Combine multiple guardrails with different execution modes:

```python
from akd.guardrails import CompositeGuardrail, CompositeGuardrailMode
from akd.guardrails.providers import GraniteGuardianTool, RiskAgent

granite = GraniteGuardianTool()
risk = RiskAgent()

# AND mode: both must pass (parallel execution)
combined = CompositeGuardrail(
    guardrails=[granite, risk],
    mode=CompositeGuardrailMode.ALL,
    parallel=True,
)

# OR mode: at least one must pass
combined = CompositeGuardrail(
    guardrails=[granite, risk],
    mode=CompositeGuardrailMode.ANY,
)

# FAIL_FAST mode: sequential, stop on first failure
combined = CompositeGuardrail(
    guardrails=[granite, risk],
    mode=CompositeGuardrailMode.FAIL_FAST,
)
```

### Operator Composition

More readable composition using operators:

```python
granite = GraniteGuardianTool()
risk = RiskAgent()

# AND: both must pass
combined = granite & risk

# OR: at least one must pass
combined = granite | risk

# FAIL_FAST: sequential, cheap check first
combined = granite >> risk

# Complex composition
combined = (granite >> risk) | fallback_guardrail
```

### Composition Modes

| Mode | Operator | Execution | Pass Condition | Risk Reporting |
|------|----------|-----------|----------------|----------------|
| ALL | `&` | Parallel | All must pass | Merges all risks |
| ANY | `|` | Parallel | Any one passes | Reports only if all fail |
| FAIL_FAST | `>>` | Sequential | All must pass | Returns first failure |

### Automatic Flattening

Nested same-mode compositions are automatically flattened:

```python
# These are equivalent:
(g1 & g2) & g3  # Flattened to CompositeGuardrail([g1, g2, g3], ALL)
CompositeGuardrail([g1, g2, g3], mode=CompositeGuardrailMode.ALL)
```

---

## Decorators

### @guardrail Decorator

Apply guardrails to agent/tool classes at definition time:

```python
from akd.guardrails import guardrail
from akd.guardrails.providers import GraniteGuardianTool, RiskAgent
from akd.agents import BaseAgent

granite = GraniteGuardianTool()
risk = RiskAgent()

@guardrail(
    input_guardrail=granite >> risk,      # Fail-fast: granite first
    output_guardrail=granite,
    fail_on_input_risk=True,               # Raise exception on input risk
    fail_on_output_risk=False,             # Just log output risks
    input_fields=["query", "message"],     # Fields to extract from input
    output_fields=["answer", "response"],  # Fields to extract from output
    debug=True,
)
class MyAgent(BaseAgent):
    """Agent with guardrails."""

    async def _arun(self, params):
        # Your implementation
        return output
```

**Parameters:**
- `input_guardrail: GuardrailProtocol | None` - Guardrail for input checking
- `output_guardrail: GuardrailProtocol | None` - Guardrail for output checking
- `fail_on_input_risk: bool` - Raise `InputGuardrailTriggered` on input risk (default: False)
- `fail_on_output_risk: bool` - Raise `OutputGuardrailTriggered` on output risk (default: False)
- `input_fields: list[str] | None` - Fields to extract text from input
- `output_fields: list[str] | None` - Fields to extract text from output
- `debug: bool` - Enable debug logging (default: False)

**Behavior:**
1. Intercepts `_arun()` method
2. Pre-execution: Runs input guardrail on extracted text
3. Post-execution: Runs output guardrail with output + context
4. Wraps response with `GuardrailResultMixin`

### GuardrailResultMixin

Added to response models by `@guardrail`:

```python
class GuardrailResultMixin(BaseModel):
    input_guardrail_result: GuardrailOutput | None = None
    output_guardrail_result: GuardrailOutput | None = None

    @computed_field
    @property
    def guardrails_passed(self) -> bool:
        """True if no guardrails were triggered."""
```

### apply_guardrails() Function

Apply guardrails to existing instances at runtime:

```python
from akd.guardrails import apply_guardrails
from akd.guardrails.providers import GraniteGuardianTool

# Create agent without guardrails
agent = MyAgent()

# Apply guardrails to existing instance
guarded = apply_guardrails(
    agent,
    input_guardrail=GraniteGuardianTool(),
    output_guardrail=None,
    fail_on_input_risk=True,
    debug=True,
)

# Use guarded instance
result = await guarded.arun(params)
```

**Benefits:**
- Non-invasive: Works with existing instances
- State preservation: Copies internal state from original
- Flexible: Apply different guardrails to different instances

---

## Exception Handling

```python
from akd.errors import (
    InputGuardrailTriggered,
    OutputGuardrailTriggered,
    GuardrailError,
)

# Handle guardrail exceptions
try:
    result = await agent.arun(params)
except InputGuardrailTriggered as e:
    print(f"Input validation failed: {e.output.summary}")
    print(f"Detected risks: {e.output.detected_risks}")
except OutputGuardrailTriggered as e:
    print(f"Output validation failed: {e.output.summary}")

# Handle composition errors
try:
    bad_composition = guardrail & "not a guardrail"
except GuardrailError as e:
    print(f"Invalid composition: {e}")
```

**Exception Types:**
- `InputGuardrailTriggered` - Input failed validation with `fail_on_input_risk=True`
- `OutputGuardrailTriggered` - Output failed validation with `fail_on_output_risk=True`
- `GuardrailError` - Invalid guardrail composition (not implementing protocol)

---

## Creating Custom Guardrails

### Implementing GuardrailProtocol

Any class implementing `check()` and `acheck()` can be a guardrail:

```python
from akd.guardrails import (
    GuardrailInput,
    GuardrailOutput,
    GuardrailProtocol,
    GuardrailOperatorMixin,
)

class MyCustomGuardrail(GuardrailOperatorMixin, GuardrailProtocol):
    """Custom guardrail implementation."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    async def acheck(self, params: GuardrailInput) -> GuardrailOutput:
        """Async guardrail check."""
        # Your validation logic
        content = params.content
        context = params.context
        categories = params.risk_categories

        detected = []
        risk_results = {}

        for category in categories:
            is_risky = await self._check_category(content, category)
            if is_risky:
                detected.append(category)
            risk_results[category] = {"is_risky": is_risky}

        return GuardrailOutput(
            detected_risks=detected,
            risk_results=risk_results,
            provider=self.__class__.__name__,
            extra={"threshold": self.threshold},
        )

    def check(self, params: GuardrailInput) -> GuardrailOutput:
        """Sync guardrail check."""
        import asyncio
        return asyncio.run(self.acheck(params))

    async def _check_category(self, content: str, category) -> bool:
        """Check single category - implement your logic."""
        # Your implementation
        return False
```

### Creating Custom Risk Categories

```python
from akd.guardrails.categories._base import RiskCategory, RiskMetadata

class DomainRiskCategory(RiskCategory):
    """Domain-specific risk categories."""

    DATA_LEAKAGE = (
        "data_leakage",
        RiskMetadata(
            description="Potential exposure of sensitive data",
            severity="high",
            extra={"compliance": ["GDPR", "HIPAA"]},
        ),
    )
    MISINFORMATION = (
        "misinformation",
        RiskMetadata(
            description="Factually incorrect information",
            severity="high",
        ),
    )
    LOW_QUALITY = (
        "low_quality",
        RiskMetadata(
            description="Poor quality or incomplete response",
            severity="low",
        ),
    )
```

### YAML-Based Category Loading

For dynamic category loading:

```python
from pathlib import Path
from akd.guardrails.categories.atlas import build_risk_category_from_yaml

# YAML format:
# risks:
#   - id: "my-risk-id"
#     name: "My Risk"
#     description: "Description of the risk"
#     url: "https://..."
#     tag: "my-risk"

MyRiskCategory = build_risk_category_from_yaml(
    enum_name="MyRiskCategory",
    yaml_path=Path("/path/to/risks.yaml"),
)

# Use like any other category
risk = MyRiskCategory.MY_RISK_ID  # Converted from "my-risk-id"
```

---

## Examples

### Basic Guardrail Usage

```python
from akd.guardrails import GuardrailInput, GraniteRiskCategory
from akd.guardrails.providers import GraniteGuardianTool

# Create guardrail
tool = GraniteGuardianTool()

# Check content
output = await tool.acheck(GuardrailInput(
    content="How can I create a computer virus?",
    risk_categories=[GraniteRiskCategory.HARM, GraniteRiskCategory.JAILBREAK],
))

if output.passed:
    print("Content is safe")
else:
    print(f"Risks detected: {output.summary}")
    for risk, details in output.risk_results.items():
        print(f"  - {risk.value}: {details}")
```

### Composed Guardrails

```python
from akd.guardrails.providers import GraniteGuardianTool, RiskAgent

granite = GraniteGuardianTool()
risk = RiskAgent()

# Strategy: Cheap check first, expensive check only if cheap passes
composed = granite >> risk

output = await composed.acheck(GuardrailInput(content="..."))
print(f"Provider: {output.provider}")  # "CompositeGuardrail(fail_fast)[...]"
print(f"Passed: {output.passed}")

# Check if short-circuited
if output.extra.get("short_circuited"):
    print(f"Failed at index: {output.extra['failed_at_index']}")
```

### Agent with Decorators

```python
from akd.guardrails import guardrail
from akd.guardrails.providers import GraniteGuardianTool, RiskAgent
from akd.agents import BaseAgent
from pydantic import BaseModel

class QueryInput(BaseModel):
    query: str
    context: str | None = None

class QueryOutput(BaseModel):
    answer: str
    sources: list[str]

@guardrail(
    input_guardrail=GraniteGuardianTool(),
    output_guardrail=RiskAgent(),
    fail_on_input_risk=True,
    fail_on_output_risk=False,
)
class ResearchAgent(BaseAgent[QueryInput, QueryOutput]):
    """Research agent with guardrails."""

    input_schema = QueryInput
    output_schema = QueryOutput

    async def _arun(self, params: QueryInput) -> QueryOutput:
        # Your implementation
        return QueryOutput(
            answer="Research findings...",
            sources=["source1", "source2"],
        )

# Usage
agent = ResearchAgent()
try:
    result = await agent.arun(QueryInput(query="What is quantum computing?"))

    # Access guardrail results
    print(f"Guardrails passed: {result.guardrails_passed}")

    if result.input_guardrail_result:
        print(f"Input risks: {result.input_guardrail_result.detected_risks}")

    if result.output_guardrail_result:
        print(f"Output risks: {result.output_guardrail_result.detected_risks}")

except InputGuardrailTriggered as e:
    print(f"Input blocked: {e.output.summary}")
```

### Runtime Guardrail Application

```python
from akd.guardrails import apply_guardrails
from akd.guardrails.providers import GraniteGuardianTool

# Existing agent without guardrails
agent = MyAgent()

# Apply guardrails at runtime
guarded = apply_guardrails(
    agent,
    input_guardrail=GraniteGuardianTool(),
    fail_on_input_risk=True,
)

# Use guarded agent
result = await guarded.arun(params)
```

### Multi-Turn Conversation Check

```python
from akd.guardrails import GuardrailInput, ScienceRiskCategory
from akd.guardrails.providers import RiskAgent

# Build input from conversation history
input = GuardrailInput.from_multi_turn(
    inputs=[
        "What causes climate change?",
        "Can you provide more details about CO2 levels?",
    ],
    outputs=[
        "Climate change is primarily caused by greenhouse gas emissions...",
        "CO2 levels have risen from 280 ppm to over 420 ppm...",
    ],
    risk_categories=[
        ScienceRiskCategory.HALLUCINATION_IDENTIFICATION,
        ScienceRiskCategory.ATTRIBUTION,
    ],
    additional_context="Agent: Climate Science Research Assistant",
)

agent = RiskAgent()
output = await agent.acheck(input)
print(f"Scientific integrity check: {output.passed}")
```

### Custom Guardrail Implementation

```python
from akd.guardrails import (
    GuardrailInput,
    GuardrailOutput,
    GuardrailOperatorMixin,
    GuardrailProtocol,
    RiskCategory,
    RiskMetadata,
)

class ContentLengthRisk(RiskCategory):
    TOO_SHORT = ("too_short", RiskMetadata(description="Response too short", severity="low"))
    TOO_LONG = ("too_long", RiskMetadata(description="Response too long", severity="low"))

class ContentLengthGuardrail(GuardrailOperatorMixin, GuardrailProtocol):
    """Validates content length."""

    def __init__(self, min_length: int = 10, max_length: int = 10000):
        self.min_length = min_length
        self.max_length = max_length

    async def acheck(self, params: GuardrailInput) -> GuardrailOutput:
        content = params.content
        detected = []
        risk_results = {}

        if len(content) < self.min_length:
            detected.append(ContentLengthRisk.TOO_SHORT)
            risk_results[ContentLengthRisk.TOO_SHORT] = {
                "length": len(content),
                "min_required": self.min_length,
            }

        if len(content) > self.max_length:
            detected.append(ContentLengthRisk.TOO_LONG)
            risk_results[ContentLengthRisk.TOO_LONG] = {
                "length": len(content),
                "max_allowed": self.max_length,
            }

        return GuardrailOutput(
            detected_risks=detected,
            risk_results=risk_results,
            provider="ContentLengthGuardrail",
        )

    def check(self, params: GuardrailInput) -> GuardrailOutput:
        import asyncio
        return asyncio.run(self.acheck(params))

# Use with composition
granite = GraniteGuardianTool()
length = ContentLengthGuardrail(min_length=50, max_length=5000)

# Combine guardrails
combined = length >> granite  # Check length first (cheap), then content (expensive)
```

---

## Best Practices

1. **Use FAIL_FAST for efficiency**: Put cheap guardrails first with `>>` operator
2. **Set appropriate thresholds**: Adjust `pass_threshold` based on your use case
3. **Handle exceptions gracefully**: Catch `InputGuardrailTriggered` and `OutputGuardrailTriggered`
4. **Log warnings in production**: Use `fail_on_*_risk=False` with logging enabled
5. **Validate category types**: Keep `validate_categories=True` in development
6. **Use multi-turn for conversations**: Leverage `from_multi_turn()` for context-aware checks
7. **Compose thoughtfully**: Consider execution order and failure modes

---

## Summary

The AKD Guardrails system provides a flexible, composable framework for content validation:

- **GuardrailProtocol**: Implement `check()` and `acheck()` for custom guardrails
- **Providers**: GraniteGuardianTool (Ollama), RiskAgent (DeepEval DAG), CompositeGuardrail
- **Composition**: Use `&` (AND), `|` (OR), `>>` (FAIL_FAST) operators
- **Integration**: `@guardrail` decorator or `apply_guardrails()` function
- **Categories**: GraniteRiskCategory, ScienceRiskCategory, or custom enums

For questions or issues, refer to the source modules in `akd.guardrails`.
