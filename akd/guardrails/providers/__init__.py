"""Guardrail providers implementing GuardrailProtocol.

Providers are lazy-loaded so that importing ``akd.guardrails`` (or this
package) stays lightweight. ``RiskAgent`` in particular pulls in heavy
dependencies (deepeval, litellm) that are only needed when it is used.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from akd.guardrails.providers.composite import (
        CompositeGuardrail,
        CompositeGuardrailMode,
    )
    from akd.guardrails.providers.granite_guardian import (
        GraniteGuardianTool,
        GraniteGuardianToolConfig,
        GuardianModelID,
        MultiRiskGraniteGuardianTool,
        MultiRiskGraniteGuardianToolConfig,
        OllamaType,
    )
    from akd.guardrails.providers.risk_agent import (
        Criterion,
        CriterionImportance,
        RiskAgent,
        RiskAgentConfig,
    )

_LAZY_IMPORTS = {
    # Composite (lightweight)
    "CompositeGuardrail": ".composite",
    "CompositeGuardrailMode": ".composite",
    # Granite Guardian (lightweight - httpx only)
    "GraniteGuardianTool": ".granite_guardian",
    "GraniteGuardianToolConfig": ".granite_guardian",
    "GuardianModelID": ".granite_guardian",
    "MultiRiskGraniteGuardianTool": ".granite_guardian",
    "MultiRiskGraniteGuardianToolConfig": ".granite_guardian",
    "OllamaType": ".granite_guardian",
    # Risk Agent (heavy - deepeval, litellm)
    "RiskAgent": ".risk_agent",
    "RiskAgentConfig": ".risk_agent",
    "Criterion": ".risk_agent",
    "CriterionImportance": ".risk_agent",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        try:
            module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        except ModuleNotFoundError as e:
            if e.name and e.name.split(".")[0] == "deepeval":
                raise ModuleNotFoundError(
                    f"{name} requires the 'deepeval' package. Install it with: uv pip install 'akd[ml]'",
                ) from e
            raise
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


__all__ = [
    # Composite
    "CompositeGuardrail",
    "CompositeGuardrailMode",
    # Granite Guardian
    "GraniteGuardianTool",
    "GraniteGuardianToolConfig",
    "GuardianModelID",
    "MultiRiskGraniteGuardianTool",
    "MultiRiskGraniteGuardianToolConfig",
    "OllamaType",
    # Risk Agent
    "RiskAgent",
    "RiskAgentConfig",
    "Criterion",
    "CriterionImportance",
]
