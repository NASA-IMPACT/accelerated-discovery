"""Composite guardrail for combining multiple providers."""

import asyncio
from collections.abc import Sequence
from typing import Literal

from akd.guardrails._base import GuardrailInput, GuardrailOutput, GuardrailProtocol


class CompositeGuardrail(GuardrailProtocol):
    """Composite guardrail combining multiple providers with AND/OR logic.

    Mode semantics:
    - "all" (AND): Input must pass ALL guardrails. Merges all detected risks.
    - "any" (OR): Input must pass at least ONE guardrail. Only reports risks if all fail.

    Usage:
        granite = GraniteGuardianTool()
        risk = RiskAgent()

        # AND: both must pass
        combined = CompositeGuardrail(guardrails=[granite, risk], mode="all")
        output = await combined.acheck(input)

        # OR: at least one must pass
        combined = CompositeGuardrail(guardrails=[granite, risk], mode="any")
    """

    def __init__(
        self,
        guardrails: Sequence[GuardrailProtocol],
        mode: Literal["all", "any"] = "all",
        parallel: bool = True,
    ):
        """Initialize composite guardrail.

        Args:
            guardrails: Sequence of guardrail providers to combine.
            mode: Combination mode - "all" (AND) or "any" (OR).
            parallel: Whether to run guardrails in parallel. Default True.
        """
        self.guardrails = list(guardrails)
        self.mode = mode
        self.parallel = parallel

    async def acheck(self, params: GuardrailInput) -> GuardrailOutput:
        """Run all guardrails and merge results based on mode."""
        if self.parallel:
            results = await asyncio.gather(*[g.acheck(params) for g in self.guardrails])
        else:
            results = [await g.acheck(params) for g in self.guardrails]

        return self._merge_results(list(results))

    def check(self, params: GuardrailInput) -> GuardrailOutput:
        """Sync version - runs acheck in event loop."""
        return asyncio.run(self.acheck(params))

    def _merge_results(self, results: list[GuardrailOutput]) -> GuardrailOutput:
        """Merge results from multiple guardrails based on mode."""
        return self._merge_results_all(results) if self.mode == "all" else self._merge_results_any(results)

    def _merge_results_all(self, results: list[GuardrailOutput]) -> GuardrailOutput:
        """AND mode: merge all detected risks from all guardrails."""
        providers = [r.provider or "unknown" for r in results]
        provider_str = f"CompositeGuardrail(all)[{', '.join(providers)}]"

        all_detected = []
        all_risk_results = {}
        for r in results:
            all_detected.extend(r.detected_risks)
            all_risk_results.update(r.risk_results or {})

        return GuardrailOutput(
            detected_risks=all_detected,
            risk_results=all_risk_results,
            provider=provider_str,
            extra={"mode": "all", "sub_results": [r.model_dump() for r in results]},
        )

    def _merge_results_any(self, results: list[GuardrailOutput]) -> GuardrailOutput:
        """OR mode: pass if any guardrail passes, only report risks if all fail."""
        providers = [r.provider or "unknown" for r in results]
        provider_str = f"CompositeGuardrail(any)[{', '.join(providers)}]"

        any_passed = any(r.passed for r in results)

        if any_passed:
            return GuardrailOutput(
                detected_risks=[],
                provider=provider_str,
                extra={"mode": "any", "sub_results": [r.model_dump() for r in results]},
            )

        # All failed - merge all detected risks
        all_detected = []
        all_risk_results = {}
        for r in results:
            all_detected.extend(r.detected_risks)
            all_risk_results.update(r.risk_results or {})

        return GuardrailOutput(
            detected_risks=all_detected,
            risk_results=all_risk_results,
            provider=provider_str,
            extra={"mode": "any", "sub_results": [r.model_dump() for r in results]},
        )
