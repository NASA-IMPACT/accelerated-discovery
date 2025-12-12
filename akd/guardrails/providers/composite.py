"""Composite guardrail for combining multiple providers."""

import asyncio
from collections.abc import Sequence
from enum import Enum

from akd.guardrails._base import GuardrailInput, GuardrailOutput, GuardrailProtocol


class CompositeGuardrailMode(Enum):
    """Mode for combining guardrail results."""

    ALL = "all"  # AND: all guardrails must pass
    ANY = "any"  # OR: at least one guardrail must pass
    FAIL_FAST = "fail_fast"  # Sequential, stop on first failure


class CompositeGuardrail(GuardrailProtocol):
    """Composite guardrail combining multiple providers with AND/OR/fail_fast logic.

    Mode semantics:
    - ALL (AND): Input must pass ALL guardrails. Merges all detected risks.
    - ANY (OR): Input must pass at least ONE guardrail. Only reports risks if all fail.
    - FAIL_FAST: Run sequentially, return immediately on first failure (short-circuit).

    Usage:
        granite = GraniteGuardianTool()
        risk = RiskAgent()

        # AND: both must pass (parallel)
        combined = CompositeGuardrail(guardrails=[granite, risk], mode=CompositeGuardrailMode.ALL)
        output = await combined.acheck(input)

        # OR: at least one must pass
        combined = CompositeGuardrail(guardrails=[granite, risk], mode=CompositeGuardrailMode.ANY)

        # Fail fast: stop on first failure (cheap guardrail first)
        combined = CompositeGuardrail(guardrails=[granite, risk], mode=CompositeGuardrailMode.FAIL_FAST)
    """

    def __init__(
        self,
        guardrails: Sequence[GuardrailProtocol],
        mode: CompositeGuardrailMode = CompositeGuardrailMode.ALL,
        parallel: bool = True,
    ):
        """Initialize composite guardrail.

        Args:
            guardrails: Sequence of guardrail providers to combine.
            mode: Combination mode (ALL, ANY, or FAIL_FAST).
            parallel: Whether to run guardrails in parallel (ignored for FAIL_FAST).

        Raises:
            TypeError: If mode is not a CompositeGuardrailMode enum.
            ValueError: If guardrails sequence is empty.
        """
        if not isinstance(mode, CompositeGuardrailMode):
            raise TypeError(f"mode must be CompositeGuardrailMode, got {type(mode).__name__}")
        if not guardrails:
            raise ValueError("guardrails sequence cannot be empty")

        self.guardrails = list(guardrails)
        self.mode = mode
        self.parallel = parallel

    async def acheck(self, params: GuardrailInput) -> GuardrailOutput:
        """Run all guardrails and merge results based on mode."""
        if self.mode == CompositeGuardrailMode.FAIL_FAST:
            return await self._run_fail_fast(params)

        if self.parallel:
            results = await asyncio.gather(*[g.acheck(params) for g in self.guardrails])
        else:
            results = [await g.acheck(params) for g in self.guardrails]

        return self._merge_results(list(results))

    def check(self, params: GuardrailInput) -> GuardrailOutput:
        """Sync version - runs acheck in event loop."""
        return asyncio.run(self.acheck(params))

    async def _run_fail_fast(self, params: GuardrailInput) -> GuardrailOutput:
        """Run guardrails sequentially, return on first failure."""
        executed_results: list[GuardrailOutput] = []

        for guardrail in self.guardrails:
            result = await guardrail.acheck(params)
            executed_results.append(result)

            if not result.passed:
                # Short-circuit on first failure
                return GuardrailOutput(
                    detected_risks=result.detected_risks,
                    risk_results=result.risk_results,
                    provider=f"CompositeGuardrail(fail_fast)[{result.provider or 'unknown'}]",
                    extra={
                        "mode": "fail_fast",
                        "short_circuited": True,
                        "failed_at_index": len(executed_results) - 1,
                        "result": result.model_dump(),
                    },
                )

        # All passed
        providers = [r.provider or "unknown" for r in executed_results]
        return GuardrailOutput(
            detected_risks=[],
            provider=f"CompositeGuardrail(fail_fast)[{', '.join(providers)}]",
            extra={
                "mode": "fail_fast",
                "short_circuited": False,
                "sub_results": [r.model_dump() for r in executed_results],
            },
        )

    def _merge_results(self, results: list[GuardrailOutput]) -> GuardrailOutput:
        """Merge results from multiple guardrails based on mode."""
        return (
            self._merge_results_all(results)
            if self.mode == CompositeGuardrailMode.ALL
            else self._merge_results_any(results)
        )

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
