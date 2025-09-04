import asyncio
from dataclasses import dataclass
from typing import List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.dag.nodes import (
    BinaryJudgementNode,
    VerdictNode,
    construct_node_verbose_log,
    decrement_indegree,
)
from deepeval.metrics.dag.schema import BinaryJudgementVerdict
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from loguru import logger

from akd.tools.granite_guardian_tool import (
    GraniteGuardianInputSchema,
    GraniteGuardianOutputSchema,
    GraniteGuardianTool,
    RiskDefinition,
)


@dataclass
class GraniteGuardianBinaryNode(BinaryJudgementNode):
    """
    Granite Guardian Binary Judgement Node.
    The field 'criteria' has been repurposed to be a GG risk definition
    and the LLMTestCase object is checked according to that definition.

    The LLMTestCase can contain LLMTestCaseParams.RETRIEVAL_CONTEXT,
    LLMTestCaseParams.INPUT and LLMTestCaseParams.ACTUAL_OUTPUT and these
    are passed as 'context', 'query' and 'response' to the Granite Guardian
    tool respectively.
    """

    criteria: RiskDefinition
    children: List[VerdictNode]
    evaluation_params: Optional[List[LLMTestCaseParams]] = None
    granite_guardian_tool: GraniteGuardianTool = None  # I only made this a default field to avoid problems with dataclass inheritance

    def __hash__(self):
        return id(self)

    def _get_guardian_input(
        self,
        test_case: LLMTestCase,
    ) -> GraniteGuardianInputSchema:
        # Gather input data from evaluation_params
        input_kwargs = {}
        for param in self.evaluation_params or []:
            val = getattr(test_case, param.value)
            if param == LLMTestCaseParams.RETRIEVAL_CONTEXT:
                input_kwargs["context"] = val
            elif param == LLMTestCaseParams.INPUT:
                input_kwargs["query"] = val
            elif param == LLMTestCaseParams.ACTUAL_OUTPUT:
                input_kwargs["response"] = val
        input_kwargs["risk_type"] = self.criteria
        return GraniteGuardianInputSchema(**input_kwargs)

    async def _a_execute(
        self,
        metric: BaseMetric,
        test_case: LLMTestCase,
        depth: int,
    ):
        self._depth = max(0, self._depth, depth)
        decrement_indegree(self)
        if self._indegree > 0:
            return

        guardian_input = self._get_guardian_input(test_case)
        guardian_output: GraniteGuardianOutputSchema = (
            await self.granite_guardian_tool.arun(guardian_input)
        )

        has_risk = guardian_output.risk_results[0].get("is_risky")
        if has_risk is None:
            skipped = guardian_output.risk_results[0].get("skipped", False)
            reason = guardian_output.risk_results[0].get("reason", "Unknown reason.")

            if skipped:
                error_message = f"Operation skipped: {reason}"
            else:
                error_message = "Missing 'is_risky' in Guardian output, but instance not skipped by Guardian tool."

            logger.error(
                f"[GuardianDeepeval] Guardian Tool unexpected output error error: {error_message}",
            )
            return {"error": str(error_message)}

        else:
            pass

        self._verdict = BinaryJudgementVerdict(
            verdict=has_risk,
            reason=f"Violates {self.criteria}",
        )
        metric._verbose_steps.append(
            construct_node_verbose_log(self, self._depth),
        )
        await asyncio.gather(
            *(
                child._a_execute(
                    metric=metric,
                    test_case=test_case,
                    depth=self._depth + 1,
                )
                for child in self.children
            ),
        )
