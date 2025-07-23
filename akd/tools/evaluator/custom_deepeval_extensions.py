import asyncio
from dataclasses import dataclass
from enum import Enum
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
from pydantic import model_validator
from typing_extensions import Self

from akd.tools.granite_guardian_tool import (
    GraniteGuardianInputSchema,
    GraniteGuardianOutputSchema,
    GraniteGuardianTool,
    RiskDefinition,
)
from akd.tools.search import SearchResultItem, SearchToolOutputSchema


@dataclass
class LLMTestCaseGuardian(LLMTestCase):
    """
    LLMTestCase class for Granite Guardian Binary Judgement node.
    Contains additional field 'search_result'
    """

    input: Optional[str] = None
    actual_output: Optional[str] = None
    search_result: Optional[SearchResultItem] = None

    @model_validator(mode="after")
    def check_mutually_exclusive_fields(self) -> Self:
        if self.search_result:
            if self.input or self.actual_output:
                raise ValueError(
                    "If 'search_result' is provided, 'input' and 'actual_output' must not be provided.",
                )

        if not self.search_result:
            if not (self.input and self.actual_output):
                raise ValueError(
                    "If 'search_result' is not provided, both 'input' and 'actual_output' must be provided.",
                )

        return self


class LLMTestCaseParamsGuardian(Enum):
    """
    LLMTestCaseParams class for Granite Guardian Binary Judgement node.
    Contains additional field 'search_result'
    """

    INPUT = LLMTestCaseParams.INPUT.value
    ACTUAL_OUTPUT = LLMTestCaseParams.ACTUAL_OUTPUT.value
    EXPECTED_OUTPUT = LLMTestCaseParams.EXPECTED_OUTPUT.value
    CONTEXT = LLMTestCaseParams.CONTEXT.value
    RETRIEVAL_CONTEXT = LLMTestCaseParams.RETRIEVAL_CONTEXT.value
    TOOLS_CALLED = LLMTestCaseParams.TOOLS_CALLED.value
    EXPECTED_TOOLS = LLMTestCaseParams.EXPECTED_OUTPUT.value
    SEARCH_RESULT = "search_result"


@dataclass
class GraniteGuardianBinaryNode(BinaryJudgementNode):
    """
    Granite Guardian Binary Judgement Node.
    The field 'criteria' has been repurposed to be a GG risk definition
    and the LLMTestCaseGuardian object is checked according to that definition
    """

    criteria: RiskDefinition
    children: List[VerdictNode]
    evaluation_params: Optional[List[LLMTestCaseParamsGuardian]] = None
    granite_guardian_tool: GraniteGuardianTool = None  # I only made this a default field to avoid problems with dataclass inheritance

    def __hash__(self):
        return id(self)

    def _get_guardian_input(
        self,
        test_case: LLMTestCaseGuardian,
    ) -> GraniteGuardianInputSchema:
        # Gather input data from evaluation_params
        input_kwargs = {}
        for param in self.evaluation_params or []:
            val = getattr(test_case, param.value)
            if param == LLMTestCaseParamsGuardian.SEARCH_RESULT:
                input_kwargs["search_results"] = SearchToolOutputSchema(results=[val])
            elif param == LLMTestCaseParamsGuardian.INPUT:
                input_kwargs["query"] = val
            elif param == LLMTestCaseParamsGuardian.ACTUAL_OUTPUT:
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
