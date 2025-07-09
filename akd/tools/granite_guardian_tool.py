from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from ollama import chat
from pydantic.fields import Field

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig
from akd.tools.search import SearchResultItem, SearchToolOutputSchema


class RiskDefinition(Enum):
    """
    Enumeration of possible risk definitions defined in Granite Guardian.
    """

    HARM = "harm"
    SOCIAL_BIAS = "social_bias"
    PROFANITY = "profanity"
    SEXUAL_CONTENT = "sexual_content"
    UNETHICAL_BEHAVIOR = "unethical_behavior"
    VIOLENCE = "violence"
    JAILBREAK = "jailbreak"
    GROUNDEDNESS = "groundedness"
    RELEVANCE = "relevance"
    ANSWER_RELEVANCE = "answer_relevance"


class GraniteGuardianInputSchema(InputSchema):
    """
    Input schema for the Granite Guardian tool.
    """

    query: Optional[str] = Field(None, description="Query text to evaluate.")
    response: Optional[str] = Field(None, description="Assistant/model response.")
    search_results: Optional[SearchToolOutputSchema] = Field(
        None,
        description="Search result outputs for batch risk analysis.",
    )
    risk_type: Optional[str] = Field(
        default=RiskDefinition.ANSWER_RELEVANCE,
        description="Type of risk check to apply.",
    )


class GraniteGuardianOutputSchema(OutputSchema):
    """
    Output schema for the Granite Guardian tool.
    """

    risk_results: List[Dict[str, Any]] = Field(
        ...,
        description="Risk evaluation results per input or search result.",
    )


class GraniteGuardianToolConfig(BaseToolConfig):
    """
    Configuration for Granite Guardian Tool.
    """

    model: str = "granite3-guardian"
    default_risk_type: RiskDefinition = Field(
        default=RiskDefinition.ANSWER_RELEVANCE,
        description="Default risk check",
    )
    snippet_n_chars: int = Field(
        100,
        description="Number of characters to truncate the snippet to.",
    )


class GraniteGuardianTool(
    BaseTool[GraniteGuardianInputSchema, GraniteGuardianOutputSchema],
):
    name = "granite_guardian_tool"
    description = "Evaluates risks using the Granite Guardian model via Ollama."
    input_schema = GraniteGuardianInputSchema
    output_schema = GraniteGuardianOutputSchema
    config_schema = GraniteGuardianToolConfig

    def __init__(
        self,
        config: GraniteGuardianToolConfig | None = None,
        debug: bool = False,
    ):
        config = config or GraniteGuardianToolConfig()
        super().__init__(config, debug)
        self.model = config.model
        self.default_risk_type = config.default_risk_type
        self.snippet_n_chars = config.snippet_n_chars

    async def _arun(
        self,
        params: GraniteGuardianInputSchema,
    ) -> GraniteGuardianOutputSchema:
        risk_type = params.risk_type or self.default_risk_type

        self._set_risk_definition(risk_type)

        if params.search_results:
            outputs = self._process_search_results(params.search_results.results)
        elif params.query and params.response:
            outputs = self._process_multiturn(params.query, params.response)
        elif params.query:
            outputs = self._process_singleturn(params.query)
        else:
            raise ValueError(
                "Must provide either 'query', 'query + response', or 'search_results'.",
            )

        return GraniteGuardianOutputSchema(risk_results=outputs)

    def _set_risk_definition(self, definition: str):
        try:
            chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"/set system {definition}",
                    },
                ],
            )
            if self.debug:
                logger.debug(f"[GuardianTool] Set risk definition: {definition}")
        except Exception as e:
            logger.error(f"[GuardianTool] Error setting risk definition: {e}")

    def _call_guardian(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            result = chat(model=self.model, messages=messages)
            label = result["message"]["content"].strip().lower()
            return {
                "risk_label": label,
                "is_risky": label == "yes",
                "raw_response": result,
            }
        except Exception as e:
            logger.error(f"[GuardianTool] Ollama error: {e}")
            return {"error": str(e)}

    def _process_singleturn(self, query: str) -> List[Dict[str, Any]]:
        messages = [{"role": "user", "content": query}]
        return [self._call_guardian(messages)]

    def _process_multiturn(self, query: str, response: str) -> List[Dict[str, Any]]:
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]
        return [self._call_guardian(messages)]

    def _process_search_results(
        self,
        results: List[SearchResultItem],
    ) -> List[Dict[str, Any]]:
        outputs = []
        for idx, item in enumerate(results):
            if not item.query or not item.content:
                outputs.append(
                    {
                        "index": idx,
                        "skipped": True,
                        "reason": "Missing query or content",
                    },
                )
                continue
            messages = [
                {"role": "user", "content": item.query},
                {"role": "assistant", "content": item.content},
            ]
            res = self._call_guardian(messages)
            outputs.append(
                {
                    "index": idx,
                    "query": item.query,
                    "snippet": item.content[: self.snippet_n_chars],
                    "risk_label": res.get("risk_label"),
                    "is_risky": res.get("is_risky"),
                    "raw_response": res.get("raw_response"),
                },
            )
        return outputs
