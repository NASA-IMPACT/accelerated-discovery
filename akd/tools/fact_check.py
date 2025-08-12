from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from pydantic import Field, HttpUrl

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig


class FactCheckInputSchema(InputSchema):
    """Input schema for the Fact-Checking Tool."""

    question: str = Field(..., description="The original question that was asked.")
    answer: str = Field(..., description="The LLM answer to be fact-checked.")


class FactCheckOutputSchema(OutputSchema):
    """Output schema for the Fact-Checking Tool's results."""

    fact_reasoner_score: Dict[str, Any] = Field(
        ...,
        description="The full scoring dictionary from the FactReasoner.",
    )
    supported_atoms: List[Dict[str, Any]] = Field(
        ...,
        description="List of atoms determined to be supported.",
    )
    not_supported_atoms: List[Dict[str, Any]] = Field(
        ...,
        description="List of atoms determined to be not supported.",
    )
    contexts: List[Dict[str, Any]] = Field(
        ...,
        description="List of retrieved contexts used for the check.",
    )
    graph_id: Optional[str] = Field(
        None,
        description="The unique ID for the generated fact graph.",
    )


class FactCheckToolConfig(BaseToolConfig):
    """Configuration for the FactCheckTool."""

    base_url: HttpUrl = Field(
        # default="http://localhost:8011",
        default="https://factreasoner-service-app.1yhbkn094k2v.us-south.codeengine.appdomain.cloud",
        description="The base URL of the remote Fact-Checking and Correction Service.",
    )


class FactCheckTool(
    BaseTool[FactCheckInputSchema, FactCheckOutputSchema],
):
    """
    A tool that calls an API to perform fact-checking on a given answer.
    """

    name = "fact_check_tool"
    description = (
        "Calls an API to run the FactReasoner pipeline on a question and answer."
    )
    input_schema = FactCheckInputSchema
    output_schema = FactCheckOutputSchema
    config_schema = FactCheckToolConfig

    def __init__(
        self,
        config: FactCheckToolConfig | None = None,
        debug: bool = False,
    ):
        """Initializes the FactCheckTool and its HTTP client."""
        config = config or FactCheckToolConfig()
        super().__init__(config, debug)

        logger.info("Initializing FactCheckTool...")
        self.api_client = httpx.AsyncClient(base_url=str(self.config.base_url))

    async def _arun(
        self,
        params: FactCheckInputSchema,
    ) -> FactCheckOutputSchema:
        """
        Calls the /fact-check/ endpoint on the remote service.
        """
        logger.info(
            f"Sending fact-check request for question: '{params.question[:50]}...'",
        )

        try:
            response = await self.api_client.post(
                "/fact-check/",
                json=params.model_dump(),
                timeout=1500.0,  # 25 mins timeout for potentially slow API calls
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            results = response.json()
            return FactCheckOutputSchema(**results)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error occurred while calling fact-check API: {e.response.status_code} - {e.response.text}",
            )
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise
