import asyncio
import os
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
    logging_metadata: Dict[str, Any] = Field(
        {},
        description="Additional logging metadata from the run.",
    )


class FactCheckToolConfig(BaseToolConfig):
    """Configuration for the FactCheckTool."""

    base_url: HttpUrl = Field(
        default=os.getenv(
            "FACT_CHECK_API_URL",
            default="http://localhost:8011",
        ),
        description="The base URL of the remote Fact-Checking and Correction Service.",
    )
    polling_interval_seconds: int = Field(
        default=120,
        description="How often to poll for job results.",
    )
    job_timeout_seconds: int = Field(
        default=1800,
        description="Maximum time to wait for a job to complete (30 minutes).",
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
        Starts a fact-checking job and polls for its completion.
        """
        logger.info(
            f"Sending fact-check request for question: '{params.question[:50]}...'",
        )

        try:
            # Start the job
            start_response = await self.api_client.post(
                "/fact-check/start",
                json=params.model_dump(),
                timeout=60.0,
            )
            start_response.raise_for_status()
            job_id = start_response.json()["job_id"]
            logger.info(f"Successfully started job with ID: {job_id}")

            # Poll for the result
            total_wait_time = 0
            while total_wait_time < self.config.job_timeout_seconds:
                logger.info(f"Polling status for job {job_id}...")
                status_response = await self.api_client.get(
                    f"/fact-check/status/{job_id}",
                    timeout=60.0,
                )
                status_response.raise_for_status()
                status_data = status_response.json()

                if status_data["status"] == "completed":
                    logger.info(f"Job {job_id} completed successfully.")
                    return FactCheckOutputSchema(**status_data["result"])
                elif status_data["status"] == "failed":
                    raise Exception(
                        f"Job {job_id} failed on the server: {status_data.get('error', 'Unknown error')}",
                    )
                elif status_data["status"] == "pending":
                    logger.info(
                        f"Job {job_id} is in progress... (waited {total_wait_time}s)",
                    )
                    await asyncio.sleep(self.config.polling_interval_seconds)
                    total_wait_time += self.config.polling_interval_seconds

            raise asyncio.TimeoutError(
                f"Job {job_id} did not complete within the {self.config.job_timeout_seconds}s timeout.",
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error occurred while calling fact-check API: {e.response.status_code} - {e.response.text}",
            )
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise
