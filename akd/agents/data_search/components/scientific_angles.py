"""
Scientific Angles Generation Component.

Uses LLM to generate scientific angles for data discovery based on research queries
and relevant scientific documents.
"""

import asyncio
from typing import List, Dict, Any

from loguru import logger

from akd.agents._base import InstructorBaseAgent, BaseAgentConfig
from akd._base import InputSchema
from ..schemas import ScientificAnglesOutput
from .prompt_loader import load_prompt_template, load_and_format_prompt


class ScientificAnglesInputSchema(InputSchema):
    """Input schema for scientific angles generation."""

    query: str
    documents: List[Dict[str, Any]]


class ScientificAnglesComponent(
    InstructorBaseAgent[ScientificAnglesInputSchema, ScientificAnglesOutput]
):
    """
    Component for generating scientific angles using LLM.

    Takes a scientific query and relevant documents, then generates 2-6 scientific angles
    that represent different perspectives for finding relevant Earth science data.
    """

    input_schema = ScientificAnglesInputSchema
    output_schema = ScientificAnglesOutput

    def __init__(self, config: BaseAgentConfig | None = None, debug: bool = False):
        """Initialize the scientific angles component."""
        # Set up specialized configuration for scientific angle generation
        if config is None:
            config = BaseAgentConfig()

        # Override system prompt for scientific angle generation
        config.system_prompt = load_prompt_template("scientific_angles_system")
        config.temperature = 0.1  # Low temperature for consistent, factual output

        super().__init__(config=config, debug=debug)


    async def process(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> ScientificAnglesOutput:
        """
        Generate scientific angles for data discovery.

        Args:
            query: Natural language scientific research query
            documents: List of relevant scientific documents (may be empty)

        Returns:
            ScientificAnglesOutput with generated angles
        """
        if self.debug:
            logger.debug(f"Generating scientific angles for query: '{query}'")
            logger.debug(f"Using {len(documents)} reference documents")

        # Format the user prompt
        user_prompt = self._format_user_prompt(query, documents)

        # Add user message to memory
        self.memory.append({"role": "user", "content": user_prompt})

        # Retry with exponential backoff for rate limiting
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                # Generate angles using LLM
                response = await self.get_response_async()

                if self.debug:
                    logger.debug(f"Generated {len(response.angles)} scientific angles")
                    for i, angle in enumerate(response.angles, 1):
                        logger.debug(f"  {i}. {angle.title}")

                return response

            except Exception as e:
                if attempt == max_retries:
                    error_msg = f"Failed to generate scientific angles after {max_retries + 1} attempts: {e}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                # Check if it's a rate limit error
                if "429" in str(e) or "rate" in str(e).lower():
                    delay = base_delay * (2**attempt)
                    if self.debug:
                        logger.warning(
                            f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})"
                        )
                    await asyncio.sleep(delay)
                else:
                    # Non-rate-limit error, don't retry
                    error_msg = f"Failed to generate scientific angles: {e}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

    def _format_user_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Format the user prompt with query and documents."""
        # Format documents section
        documents_section = []
        if documents:
            for i, doc in enumerate(documents, 1):
                # Format document - assume documents have 'title' and 'content' fields
                title = doc.get("title", f"Document {i}")
                content = doc.get("content", str(doc))
                documents_section.extend([f"**Document {i}: {title}**", content, ""])
        else:
            documents_section.append("None Provided")
        
        formatted_documents = "\n".join(documents_section)
        
        return load_and_format_prompt(
            "scientific_angles_user",
            query=query,
            documents=formatted_documents
        )

    async def _arun(
        self, params: ScientificAnglesInputSchema, **kwargs
    ) -> ScientificAnglesOutput:
        """Execute the scientific angles generation."""
        return await self.process(params.query, params.documents)
