from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import Field

# Local fm_factual modules (subtree path)
from akd.tools.fact_reasoner.fm_factual.atom_extractor import AtomExtractor
from akd.tools.fact_reasoner.fm_factual.atom_reviser import AtomReviser
from akd.tools.fact_reasoner.fm_factual.context_retriever import ContextRetriever
from akd.tools.fact_reasoner.fm_factual.fact_reasoner import FactReasoner
from akd.tools.fact_reasoner.fm_factual.nli_extractor import NLIExtractorOld
from akd.tools.fact_reasoner.fm_factual.query_builder import QueryBuilder

from ._base import BaseIOSchema, BaseTool, BaseToolConfig


class FactReasonerToolConfig(BaseToolConfig):
    gen_model: str
    context_retriever_service_type: str
    cache_dir: str
    collection_name: str
    k: int
    merlin_path: str


class FactReasonerInputSchema(BaseIOSchema):
    """Schema for input of a tool for generating factuality scores."""

    response: str = Field(
        ..., description="The response to be passed to the FactReasoner module."
    )


class FactReasonerOutputSchema(BaseIOSchema):
    """Schema for output of a tool for generating factuality scores."""

    results: Dict[str, Any] = Field(..., description="FactReasoner result dictionary.")
    marginals: List[Dict[str, Any]] = Field(
        ...,
        description="Marginal probabilities from the reasoning engine.",
    )


class FactReasonerTool(BaseTool):
    input_schema = FactReasonerInputSchema
    output_schema = FactReasonerOutputSchema

    def __init__(
        self,
        config: Optional[FactReasonerToolConfig] = None,
        debug: bool = False,
    ):
        config = config or FactReasonerToolConfig()
        super().__init__(config, debug)

        # Initialize pipeline components
        self.context_retriever = ContextRetriever(
            service_type=config.context_retriever_service_type,
            top_k=config.k,
            cache_dir=config.cache_dir,
            debug=debug,
        )
        self.query_builder = QueryBuilder(model=config.gen_model, prompt_version="v1")
        self.atom_extractor = AtomExtractor(config.gen_model)
        self.atom_reviser = AtomReviser(config.gen_model)
        self.nli_extractor = NLIExtractorOld(config.gen_model, prompt_version="v2")

        self.pipeline = FactReasoner(
            context_retriever=self.context_retriever,
            atom_extractor=self.atom_extractor,
            atom_reviser=self.atom_reviser,
            nli_extractor=self.nli_extractor,
            query_builder=self.query_builder,
            merlin_path=config.merlin_path,
        )

    @classmethod
    def from_params(
        cls,
        gen_model: str,
        collection_name: str,
        merlin_path: str,
        cache_dir: str,
        db_path: str,
        k: int,
        debug: bool,
    ) -> FactReasonerTool:
        config = FactReasonerToolConfig(
            gen_model=gen_model,
            collection_name=collection_name,
            merlin_path=merlin_path,
            cache_dir=cache_dir,
            k=k,
        )
        return cls(config, debug)

    async def arun(self, params: FactReasonerInputSchema) -> FactReasonerOutputSchema:
        # Build the pipeline
        self.pipeline.build(
            response=params.response,
            has_atoms=False,
            has_contexts=False,
            revise_atoms=True,
            remove_duplicates=True,
            contexts_per_atom_only=False,
            rel_atom_context=True,
            rel_context_context=False,
            text_only=False,
        )

        results, marginals = self.pipeline.score()

        if self.debug:
            logger.debug(f"[FactReasoner] Results: {results}")
            logger.debug(f"[FactReasoner] Marginals: {marginals}")

        return FactReasonerOutputSchema(results=results, marginals=marginals)
