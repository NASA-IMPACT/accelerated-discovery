"""CMR handler for NASA Common Metadata Repository."""

from .components import (
    CMRApproachCollectionFilteringComponent,
    CMRFinalCollectionRankingComponent,
    CMRKnownParametersComponent,
    CMRSearchableParametersComponent,
)
from .config import CMRHandlerConfig
from .handler import CMRHandler
from .llm_reranker_adapter import LLMRerankerAdapter
from .schemas import (
    CMRApproachCollectionFilteringInputSchema,
    CMRApproachCollectionFilteringOutput,
    CMRFinalCollectionRankingInputSchema,
    CMRFinalCollectionRankingOutput,
    CMRFinalRankedCollection,
    CMRKnownParametersInputSchema,
    CMRKnownParametersOutput,
    CMRQueryApproach,
    CMRSearchableParametersInputSchema,
    CMRSearchableParametersOutput,
    CMRSearchableQuery,
)

__all__ = [
    # Handler and config
    "CMRHandler",
    "CMRHandlerConfig",
    # LLM Reranker Adapter
    "LLMRerankerAdapter",
    # Components
    "CMRKnownParametersComponent",
    "CMRSearchableParametersComponent",
    "CMRApproachCollectionFilteringComponent",
    "CMRFinalCollectionRankingComponent",
    # Schemas
    "CMRQueryApproach",
    "CMRKnownParametersInputSchema",
    "CMRKnownParametersOutput",
    "CMRSearchableQuery",
    "CMRSearchableParametersInputSchema",
    "CMRSearchableParametersOutput",
    "CMRApproachCollectionFilteringInputSchema",
    "CMRApproachCollectionFilteringOutput",
    "CMRFinalCollectionRankingInputSchema",
    "CMRFinalCollectionRankingOutput",
    "CMRFinalRankedCollection",
]
