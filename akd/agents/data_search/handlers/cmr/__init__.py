"""CMR handler for NASA Common Metadata Repository."""

from .components import (
    CMRApproachCollectionFilteringComponent,
    CMRFinalCollectionRankingComponent,
    CMRKnownParametersComponent,
    CMRSearchableParametersComponent,
)
from .config import CMRHandlerConfig
from .handler import CMRHandler
from .schemas import (
    CMRApproachCollectionFilteringInputSchema,
    CMRApproachCollectionFilteringOutput,
    CMRFilteredRankedCollection,
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
    "CMRFilteredRankedCollection",
    "CMRFinalCollectionRankingInputSchema",
    "CMRFinalCollectionRankingOutput",
    "CMRFinalRankedCollection",
]
