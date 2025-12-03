"""
Data search workflow constants for performance tuning.

These values control LLM prompt generation, Instructor validation, and dynamic calculations
throughout the entire data search pipeline.

Edit these values to tune system performance and parallelism.
"""

# ============================================================================
# UNIVERSAL WORKFLOW LIMITS (Apply to all handlers)
# ============================================================================

# Topic Splitting
MAX_TOPICS = 3
MIN_TOPICS = 1

# Scientific Decomposition (per topic)
MAX_DECOMPOSITIONS_PER_TOPIC = 3
MIN_DECOMPOSITIONS_PER_TOPIC = 1

# ============================================================================
# CMR HANDLER LIMITS
# ============================================================================

# Approach Generation (per decomposition)
# - LLM generates this many approaches before keyword-only injection
CMR_MAX_LLM_APPROACHES = 3
CMR_MIN_LLM_APPROACHES = 1

# - Actual execution max after keyword-only injection (when enabled)
CMR_MAX_TOTAL_APPROACHES_WITH_KEYWORD = CMR_MAX_LLM_APPROACHES + 1  # = 5

# Search Variations (per approach)
CMR_MAX_SEARCH_VARIATIONS_PER_APPROACH = 3
CMR_MIN_SEARCH_VARIATIONS_PER_APPROACH = 0

# Calculated Limits (auto-update when above values change)
CMR_MAX_SEARCHABLE_QUERIES = (
    CMR_MAX_TOTAL_APPROACHES_WITH_KEYWORD * CMR_MAX_SEARCH_VARIATIONS_PER_APPROACH
)  # = 5 * 3 = 15

# ============================================================================
# PDS4 HANDLER LIMITS
# ============================================================================

# Strategy Generation (per decomposition)
PDS4_MAX_STRATEGIES = 4
PDS4_MIN_STRATEGIES = 1

# URN Filtering (per strategy)
# NOTE: URN counts are now dynamically determined by LLM filtering component
# The following are safety limits, not hard limits passed to the LLM
PDS4_MAX_INVESTIGATION_URNS_PER_STRATEGY = 3  # Safety limit (deprecated when LLM filtering enabled)
PDS4_MAX_TARGET_URNS_PER_STRATEGY = 3  # Safety limit (deprecated when LLM filtering enabled)
PDS4_MAX_INSTRUMENT_URNS_PER_STRATEGY = 3  # Safety limit (deprecated when LLM filtering enabled)

# URN Combination Safety Limit
# Maximum total combinations to prevent combinatorial explosion when LLM selects many URNs
PDS4_MAX_URN_COMBINATIONS_PER_STRATEGY = 100  # Safety limit (increased from 27 for LLM flexibility)

# Collection Filtering (per strategy)
PDS4_MAX_COLLECTIONS_PER_STRATEGY = 5
PDS4_MIN_COLLECTIONS_PER_STRATEGY = 0

# Final Ranking (cross-strategy)
PDS4_FINAL_COLLECTION_COUNT = 25
