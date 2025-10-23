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
CMR_MAX_LLM_APPROACHES = 4
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
# PDS4 HANDLER LIMITS (Placeholder - not yet implemented)
# ============================================================================

# When PDS4 is implemented, add similar constants here:
# PDS4_MAX_LLM_APPROACHES = 4
# PDS4_MAX_SEARCH_VARIATIONS_PER_APPROACH = 3
# etc.
