## Research Context

**Original Research Question:** {original_query}

**Topic:** {topic_title}

**Scientific Decomposition:** {decomposition_title}
**Decomposition Justification:** {decomposition_justification}

## Known Parameters for This Approach

The following known parameters have been identified for this scientific decomposition:

**Instrument:** {approach_instrument}
**Platform:** {approach_platform}
**Processing Level:** {approach_processing_level}
**Temporal Range:** {approach_temporal}
**Spatial Bounds:** {approach_bounding_box}
**Temporal Resolution:** {approach_temporal_resolution}
**Spatial Resolution:** {approach_spatial_resolution}

## Your Task

**Analyze these parameters and decide:**

1. Are instrument + platform + spatial/temporal constraints **specific enough** to narrow results to the target phenomenon?
   - **YES** → Return `[""]` (single empty string)
   - **NO** → Add {min_variations}-{max_variations} focused search strings to narrow the scope

2. If adding search strings, what specific **phenomenon or measurable** from the decomposition should be targeted?
   - Example: For "sea surface temperature" decomposition → `["sea surface temperature"]` or `["SST"]`
   - Example: For "chlorophyll concentration" decomposition → `["chlorophyll"]` or `["ocean color"]`

**Remember**: Each search string you provide creates a separate query. Empty string means "use only the known parameters above."

Generate your search variations as a list of strings ({min_variations}-{max_variations} items).
