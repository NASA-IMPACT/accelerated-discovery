I need you to generate search variations that will discover relevant datasets when combined with the known parameters already identified.

## Research Context

**Original Research Question:** {original_query}

**Topic:** {topic_title}
**Scientific Decomposition:** {decomposition_title}
**Decomposition Justification:** {decomposition_justification}

## Query Approach (Known Parameters)

The following known parameters have been identified for this scientific decomposition:

**Instrument:** {approach_instrument}
**Platform:** {approach_platform}
**Processing Level:** {approach_processing_level}
**Temporal Range:** {approach_temporal}
**Spatial Bounds:** {approach_bounding_box}
**Temporal Resolution:** {approach_temporal_resolution}
**Spatial Resolution:** {approach_spatial_resolution}

## Task

Generate 0-5 separate search variations for this query approach. Each variation should be a different keyword combination (or empty for no additional keywords).

**Remember: CMR uses AND logic** - all keywords in a search must match, so fewer keywords = more results.

## Decision Framework

1. **Assess Specificity**: Are the known parameters (instrument/platform/level) already specific enough?
   - If YES: Include an empty search (no additional keywords)
   - If NO: Add keyword variations to narrow the scope

2. **Generate Variations**: Create separate searches for different ways scientists might describe this data:
   - Formal scientific terminology
   - Common abbreviations or acronyms
   - Alternative expressions of the same concept

3. **Keep Focused**: Each search should target one specific aspect, not combine multiple unrelated terms

## Examples

**High specificity case** (MODIS Terra Level 2):
- "" (empty - let instrument/platform filter)
- "sea surface temperature"
- "SST"

**Low specificity case** (Generic satellite, broad temporal range):
- "land cover"
- "vegetation classification"
- "LULC"

Generate your search variations as a list of keyword strings. Use empty string for no additional keywords.
