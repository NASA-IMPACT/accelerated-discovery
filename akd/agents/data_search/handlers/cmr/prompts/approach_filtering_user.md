# Filter and Rank Collections for Query Approach

## Research Context

**Original Query**: {original_query}

**Topic**: {topic_title}
**Topic Context**: {topic_context}

**Decomposition**: {decomposition_title}
**Scientific Justification**: {decomposition_justification}

## This Query Approach

**Instrument**: {approach_instrument}
**Platform**: {approach_platform}
**Processing Level**: {approach_processing_level}
**Temporal Range**: {approach_temporal_range}
**Spatial Bounds**: {approach_spatial_bounds}
**Temporal Resolution Required**: {approach_temporal_resolution}
**Spatial Resolution Required**: {approach_spatial_resolution}
**Keywords**: {approach_keywords}

## Collections to Evaluate ({num_collections} total)

{collections_list}

## Task

**Part 1**: Filter out collections with fundamental mismatches (spatial, temporal, resolution, wrong measurement)

**Part 2**: From remaining collections, select and rank the **best 0-{max_collections}** for this approach

Focus on collections that:
- Directly measure {decomposition_title}
- Match the approach parameters (instrument, platform, etc.)
- Have appropriate coverage and quality

Return **0 collections** if none meet the requirements. Quality over quantity.
