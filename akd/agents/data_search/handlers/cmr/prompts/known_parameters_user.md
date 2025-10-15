I need you to identify the known parameters that can be directly extracted from this research context for CMR data discovery.

## Context Information

**Original Research Question:** {original_query}

**Topic:** {topic_title}
**Topic Context:** {topic_context}

**Scientific Decomposition:** {decomposition_title}
**Scientific Justification:** {decomposition_justification}

## Task

Analyze this research context and identify query approaches using only **known parameters** - those that can be directly identified without searching:

- **Instruments/Platforms**: Only if explicitly mentioned or clearly implied
- **Temporal Constraints**: Extract exact date ranges and convert to ISO format
- **Spatial Constraints**: Extract geographic bounds and convert to decimal degrees (west,south,east,north)
- **Processing Levels**: If specified or clearly required
- **Resolution Requirements**: If mentioned

## Guidelines

1. **Spatial/Temporal Attention**: Carefully identify any explicit or implicit spatial and temporal requirements
2. **Resolution Requirements**: Consider any temporal or spatial resolution needs
3. **Multiple Approaches**: Create {min_approaches}-{max_approaches} different query approaches if multiple instruments/platforms could provide relevant data
4. **Broad Coverage**: Ensure at least one approach is broad enough to avoid over-filtering
5. **Univerality**: If spatial, temporal, or resolution requirements are essential to the question then they should universally apply to all query approaches generated. Do not apply spatial/temporal/resolution values piecemeal unless there is a very good reason to do so.

## Important

- Only include parameters that are directly identifiable from the context
- Do NOT include keywords or search terms (those will be handled separately)
- Focus on hard filters that can be applied directly to CMR queries
- Consider what instruments/platforms have coverage during the specified time periods and spatial areas
