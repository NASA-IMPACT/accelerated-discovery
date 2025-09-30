# Collection Ranking Task

## Research Context

**Original Research Query**: {original_query}

**Topic**: {topic_title}
**Topic Context**: {topic_context}

**Scientific Decomposition**: {decomposition_title}
**Scientific Justification**: {decomposition_justification}

**Query Approaches Used**:
- **Instruments**: {approach_instruments}
- **Platforms**: {approach_platforms}
- **Keywords**: {approach_keywords}
- **Temporal Ranges**: {approach_temporal_ranges}
- **Spatial Bounds**: {approach_spatial_bounds}
- **Processing Levels**: {approach_processing_levels}
- **Required Temporal Resolution**: {approach_temporal_resolutions}
- **Required Spatial Resolution**: {approach_spatial_resolutions}

## Collections to Evaluate

You have {num_collections} collections to evaluate and filter:

{collections_list}

## Critical Task Instructions

**Primary Focus**: Evaluate collections specifically for the **{decomposition_title}** decomposition, NOT the entire research question.

**Filtering Approach**: Apply the systematic filtering process from your system instructions:
1. **Fundamental Compatibility**: Does it measure {decomposition_title}?
2. **Spatial Coverage/Resolution**: Adequate for the research area and scale?
3. **Temporal Coverage/Resolution**: Covers required time period and frequency?
4. **Data Quality/Processing Level**: Appropriate quality and processing level?
5. **Accessibility/Usability**: Accessible and in usable format?
6. **Scientific Relevance Scoring**: Score based on direct relevance, quality, and coverage

## Expected Output

**For Selected Collections** (targeting 3-7 collections):
1. **Collection Index**: 0-based index from the list above
2. **Relevance Score**: 0.0-1.0 with clear justification using scoring criteria
3. **Filter Analysis**: Which filters it passed and any concerns noted
4. **Ranking Justification**: Why this collection ranks at this position
5. **Usage Notes**: Special considerations for research application

**For Eliminated Collections**:
1. **Collection Identifier**: Short name or title
2. **Elimination Reason**: Specific filter failure (be specific about which filter and why)
3. **Alternative Suggestions**: If applicable, note what would make it suitable

**Summary Assessment**:
- Overall collection quality for this {decomposition_title} decomposition
- Any coverage gaps or limitations identified
- Confidence level in the ranking results
- Recommendations for complementary data sources

## Quality Thresholds

- **Minimum relevance score for inclusion**: 0.4
- **Preferred relevance score**: ≥0.7 for primary recommendations
- **Target collection count**: 3-7 collections (quality over quantity)

Remember: You are optimizing for the **{decomposition_title}** measurement specifically, considering how it contributes to understanding **{topic_title}** within the broader context of: {original_query}
