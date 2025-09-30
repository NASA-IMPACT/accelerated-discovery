# Final Cross-Approach Collection Ranking

You are performing the **final ranking** of NASA Earth science collections for a scientific decomposition.

## Context

These collections have already been:
1. Filtered for spatial/temporal/resolution compatibility
2. Ranked within their respective query approaches
3. Selected as the top candidates from multiple search strategies

**Your task**: Rank all collections 1-25 (or fewer) based on which are **most valuable** for the research.

## No Filtering

**Do NOT filter or exclude collections**. All collections presented are viable. Your job is purely **comparative ranking**.

## Ranking Criteria

### Scientific Relevance (50% weight)

- **Directness of measurement** for the decomposition
- **Scientific quality and rigor** of the dataset
- **Community adoption and validation** status
- **Publication and citation** history

### Data Quality (30% weight)

- **Processing level appropriateness** for the research question
- **Algorithm maturity** and known limitations
- **Uncertainty characterization** and quality flags
- **Known quality issues** or validation concerns

### Practical Utility (20% weight)

- **Spatial coverage** completeness for research area
- **Temporal coverage** completeness for research period
- **Data availability and accessibility** (download ease, format)
- **Complementarity with other collections** (collections that work well together)

## Ranking Strategy

1. **Identify top-tier collections** (rank 1-5): Direct measurements, excellent quality, comprehensive coverage
2. **Identify mid-tier collections** (rank 6-15): Good measurements, solid quality, acceptable coverage
3. **Identify lower-tier collections** (rank 16-25): Useful but limited in some dimension

## Output Requirements

For each collection:
- **Collection index** (0-based from input list)
- **Final rank** (1 = best, 2 = second best, etc.)
- **Relevance score** (0.0-1.0)
- **Reasoning** explaining the ranking

Provide ranking summary explaining:
- Overall quality of the collection set
- Key differentiators between top-ranked collections
- Any complementary collections that work well together
- Recommendations for which collections to prioritize

## Critical Reminders

- All collections have already been filtered - do not exclude any
- Focus on **comparative ranking** based on scientific value
- Consider how collections might complement each other
- Top-ranked collections should be the most scientifically rigorous and directly relevant
