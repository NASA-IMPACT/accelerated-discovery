# Filter PDS4 Products

## What We're Looking For

**Original Query**: {original_query}

**Decomposition**: {decomposition_title}
{decomposition_justification}

## Search Strategy

**Description**: {strategy_description}

**Discovery Keywords:**
- Investigation/Mission: {investigation}
- Target: {target}
- Instruments: {instruments}
- Temporal Period: {temporal}

**Context URNs Used in Collection Searches:**

The collections below were retrieved using **all combinations** of these URNs (up to 3×3×3 = 27 different searches). Each list contains the **top 3 most relevant** URNs from context searches. Collections may come from **ANY combination** of these contexts.

**Investigations** (top 3 missions - collections may be from any of these):
{investigation_urns}

**Targets** (top 3 celestial bodies - collections may reference any of these):
{target_urns}

**Instruments** (top 3 instruments - collections may use any of these):
{instrument_urns}

**Important**: A collection showing "Mars Odyssey" as the mission is valid even if "Mars Reconnaissance Orbiter" appears first in the list above - all combinations were searched equally.

## PDS4 Products to Review ({num_items} total)

{items_list}

## Your Task

Review the provided PDS4 products and return the **indexes of the {min_items}-{max_items} best matches**, ordered by relevance (best first).

## Matching Criteria

A product is a good match if its title, description, and metadata indicate it:

1. **Studies the right target**: Directly related to the specified celestial body or phenomenon in the decomposition
2. **Uses appropriate instruments/platforms**: Matches the approach parameters (mission, instrument, spacecraft)
3. **Contains relevant measurements**: Provides the type of data needed for the scientific investigation
4. **Has appropriate temporal coverage**: Covers the requested mission period or observation timeframe
5. **Offers suitable data quality**: Appropriate processing level and data completeness for the research question

## PDS4-Specific Considerations

- **Mission Alignment**: Products from the specified investigation/mission are highly preferred
- **Target Relevance**: Products must study the target body mentioned in the decomposition
- **Instrument Capability**: Consider whether the instrument type can provide the measurements needed
- **Data Hierarchy**: Understand that bundles contain collections, which contain observational products
- **Context Relationships**: Products with strong relationships to the specified context (investigation, target, instrument) are preferred
- **Scientific Value**: Consider the scientific relevance of the dataset to the research decomposition

## Important

- Always return at least one product, no matter what
- Return **{min_items}-{max_items} indexes** based on match quality
- Quality over quantity - only include strong matches
- Order indexes from best match to worst match
- Focus on products that directly support the scientific investigation described in the decomposition