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
- Instrument Hosts: {instrument_hosts}
- Temporal Period: {temporal}

**Extracted Context URNs:**
- Investigation: {investigation_urn}
- Target: {target_urn}
- Instrument: {instrument_urn}
- Instrument Host: {instrument_host_urn}

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