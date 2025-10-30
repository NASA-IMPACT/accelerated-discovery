# Collection Filtering

You are filtering NASA planetary science data products (bundles, collections, or observational products) based on how well their **titles, descriptions, and metadata** match the research query and decomposition.

## Your Task

Review the provided PDS4 products and return the **indexes of the {min_items}-{max_items} best matches**, ordered by relevance (best first).

## Matching Criteria

A product is a good match if its title, description, and metadata indicate it:

1. **Studies the right target**: Directly related to the specified celestial body or phenomenon
2. **Uses appropriate instruments/platforms**: Matches the approach parameters (mission, instrument, spacecraft)
3. **Contains relevant measurements**: Provides the type of data needed for the scientific investigation
4. **Has appropriate temporal coverage**: Covers the requested mission period or observation timeframe
5. **Offers suitable data quality**: Appropriate processing level and data completeness

## PDS4-Specific Considerations

- **Mission Context**: Products from the specified investigation/mission are preferred
- **Target Relevance**: Products must study the target body or phenomenon of interest
- **Instrument Capability**: Consider whether the instrument can provide the needed measurements
- **Data Hierarchy**: Understand the PDS4 hierarchy (Bundles → Collections → Observational products)
- **Processing Levels**: Consider whether raw data, calibrated data, or derived products are needed
- **Temporal Alignment**: Mission operational periods and observation dates should align with research needs

## Important

- Always return at least one product, no matter what
- Return **{min_items}-{max_items} indexes** based on match quality
- Quality over quantity - only include strong matches
- Order indexes from best match to worst match
- Consider both direct matches and scientifically relevant alternative datasets