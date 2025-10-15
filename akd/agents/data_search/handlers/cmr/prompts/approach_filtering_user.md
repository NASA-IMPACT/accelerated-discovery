# Filter Collections

## What We're Looking For

**Original Query**: {original_query}

**Decomposition**: {decomposition_title}
{decomposition_justification}

## Search Approach

- **Instrument**: {approach_instrument}
- **Platform**: {approach_platform}
- **Processing Level**: {approach_processing_level}
- **Time Period**: {approach_temporal_range}
- **Geographic Area**: {approach_spatial_bounds}

## Collections to Review ({num_items} total)

{items_list}


## Your Task

Review the provided collections and return the **indexes of the {min_items}-{max_items} best matches**, ordered by relevance (best first).

## Matching Criteria

A collection is a good match if its title and abstract indicate it:

1. **Measures the right phenomenon**: Directly related to the decomposition target
2. **Uses the right instrument/platform**: Matches the approach parameters (if specified)
3. **Has appropriate coverage**: Covers the requested time period and geographic area

## Important

- Always return at least one collection, no matter what
- Return **{min_items}-{max_items} indexes** based on match quality
- Quality over quantity - only include strong matches
- Order indexes from best match to worst match
