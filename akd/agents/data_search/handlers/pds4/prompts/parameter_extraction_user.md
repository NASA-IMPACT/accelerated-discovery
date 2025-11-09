# Extract PDS4 Parameters and Tool Strategies

## Research Context

**Original Research Question:** {original_query}

**Topic:** {topic_title}
**Topic Context:** {topic_context}

**Scientific Decomposition:** {decomposition_title}
**Scientific Justification:** {decomposition_justification}

## Your Task

Analyze this planetary science research context and generate **{min_approaches}-{max_approaches} PDS4 MCP tool strategies** - different approaches for discovering relevant datasets using PDS4's context-based discovery system.

## Parameter Extraction

For each approach, extract and structure:

**Target Information:**
- Celestial bodies mentioned (planets, moons, asteroids, etc.)
- Target types using PDS4 classifications (Planet, Satellite, Asteroid, Comet, etc.)
- Specific regions or phenomena of interest

**Mission/Investigation Context:**
- Specific missions mentioned (Mars 2020, Cassini, Juno, etc.)
- Mission types or families (Mars rovers, Jupiter orbiters, etc.)
- Operational periods or mission phases

**Instrument Requirements:**
- Instrument types using PDS4 classifications (Spectrometer, Imager, Radio-Radar, etc.)
- Specific instruments mentioned (ChemCam, HiRISE, VIMS, etc.)
- Measurement capabilities needed

**Platform Context:**
- Instrument host types (Rover, Spacecraft, Lander, Observatory, etc.)
- Specific platforms (Perseverance, Curiosity, Cassini, etc.)

**Temporal Context:**
- Mission operational periods
- Specific observation timeframes
- Comparative time periods

**Scientific Keywords:**
- Key phenomena to search for
- Measurement types
- Analysis targets

## PDS4 MCP Tool Strategy Generation

For each approach, specify:

**Tool Strategy:** The sequence of PDS4 MCP tools to use
- Example: `search_investigations("mars rover") → search_collections(ref_lid_investigation=urn)`
- Example: `search_targets("europa") + search_instruments("spectrometer") → search_collections(ref_lid_target=urn1, ref_lid_instrument=urn2)`

**Primary Keywords:** Main search terms for context discovery tools

## Strategy Examples

**Investigation-First:**
```
Tool Strategy: search_investigations(keywords) → search_collections(ref_lid_investigation=urn)
Use when: Specific mission mentioned or mission-focused research
```

**Target-First:**
```
Tool Strategy: search_targets(keywords, target_type) → search_collections(ref_lid_target=urn)
Use when: Target body is primary focus
```

**Multi-Context:**
```
Tool Strategy: search_investigations(mission) + search_targets(target) → search_collections(ref_lid_investigation=urn1, ref_lid_target=urn2)
Use when: Both mission and target are important constraints
```

**Bundle Discovery:**
```
Tool Strategy: search_bundles(title_query)
Use when: Looking for thematic datasets or broad phenomenon studies
```

## Guidelines

1. **Quality over Quantity**: Create {min_approaches}-{max_approaches} approaches only if they offer genuinely different discovery paths
2. **Tool Efficiency**: Leverage PDS4 MCP's context-based design rather than over-complicating
3. **Natural Language**: Use descriptive keywords that match how PDS4 context products are named
4. **Discovery Breadth**: Balance specific targeted searches with broader exploratory approaches
5. **Mission Alignment**: Consider which missions actually studied the targets of interest
6. **Temporal Relevance**: Account for mission operational periods and data availability

## Important

- Focus on tool strategies that leverage PDS4's context relationships (investigation → target → instrument → collection)
- Use natural language keywords that PDS4 MCP tools can understand
- Consider both narrow (highly specific) and broad (comprehensive coverage) approaches
- Ensure each approach offers a meaningfully different way to discover relevant planetary science data
- Account for the hierarchical nature of PDS4 data (bundles → collections → observational products)