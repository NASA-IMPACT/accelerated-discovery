# Extract PDS4 Query Approaches

## Research Context

**Original Research Question:** {original_query}

**Topic:** {topic_title}
**Topic Context:** {topic_context}

**Scientific Decomposition:** {decomposition_title}
**Scientific Justification:** {decomposition_justification}

## Your Task

Analyze this planetary science research context and generate **{min_approaches}-{max_approaches} PDS4 query approaches** - different discovery strategies for finding relevant datasets using PDS4's context-based discovery system.

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

## PDS4 Query Approach Generation

For each approach, generate a complete PDS4QueryApproach with these fields:

**Required Fields:**
- **approach_description**: Human-readable description of the search approach (what context searches will be performed and how they'll be combined)
- **investigation_keywords**: List of keywords for investigation/mission context search (e.g., ["mars", "rover", "curiosity"])
- **target_keywords**: List of keywords for target/celestial body context search (e.g., ["mars", "phobos"])
- **instrument_keywords**: List of keywords for instrument context search (e.g., ["spectrometer", "chemcam"])
- **primary_keywords**: Main search terms for direct bundle/collection queries (e.g., ["surface", "composition"])
- **temporal_context**: Descriptive temporal period if relevant (e.g., "2012-2020", "Apollo era")

**Auto-populated Fields (do not specify):**
- **approach_index**: Automatically assigned (0, 1, 2, 3...)
- **investigation_urns, target_urns, instrument_urns**: Populated during execution from context searches (top 3 of each)
- **Search limits**: Defaults are used unless you need specific limits

## Strategy Examples

**Investigation-First:**
```
approach_description: "Search for specific mission/investigation, then retrieve collections by investigation URN"
investigation_keywords: ["mars", "odyssey"]
Use when: Specific mission mentioned or mission-focused research
```

**Target-First:**
```
approach_description: "Search for target celestial body, then retrieve collections by target URN"
target_keywords: ["europa", "jupiter"]
Use when: Target body is primary focus
```

**Multi-Context:**
```
approach_description: "Search for mission and target, then retrieve collections using multiple URN combinations"
investigation_keywords: ["cassini"]
target_keywords: ["titan"]
Use when: Both mission and target are important constraints
```

**Bundle Discovery:**
```
approach_description: "Direct bundle search for thematic datasets"
primary_keywords: ["surface", "composition", "mars"]
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

- Focus on approaches that leverage PDS4's context relationships (investigation → target → instrument → collection)
- Use natural language keywords that PDS4 MCP tools can understand
- Provide clear `approach_description` that explains which context searches will be performed
- Consider both narrow (highly specific) and broad (comprehensive coverage) approaches
- Ensure each approach offers a meaningfully different way to discover relevant planetary science data
- Account for the hierarchical nature of PDS4 data (bundles → collections → observational products)
- Remember: URNs will be extracted from context searches during execution - focus on providing good keywords