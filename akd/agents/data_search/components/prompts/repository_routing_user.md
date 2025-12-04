**Original Research Query:** {original_query}

**Topic Context:**
**{topic_title}**
{topic_context}

**Scientific Decomposition to Route:**
**{decomposition_title}**
Scientific Justification: {decomposition_justification}

**Available NASA Repositories:** {nasa_repositories_available}

Please analyze this specific scientific decomposition and determine the single best data repository that can provide the required datasets.

Consider:
1. **Primary data types needed** - What kinds of measurements/observations are required for THIS decomposition?
2. **Data source characteristics** - Satellite vs ground-based, global vs local coverage
3. **Repository strengths** - Which repository specializes in this domain?
4. **Available NASA repositories** - Only route to NASA repositories listed above as "Available"
5. **Research context** - How does the original query and topic influence repository selection?

Return a single `route` object with exactly three fields:
- `repository`: string - the single best repository name (e.g., "CMR", "USGS", "NOAA")
- `rationale`: string - concise explanation (1-2 sentences) for why this repository was selected
- `is_external`: boolean - true if non-NASA repository, false if NASA repository
