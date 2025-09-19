**Original Research Query:** {original_query}

**Topic to Route:**
**{topic_title}**
Context: {topic_context}

Please analyze this topic and determine which repositories should be consulted for this specific topic.

Consider:
1. **Primary data types needed** - What kinds of measurements/observations are required?
2. **Data source characteristics** - Satellite vs ground-based, global vs local coverage
3. **Repository strengths** - Which repository specializes in this domain?
4. **Research context** - How does the original query influence repository selection?

Return a single route object with exactly two fields:
- repositories: list of exact repository names for this topic (e.g., "CMR", "USGS", "NOAA", "EPA") - use only the exact names from the system prompt
- rationales: list of short explanations (order-aligned with repositories)
