# Filter Context Search URNs

## Research Context Chain

**Original Research Question:** {original_query}

**Topic:** {topic}

**Scientific Decomposition:** {decomposition}

**Search Strategy:** {strategy_description}

## Context Search Keywords

These are the keywords that were used to search for context products. **Use these as your PRIMARY criterion for filtering URNs.**

**Investigation Keywords:** {investigation_keywords}
**Target Keywords:** {target_keywords}
**Instrument Keywords:** {instrument_keywords}

## Context Search Results

### Investigation Results

{investigation_results}

### Target Results

{target_results}

### Instrument Results

{instrument_results}

## Your Task

**Filter the most relevant URNs from each context type based on keyword matching.**

For each context type (investigation, target, instrument):
1. **Match URN titles/descriptions against the search keywords** (this is your PRIMARY criterion)
2. **Select URNs with strong keyword matches** - exact matches, acronyms, or close semantic matches
3. **Reject URNs with weak/no keyword matches** - even if they seem interesting
4. **Filter out non-scientific contexts** - Laboratory_Analog, Equipment, Calibrator (for targets)
5. **Consider compatibility** - selected URNs should work together logically

### Selection Freedom

- You can select **0, 1, 5, 10, or any number** of URNs per context type
- **Quality over quantity** - only select URNs that truly match the keywords
- **Empty selection (0 URNs) is valid** - if no URNs match the keywords well, select none for that type

### Keyword Matching Examples

**Strong Match (SELECT)**:
- Keyword "chemcam" → URN title contains "ChemCam" ✅
- Keyword "msl" → URN title "Mars Science Laboratory" ✅
- Keyword "spectrometer" → URN description mentions "spectroscopic analysis" ✅

**Weak Match (LIKELY REJECT)**:
- Keyword "rover" → URN about "lander" ❌
- Keyword "Mars" → URN about "Phobos" (Martian moon, not Mars) ❌
- Keywords present → URN about different domain ❌

## Output Format

Return a JSON object with:

```json
{{
  "selected_investigation_urns": ["urn:...", "urn:...", ...],  // 0 to unlimited
  "selected_target_urns": ["urn:...", "urn:...", ...],         // 0 to unlimited
  "selected_instrument_urns": ["urn:...", "urn:...", ...],     // 0 to unlimited
  "reasoning": "Detailed explanation of keyword matches and why URNs were selected or rejected"
}}
```

## Reasoning Requirements

Your reasoning must explain:
1. **For each selected URN**: Which keywords it matched and how strongly
2. **For notable excluded URNs**: Why they didn't match the keywords
3. **Compatibility notes**: Whether selected URNs work together logically
4. **Empty selections**: If you selected 0 URNs for any type, explain why

## Quality Checklist

Before finalizing, verify:
- ✅ Each selected URN has clear keyword matches
- ✅ No Laboratory_Analog, Equipment, or Calibrator targets
- ✅ Selections are compatible with each other
- ✅ Reasoning explains keyword matches
- ✅ Only genuinely relevant URNs are included

**Remember: Keywords are authoritative. If a URN doesn't match the keywords, don't select it.**
