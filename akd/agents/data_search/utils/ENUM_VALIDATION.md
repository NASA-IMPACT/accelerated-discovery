# CMR Enum Validation

## Overview

The CMR Enum Validator automatically corrects instrument and platform values in query approaches by fuzzy matching against NASA's official CMR controlled vocabularies. This ensures queries use the exact terminology expected by CMR's API, improving search accuracy.

## Architecture

### Components

**CMREnumValidator** (`cmr_enum_validator.py`)
- Validates instrument/platform pairs in CMRQueryApproach objects
- Uses fuzzy matching with configurable similarity threshold (default: 0.7)
- Implements 5 correction scenarios based on match results
- Returns corrected approach + metadata about changes made

**Fuzzy Matcher** (`improved_fuzzy_matcher.py`)
- Searches both instrument and platform registries simultaneously
- Returns best match with type (instrument/platform) and similarity score
- Uses TF-IDF vectorization for efficient matching

**CMR Enumerations** (`cmr_enums/`)
- `cmr_instruments.json` - 1,928 official instrument names
- `cmr_platforms.json` - 1,096 official platform names

## Integration in Pipeline

The validator runs in the CMR handler after known parameters extraction:

```python
# handler.py:172-181
for approach in known_params_output.query_approaches:
    corrected_approach, corrections_metadata = self.enum_validator.validate_approach(
        approach,
    )
    validated_approaches.append(corrected_approach)
    all_corrections.append(corrections_metadata)
```

Corrections are then logged at two levels:
- **Decomposition level**: Summary array of all approach corrections
- **Query level**: Specific corrections for each searchable query (via `approach_index` lookup)

## Correction Scenarios

The validator implements 5 scenarios based on fuzzy match results:

### A. Simple Replacement

Both fields match their intended types - just correct the spelling/casing.

```
Input:   instrument='modis', platform='terra'
Matches: instrument='MODIS' (0.95), platform='Terra' (0.98)
Action:  Replace both with exact CMR values
Result:  instrument='MODIS', platform='Terra'
```

### B. Single Swap

One field is misidentified as the other type.

```
Input:   instrument='Terra', platform=None
Matches: instrument → platform='Terra' (0.99)
Action:  Move value to correct field, drop incorrect field
Result:  instrument=None, platform='Terra'
```

### C. Double Swap

Both fields are reversed - instrument value is actually a platform and vice versa.

```
Input:   instrument='Aqua', platform='MODIS'
Matches: instrument → platform='Aqua' (0.99), platform → instrument='MODIS' (0.99)
Action:  Swap both values to correct fields
Result:  instrument='MODIS', platform='Aqua'
```

### D. Biased Swap (Different Scores)

Both fields match to the same type - keep the higher-scoring match.

```
Input:   instrument='MODIS', platform='VIIRS'
Matches: Both → instrument (MODIS=0.99, VIIRS=0.85)
Action:  Keep higher score, drop lower score
Result:  instrument='MODIS', platform=None
```

### E. Biased Swap (Same Scores)

Both fields match to same type with equal scores - keep original field match.

```
Input:   instrument='MODIS', platform='VIIRS'
Matches: Both → instrument (MODIS=0.92, VIIRS=0.92)
Action:  Keep match from original field (instrument), drop platform
Result:  instrument='MODIS', platform=None
```

## Output Format

### Corrections Metadata

```json
{
  "corrections_applied": true,
  "changes": [
    {
      "field": "instrument",
      "original": "modis",
      "corrected": "MODIS",
      "score": 0.95
    },
    {
      "field": "platform",
      "original": "terra",
      "corrected": "Terra",
      "score": 0.98
    }
  ]
}
```

### Special Change Flags

**Swapped**: Field value moved from another field
```json
{
  "field": "platform",
  "original": "MODIS",
  "corrected": "Terra",
  "score": 0.99,
  "swapped": true
}
```

**Moved From**: Indicates source field for swapped value
```json
{
  "field": "platform",
  "original": "Terra",
  "corrected": "Terra",
  "score": 0.99,
  "moved_from": "instrument"
}
```

**Dropped**: Field value removed (wrong type or lower score)
```json
{
  "field": "instrument",
  "original": "Terra",
  "corrected": null,
  "score": 0.0,
  "dropped": true,
  "reason": "lower_score"
}
```

## Configuration

```python
validator = CMREnumValidator(
    threshold=0.7,              # Minimum similarity score (0.0-1.0)
    debug=True,                 # Enable correction logging
    instruments_path=None,      # Optional custom instruments JSON
    platforms_path=None,        # Optional custom platforms JSON
)
```

## Logging Output

When `debug=True`, corrections are logged:

```
Corrected instrument: 'modis' → 'MODIS' (score=0.95)
Corrected platform: 'terra' → 'Terra' (score=0.98)
Applied enum corrections to 3/4 approaches
```

## Business Rules Summary

| Scenario | Condition | Action |
|----------|-----------|--------|
| Simple Replacement | Both match correct types | Replace values with exact CMR names |
| Single Swap | One field is wrong type | Move to correct field, drop wrong field |
| Double Swap | Both fields reversed | Swap both to correct fields |
| Biased Swap (Diff) | Both match same type, different scores | Keep higher score, drop lower |
| Biased Swap (Same) | Both match same type, same score | Keep original field match, drop other |

## Testing

See `examples/testing/test_enum_validator.py` for unit tests covering all scenarios.

## Performance

- **Initialization**: ~50ms (loads and indexes 3,024 enum values)
- **Validation**: ~10-20ms per approach (2 fuzzy matches + logic)
- **Caching**: Fuzzy matcher uses TF-IDF vectorization for fast lookups

## Notes

- Validation runs **after** known parameters extraction, **before** searchable parameters generation
- Empty/null instrument and platform values skip validation
- Values below threshold are not corrected (treated as no match)
- The validator is stateless - each `validate_approach()` call is independent
