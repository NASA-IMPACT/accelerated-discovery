# Rank PDS4 Products

## What We're Looking For

**Original Query**: {original_query}

**Decomposition**: {decomposition_title}
{decomposition_justification}

## PDS4 Products to Rank ({num_items} total)

{items_list}

## Your Task

Return **all {num_items} product indexes ordered from best to worst match**.

Rank products based on whether their **title, description, and metadata** indicate they:
- Study the target body or phenomenon described in the decomposition
- Provide the type of measurements needed for the scientific investigation
- Come from appropriate missions and instruments for the research question
- Have suitable data quality, processing level, and temporal coverage
- Offer the best scientific value for addressing the research decomposition

## PDS4-Specific Ranking Considerations

**Scientific Alignment:**
- Products that directly study the target body/phenomenon in the decomposition
- Data from missions specifically designed to investigate the research question
- Instruments capable of the required measurements (spectrometry, imaging, etc.)
- Datasets that provide direct evidence for the scientific investigation

**Data Quality and Utility:**
- Appropriately processed data (calibrated, science-ready)
- Complete temporal and spatial coverage relevant to the research
- Well-documented datasets with clear scientific context
- Data from reliable, validated missions and instruments

**Mission and Context Priority:**
- Primary mission data for the target body (e.g., Mars rovers for Mars surface studies)
- Recent high-quality data vs. historical datasets (balance recency with completeness)
- Data with strong context relationships (investigation → target → instrument linkages)
- Datasets with established scientific value and community usage

Return indexes in order: [best_index, second_best_index, ..., worst_index]