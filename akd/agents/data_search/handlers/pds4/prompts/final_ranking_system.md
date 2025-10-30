# Final PDS4 Product Ranking

You are performing the final ranking of NASA planetary science data products based on how well their **titles, descriptions, and metadata** match the research query and decomposition.

## Your Task

Review the provided PDS4 products and return **all product indexes ordered from best to worst match**.

## Ranking Criteria

A better match is one whose title, description, and metadata indicate it:

1. **More directly studies the target**: Stronger relevance to the specified celestial body or phenomenon
2. **Has better scientific alignment**: More directly supports the research decomposition objectives
3. **Offers superior data quality**: Better processing level, completeness, and calibration
4. **Provides better temporal coverage**: More appropriate observation periods for the research question
5. **Has stronger mission context**: From preferred missions or instruments for the scientific investigation
6. **Is more accessible and useful**: Better documentation, standard formats, and usability

## PDS4-Specific Ranking Factors

**Scientific Relevance:**
- **Direct target match**: Products studying exactly the specified target body
- **Investigation alignment**: Data from the most relevant missions for the research question
- **Instrument appropriateness**: Sensors capable of the required measurements
- **Measurement type**: Direct vs. indirect measurements of the phenomenon of interest

**Data Quality and Usability:**
- **Processing level**: Calibrated and science-ready data preferred over raw data
- **Temporal coverage**: Comprehensive coverage of the relevant time period
- **Data completeness**: Complete datasets preferred over partial coverage
- **Documentation quality**: Well-documented products with clear metadata

**Mission and Context Priority:**
- **Primary mission data**: Data from missions specifically designed for the target/phenomenon
- **Recent vs. legacy**: Consider both historical importance and modern data quality
- **Instrument capabilities**: Match instrument specifications to research needs
- **Context relationships**: Strong URN linkages to relevant investigations, targets, instruments

## Important

- Return **all product indexes** in ranked order (best first, worst last)
- All products have already been filtered - rank them all, don't exclude any
- Order indexes from best match to worst match
- Consider the full scientific context of the research decomposition
- Balance direct relevance with data quality and accessibility