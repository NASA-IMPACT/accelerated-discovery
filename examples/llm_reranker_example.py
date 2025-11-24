"""
Example script demonstrating the LLM-based reranker.

This example shows how to:
1. Configure the LLM reranker with custom criteria and categories
2. Provide field descriptions for domain-specific fields
3. The reranker creates its own internal scoring agent
4. Rerank search results using LLM-based individual scoring (single call per result evaluates all criteria)
5. Access detailed scores for post-hoc analysis
"""

import asyncio

from akd.structures import SearchResultItem
from akd.tools.reranker import (
    LLMRerankerToolConfig,
    ScoringCategory,
    ScoringCriterion,
    create_reranker,
)


async def main():
    """Demonstrate LLM-based reranking with single-call evaluation."""

    # Step 1: Configure the reranker
    # The LLMRerankerTool creates its own internal scoring agent

    reranker_config = LLMRerankerToolConfig(
        model_name="gpt-4o-mini",
        temperature=0.0,
        fields_to_evaluate={
            "title": "The title or name of the dataset",
            "content": "Description or abstract of the dataset",
            "spatial_resolution": "Ground sampling distance - lower values (e.g., 30m) are higher resolution",
            "processing_level": "Data processing level (L0=raw, L1=calibrated/geolocated, L2=derived products)",
        },
        scoring_criteria=[
            ScoringCriterion(
                name="Relevancy",
                description="How well does this result match the user's query intent?",
                weight=0.5,
                scoring_categories=[
                    ScoringCategory(
                        name="Highly Relevant",
                        description="Result directly addresses the query with highly relevant content and matches all key aspects",
                        value=3.0,
                    ),
                    ScoringCategory(
                        name="Somewhat Relevant",
                        description="Result is partially relevant but may lack specificity or miss some aspects of the query",
                        value=2.0,
                    ),
                    ScoringCategory(
                        name="Not Relevant",
                        description="Result is irrelevant or off-topic for this query",
                        value=0.0,
                    ),
                ],
            ),
            ScoringCriterion(
                name="Quality",
                description="Is this a high-quality, reliable source?",
                weight=0.3,  # 30% weight
                scoring_categories=[
                    ScoringCategory(
                        name="High Quality",
                        description="Authoritative, well-documented source with verified and comprehensive information",
                        value=3.0,
                    ),
                    ScoringCategory(
                        name="Moderate Quality",
                        description="Acceptable quality but may lack depth, completeness, or authoritative backing",
                        value=2.0,
                    ),
                    ScoringCategory(
                        name="Low Quality",
                        description="Unreliable, poorly documented, or questionable source with insufficient information",
                        value=0.0,
                    ),
                ],
            ),
            ScoringCriterion(
                name="Ease of Use",
                description="How easy would it be for a user to work with this result?",
                weight=0.2,  # 20% weight
                scoring_categories=[
                    ScoringCategory(
                        name="Easy to Use",
                        description="Ready to use with clear documentation, accessible format, and minimal setup required",
                        value=3.0,
                    ),
                    ScoringCategory(
                        name="Moderate Effort",
                        description="Usable but requires some effort, additional processing, or navigation to extract value",
                        value=2.0,
                    ),
                    ScoringCategory(
                        name="Difficult to Use",
                        description="Hard to access, poorly documented, requires significant processing, or has major usability barriers",
                        value=0.0,
                    ),
                ],
            ),
        ],
        # Approach 1: just send these data and ask llm to
        # query, result (fields), Relevancy How well does this result match the user's query intent? ->
        # MODIS Climate Data
        # Approach 2 (advanced not implemented) - for each critera
        # 1. has to mention modis
        # 2. has to mention data
    )

    reranker = create_reranker(
        reranker_type="llm",
        config=reranker_config,
        debug=True,
    )

    query = "high resolution satellite imagery for climate analysis"
    results = [
        SearchResultItem(
            query=query,
            title="MODIS Climate Data",
            content="Moderate Resolution Imaging Spectroradiometer data for climate studies",
            url="https://example.com/modis",
            extra={"spatial_resolution": "250m", "processing_level": "L2"},
        ),
        SearchResultItem(
            query=query,
            title="Random Blog Post",
            content="Some random blog about cameras",
            url="https://example.com/blog",
            extra={},
        ),
        SearchResultItem(
            query=query,
            title="Landsat High-Res Imagery",
            content="High resolution Landsat satellite imagery with climate applications",
            url="https://example.com/landsat",
            extra={"spatial_resolution": "30m", "processing_level": "L1"},
        ),
    ]

    print(f"Query: {query}")
    print(f"Number of results: {len(results)}")
    print(f"Number of criteria: {len(reranker_config.scoring_criteria)}")
    print(
        f"\nPERFORMANCE: Using single-call evaluation - {len(results)} results × 1 call/result = {len(results)} total LLM calls",
    )
    print(
        f"            (vs. old approach: {len(results)} results × {len(reranker_config.scoring_criteria)} criteria = {len(results) * len(reranker_config.scoring_criteria)} calls)\n",
    )

    reranked_output = await reranker.arun(
        reranker.input_schema(query=query, results=results),
    )

    print(f"\nReranked order: {[r.title for r in reranked_output.results]}")

    print("\n" + "=" * 80)
    print("DETAILED SCORES (All criteria evaluated in single LLM call per result)")
    print("=" * 80)

    for idx, result in enumerate(reranked_output.results, 1):
        print(f"\n{idx}. {result.title}")
        print(f"   Total Score: {result.score:.3f}")

        if "llm_reranker" in result.extra:
            criterion_scores = result.extra["llm_reranker"]["criterion_scores"]
            for criterion_name, score_data in criterion_scores.items():
                print(
                    f"   - {criterion_name}: {score_data['category']} "
                    f"(score={score_data['score']:.1f}, weight={score_data['weight']:.2f})",
                )
                print(f"     Reasoning: {score_data['reasoning']}")

    print("\n" + "=" * 80)
    print("KEY FEATURES:")
    print("- Single LLM call evaluates ALL criteria simultaneously (efficient)")
    print("- Scores logged and stored in result.extra['llm_reranker']")
    print("- Adjust weights in config without re-running LLM")
    print("- Compare different weight combinations against SME ground truth")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
