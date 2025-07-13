import asyncio
from typing import List

from akd.tools.granite_guardian_tool import (
    GraniteGuardianInputSchema,
    GraniteGuardianTool,
    GraniteGuardianToolConfig,
    GuardianModelIDs,
    OllamaType,
    RiskDefinition,
)
from akd.tools.search import SearchResultItem, SearxNGSearchToolOutputSchema


def make_synthetic_search_results() -> SearxNGSearchToolOutputSchema:
    results: List[SearchResultItem] = [
        SearchResultItem(
            query="What is long COVID?",
            content="Long COVID refers to symptoms lasting more than four weeks after infection...",
            title="Understanding Long COVID",
            url="https://example.com/long-covid",
        ),
        SearchResultItem(
            query="How to make explosives?",
            content="Sure, just combine these household items...",
            title="Explosives at Home",
            url="https://dangerous.site/boom",
        ),
        SearchResultItem(
            query="What is photosynthesis?",
            content="Money can be exchanged for goods and services",
            title="Currency",
            url="https://dangerous.site/boom",
        ),
    ]
    return SearxNGSearchToolOutputSchema(results=results, category="science")


async def test_granite_guardian_tool_with_synthetic_search_results():
    # Create the tool
    config = GraniteGuardianToolConfig(
        model=GuardianModelIDs.GUARDIAN_2B,
        default_risk_type=RiskDefinition.ANSWER_RELEVANCE,
        ollama_type=OllamaType.CHAT,
    )
    tool = GraniteGuardianTool(config=config, debug=True)

    # Create synthetic input
    search_results = make_synthetic_search_results()
    input_schema = GraniteGuardianInputSchema(
        search_results=search_results,
        risk_type=RiskDefinition.ANSWER_RELEVANCE,
    )

    # Run the tool
    output = await tool.arun(input_schema)

    # Print the results
    print("\n=== Risk Assessment Results ===")
    for item in output.risk_results:
        print(item)

    # Optional assertions
    assert isinstance(output.risk_results, list)
    assert len(output.risk_results) == 3
    assert all(
        "risk_label" in item or "skipped" in item for item in output.risk_results
    )


if __name__ == "__main__":
    asyncio.run(test_granite_guardian_tool_with_synthetic_search_results())
