import asyncio

from akd.agents.eie_agents.collection_search import (
    CollectionSearchAgent,
    CollectionSearchInputSchema,
)
from akd.agents.eie_agents.eie_extraction import TemporalExtent

agent = CollectionSearchAgent()
params = CollectionSearchInputSchema(
    dataset_type="methane",
    location="USA",
    frequency="yearly",
    temporal_extent=TemporalExtent(start="2020-01-01", end="2024-01-01"),
)


async def main():
    result = await agent.arun(params)
    print("result is:", result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
