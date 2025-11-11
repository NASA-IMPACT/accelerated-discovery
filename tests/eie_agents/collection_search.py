import asyncio

from akd.agents.eie_agents.collection_search import (
    CollectionSearchAgent,
    CollectionSearchInputSchema,
)

agent = CollectionSearchAgent()
params = CollectionSearchInputSchema(
    dataset_type="sulfur",
    location="USA",
    bbox="",
    frequency="all",
    temporal_extent={"dates": ["201--12-25", "2011-12-25", "2020-12-25"]},
)


async def main():
    result = await agent.arun(params)
    print("result is:", result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
