import asyncio

from akd.agents.eie_agents.collection_search import (
    CollectionSearchAgent,
    CollectionSearchInputSchema,
)

agent = CollectionSearchAgent()
params = CollectionSearchInputSchema(
    dataset_type="methane",
    location="USA",
    bbox=[-179.1478506, 18.9111686, 179.7726013, 71.3868277],
    frequency="monthly",
    temporal_extent={"dates": {"start": "2020-01-01T00:00:00Z", "end": "2023-12-31T23:59:59Z"}},
)


async def main():
    result = await agent.arun(params)
    print("result is:", result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
