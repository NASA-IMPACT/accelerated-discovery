import asyncio

from akd.agents.eie_agents.collection_search import (
    CollectionSearchAgent,
    CollectionSearchInputSchema,
)

agent = CollectionSearchAgent()
params = CollectionSearchInputSchema(
    dataset_type="night light",
    location="nepal",
    bbox="",
    frequency="all",
    temporal_extent={"dates": {"start": "2019-01-01T00:00:00Z", "end": "2019-01-20T00:00:00Z"}},
)


async def main():
    result = await agent.arun(params)
    print("result is:", result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
