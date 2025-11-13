import asyncio

from akd.agents.eie_agents.eie_extraction import ExtractAgent, ExtractInputSchema

agent = ExtractAgent()
params = ExtractInputSchema(query="daily methane data for the USA between 2020 and 2025")


async def main():
    result = await agent.arun(params)
    print("result is:", result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
