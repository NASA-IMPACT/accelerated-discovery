import asyncio

from akd.agents.eie_agents.eie_extraction import ExtractAgent, ExtractInputSchema

agent = ExtractAgent()
params = ExtractInputSchema(query="Show me methane levels over Nepal from 2020 to 2024")


async def main():
    result = await agent.arun(params)
    print("result is:", result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
