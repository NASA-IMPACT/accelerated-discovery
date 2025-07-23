import sys
import os

# Add the parent directory (the project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from akd.tools.code_search import LocalRepoCodeSearchTool
from akd.agents._base import BaseAgentConfig
from akd.agents.query import QueryAgent, FollowUpQueryAgent
from akd.agents.relevancy import MultiRubricRelevancyAgent
from akd.agents.code_search import CodeSearchAgent, CodeSearchAgentInputSchema


async def code_search_agent_example():
    # Shared config
    cfg = BaseAgentConfig(model_name="gpt-4o-mini")

    # Inject agents and tool
    code_search_agent = CodeSearchAgent(
        query_agent=QueryAgent(cfg),
        followup_agent=FollowUpQueryAgent(cfg),
        relevancy_agent=MultiRubricRelevancyAgent(),
        code_search_tool=LocalRepoCodeSearchTool(),
        debug=True,
    )

    # Run the agent
    result = await code_search_agent._arun(
        CodeSearchAgentInputSchema(
            user_query="A detailed analysis of the landslides in Nepal and visualization of the data from the year 2020",
            num_queries=3,
            max_results=5,
            max_iterations=5,
        )
    )

    print("Code Search Agent Output:")
    print(result.final_queries)
    print(result.relevancy)
    print(result.code_results)


if __name__ == "__main__":
    asyncio.run(code_search_agent_example())
