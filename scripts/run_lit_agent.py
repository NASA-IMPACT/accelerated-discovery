# noqa: D100,D101,D102,D103,D104,D105,D107,D200,D201,D202,D203,D204,D205,D400,D401,F841
import argparse
import asyncio
import json

from loguru import logger

# Removed unused imports
from akd.agents.search import ControlledSearchAgent, LitSearchAgentInputSchema

# Removed unused import
# Removed unused imports


async def main(args):
    # Removed unused variable assignments

    # Use the new ControlledAgenticLitSearchAgent with proper configuration
    from akd.agents.search import ControlledSearchAgentConfig

    agent_config = ControlledSearchAgentConfig(debug=True)
    lit_agent = ControlledSearchAgent(config=agent_config, debug=True)

    result = await lit_agent.arun(
        LitSearchAgentInputSchema(query=args.query, max_results=5),
    )
    logger.info(result.model_dump())

    with open("./temp/test_lit_agent.json", "w") as f:
        f.write(json.dumps(result.model_dump(mode="json")["results"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LitAgent pipeline")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The query to run through the LitAgent pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/lit_agent.toml",
        help="Path to the TOML config file for LitAgent",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
