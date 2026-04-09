#!/usr/bin/env python3
"""
Minimal interactive planner for backend/frontend integration.

Shows how to:
1. Start a planning session
2. Handle conversation with questions
3. Generate workflow JSON
"""

import asyncio
from datetime import datetime
import sys

from akd.observability import init_observability
from akd.planner.llm_planner import create_planner


async def run_planner_chat(initial_request: str):
    """Interactive planner session."""
    # Initialize planner
    planner = await create_planner()
    session = await planner.init_planner_session(initial_request)

    # Start conversation
    response = await session.start()
    print(f"Planner: {response.message}")

    # Conversation loop
    while not response.ready_to_generate:
        # Show question if present
        if response.question:
            print(f"\nQuestion: {response.question.question}")
            if response.question.options:
                for i, opt in enumerate(response.question.options, 1):
                    print(f"  {i}. {opt}")

        # Get user input
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            return None

        # Get next response
        response = await session.respond(user_input)
        print(f"\nPlanner: {response.message}")

    # Generate workflow
    if response.ready_to_generate:
        print("\nGenerating workflow...")
        workflow = await session.generate_workflow()

        # Return workflow JSON (this is what backend would return to frontend)
        filename = f"workflow-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        workflow.save_to_file(f"{filename}")
        print(f"Workflow saved to: {filename}")

    return None


if __name__ == "__main__":
    init_observability()
    # Get initial research goal from user
    # get as argument
    initial_request = sys.argv[1]
    asyncio.run(run_planner_chat(initial_request))
    sys.exit(0)

    # sample usage
    # python3 scripts/minimal_planner.py "Find papers on AlphaFold and identify research gaps"
