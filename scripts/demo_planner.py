#!/usr/bin/env python3
"""
Demo CLI for the AKD LLM-based workflow planner.

This script provides an interactive and non-interactive CLI for demonstrating
the planner functionality and showcasing workflow generation capabilities.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from akd.observability import init_observability
from akd.planner.llm_planner import create_planner, quick_plan
from akd.planner.registry import get_agent_registry
from akd.utils import to_snake_case

app = typer.Typer(help="Demo CLI for the AKD LLM Workflow Planner")
console = Console()

# Global flag for non-interactive mode
NON_INTERACTIVE = False


def _register_ext_agents():
    """Register akd-ext agents and disable base registry agents.

    Only akd-ext agents (CMRCareAgent, CodeSearchCareAgent) should be used
    by the planner. Base agents (deep_search, gap_analysis, code_search) are
    disabled so the planner doesn't see them.

    Disabling happens only after akd-ext imports succeed to avoid leaving
    the registry empty if akd-ext is not installed.
    """
    registry = get_agent_registry()

    # Register akd-ext agents first — only disable base agents if this succeeds
    try:
        from akd_ext.agents.cmr_care import CMRCareAgent
        from akd_ext.agents.code_search_care import CodeSearchCareAgent
        from akd_ext.agents.gap import GapAgent as ExtGapAgent
        from akd_ext.agents.closed_loop_cm1 import (
            CapabilityFeasibilityMapperAgent,
            WorkflowSpecBuilderAgent,
            ExperimentImplementationAgent,
            InterpretationPaperAssemblyAgent,
        )

        ext_agents = [
            CMRCareAgent,
            CodeSearchCareAgent,
            ExtGapAgent,
            CapabilityFeasibilityMapperAgent,
            WorkflowSpecBuilderAgent,
            ExperimentImplementationAgent,
            InterpretationPaperAssemblyAgent,
        ]

        for agent_cls in ext_agents:
            agent_id = to_snake_case(agent_cls.__name__)
            if not registry.get_agent(agent_id):
                registry.register_agent(agent_cls)
                logger.info(f"Registered {agent_cls.__name__} from akd-ext")

        # Disable base registry agents now that akd-ext agents are available
        for agent_id in ["deep_search", "gap_analysis", "code_search"]:
            agent = registry.get_agent(agent_id)
            if agent and agent.enabled:
                registry.update_agent(agent_id, enabled=False)
                logger.info(f"Disabled base agent: {agent_id}")

    except ImportError:
        logger.debug("akd-ext not installed, keeping base agents enabled")
    except Exception as e:
        logger.warning(f"Failed to register akd-ext agents: {e}")


def print_header():
    """Print the application header."""
    console.print(
        Panel.fit(
            "[bold blue]AKD LLM Workflow Planner[/bold blue]\nInteractive research workflow planning system",
            border_style="blue",
        )
    )


def print_agent_registry():
    """Display available agents from the registry."""
    try:
        registry = get_agent_registry()
        agents = registry.get_enabled_agents()

        if not agents:
            console.print("[yellow]No enabled agents found in registry.[/yellow]")
            return

        table = Table(title="Available Agents")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description", style="white")
        table.add_column("Input Fields", style="blue")
        table.add_column("Output Fields", style="magenta")

        for agent in agents:
            input_count = len(agent.input_schema.fields)
            output_count = len(agent.output_schema.fields)
            table.add_row(
                agent.agent_id,
                agent.name,
                agent.description[:60] + "..." if len(agent.description) > 60 else agent.description,
                str(input_count),
                str(output_count),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error loading agent registry: {e}[/red]")


async def interactive_planning_session():
    """Run an interactive planning session."""
    print_header()
    _register_ext_agents()

    # Show available agents
    console.print("\n[bold]Available Agents:[/bold]")
    print_agent_registry()

    # Get initial research goal
    console.print("\n[bold]Let's start planning your research workflow![/bold]")
    initial_request = Prompt.ask(
        "\n[green]What is your research goal or question?[/green]",
        default="I want to analyze the current state of the art in drug discovery",
    )

    try:
        # Create planner and start session
        console.print("\n[blue]Starting planning session...[/blue]")
        planner = await create_planner()
        session = await planner.init_planner_session(initial_request)

        # Start the conversation
        response = await session.start()

        console.print(f"\n[bold cyan]Planner:[/bold cyan] {response.message}")

        # Continue conversation until ready to generate
        while not response.ready_to_generate:
            if response.question:
                console.print(f"\n[yellow]Question:[/yellow] {response.question.question}")
                if response.question.context:
                    console.print(f"[dim]{response.question.context}[/dim]")

                if response.question.options:
                    console.print("\n[blue]Options:[/blue]")
                    for i, option in enumerate(response.question.options, 1):
                        console.print(f"  {i}. {option}")

                user_input = Prompt.ask("\n[green]Your response[/green]")
            else:
                user_input = Prompt.ask("\n[green]Continue the conversation[/green]")

            if user_input.lower() in ["quit", "exit", "done"]:
                break

            response = await session.respond(user_input)
            console.print(f"\n[bold cyan]Planner:[/bold cyan] {response.message}")

        # Generate workflow if ready
        if response.ready_to_generate:
            console.print("\n[bold green]✓ Workflow planning complete![/bold green]")

            if response.workflow_plan:
                console.print("\n[bold]Research Goal:[/bold]")
                console.print(f"  {response.workflow_plan.research_goal}")
                console.print("\n[bold]Workflow Steps:[/bold]")
                for i, step in enumerate(response.workflow_plan.workflow_steps, 1):
                    console.print(f"  {i}. {step}")

            generate = Confirm.ask("\n[yellow]Generate executable workflow file?[/yellow]", default=True)

            if generate:
                console.print("\n[blue]Generating executable workflow...[/blue]")
                workflow = await session.generate_workflow()

                # Display workflow JSON
                workflow_json = workflow.to_json()
                syntax = Syntax(workflow_json, "json", theme="monokai", line_numbers=True)
                console.print("\n[bold]Generated Workflow:[/bold]")
                console.print(syntax)

                # Display workflow summary
                console.print("\n[bold]Workflow Summary:[/bold]")
                console.print(f"Workflow Type: {workflow.workflow_type}")
                console.print(f"Version: {workflow.version}")
                console.print(f"Total Nodes: {len(workflow.nodes)}")
                console.print(f"Total Edges: {len(workflow.edges)}")

                # Show nodes with io_map
                nodes_with_io_map = [node for node in workflow.nodes if node.io_map]
                console.print(f"Nodes with io_map: {len(nodes_with_io_map)}")

                if nodes_with_io_map:
                    table = Table(title="Runtime Data Flow (io_map)")
                    table.add_column("Node", style="cyan")
                    table.add_column("Field", style="green")
                    table.add_column("JSONPath Source", style="yellow")

                    for node in nodes_with_io_map:
                        for field, jsonpath in node.io_map.items():
                            table.add_row(node.type_, field, jsonpath)

                    console.print(table)

                # Auto-save to file
                filename = Prompt.ask("\n[yellow]Save workflow as[/yellow]", default="generated_workflow.json")
                workflow.save_to_file(filename)
                console.print(f"\n[green]✓ Workflow saved to: {filename}[/green]")
                console.print("[dim]You can now execute this workflow using the AKD execution engine.[/dim]")

    except Exception as e:
        console.print(f"[red]Error in planning session: {e}[/red]")
        logger.exception("Planning session error")


async def automated_planning_session(
    initial_request: str,
    hardcoded_responses: Optional[list[str]] = None,
    output_file: Optional[str] = None,
    max_turns: int = 10,
    verbose: bool = True,
):
    """
    Run an automated planning session with hardcoded responses.

    Args:
        initial_request: The initial research goal
        hardcoded_responses: List of hardcoded user responses (default: ["yes", "continue", "proceed"])
        output_file: Optional file to save the generated workflow
        max_turns: Maximum number of conversation turns (default: 10)
        verbose: Whether to show detailed output (default: True)

    Returns:
        WorkflowFormat object if successful, None otherwise
    """
    if hardcoded_responses is None:
        hardcoded_responses = ["yes", "continue", "proceed", "both", "existing studies", "methodologies"]

    if verbose:
        print_header()
        console.print("\n[bold blue]Automated Planning Session[/bold blue]")
        console.print(f"Initial Request: [green]{initial_request}[/green]")
        console.print(f"Hardcoded Responses: [dim]{hardcoded_responses}[/dim]")
        console.print(f"Max Turns: {max_turns}\n")

    _register_ext_agents()

    try:
        # Create planner and start session
        if verbose:
            console.print("[blue]Starting planning session...[/blue]")

        planner = await create_planner()
        session = await planner.init_planner_session(initial_request)

        # Start the conversation
        response = await session.start()

        if verbose:
            console.print(f"\n[bold cyan]Planner:[/bold cyan] {response.message}")

        # Continue conversation until ready to generate
        turn = 0
        response_idx = 0

        while not response.ready_to_generate and turn < max_turns:
            turn += 1

            # Determine user input
            if response.question:
                if verbose:
                    console.print(f"\n[yellow]Question:[/yellow] {response.question.question}")
                    if response.question.context:
                        console.print(f"[dim]{response.question.context}[/dim]")
                    if response.question.options:
                        console.print("[blue]Options:[/blue]")
                        for i, option in enumerate(response.question.options, 1):
                            console.print(f"  {i}. {option}")

                # Use hardcoded response (cycle through list)
                user_input = hardcoded_responses[response_idx % len(hardcoded_responses)]
                response_idx += 1
            else:
                # No question, just continue
                user_input = "continue"

            if verbose:
                console.print(f"\n[green]User (automated):[/green] {user_input}")

            response = await session.respond(user_input)

            if verbose:
                console.print(f"\n[bold cyan]Planner:[/bold cyan] {response.message}")

        # Check if we hit max turns
        if turn >= max_turns and not response.ready_to_generate:
            console.print(f"\n[red]✗ Max turns ({max_turns}) reached without completing workflow plan[/red]")
            return None

        # Generate workflow if ready
        if response.ready_to_generate:
            if verbose:
                console.print("\n[bold green]✓ Workflow planning complete![/bold green]")

            if response.workflow_plan:
                if verbose:
                    console.print("\n[bold]Research Goal:[/bold]")
                    console.print(f"  {response.workflow_plan.research_goal}")
                    console.print("\n[bold]Workflow Steps:[/bold]")
                    for i, step in enumerate(response.workflow_plan.workflow_steps, 1):
                        console.print(f"  {i}. {step}")

            # Generate workflow automatically
            if verbose:
                console.print("\n[blue]Generating executable workflow...[/blue]")

            workflow = await session.generate_workflow()

            if verbose:
                # Display workflow JSON
                workflow_json = workflow.to_json()
                syntax = Syntax(workflow_json, "json", theme="monokai", line_numbers=True)
                console.print("\n[bold]Generated Workflow:[/bold]")
                console.print(syntax)

                # Display workflow summary
                console.print("\n[bold]Workflow Summary:[/bold]")
                console.print(f"Workflow Type: {workflow.workflow_type}")
                console.print(f"Version: {workflow.version}")
                console.print(f"Total Nodes: {len(workflow.nodes)}")
                console.print(f"Total Edges: {len(workflow.edges)}")

                # Show nodes with io_map
                nodes_with_io_map = [node for node in workflow.nodes if node.io_map]
                console.print(f"Nodes with io_map: {len(nodes_with_io_map)}")

                if nodes_with_io_map:
                    table = Table(title="Runtime Data Flow (io_map)")
                    table.add_column("Node", style="cyan")
                    table.add_column("Field", style="green")
                    table.add_column("JSONPath Source", style="yellow")

                    for node in nodes_with_io_map:
                        for field, jsonpath in node.io_map.items():
                            table.add_row(node.type_, field, jsonpath)

                    console.print(table)

            # Save to file if specified
            if output_file:
                workflow.save_to_file(output_file)
                console.print(f"\n[green]✓ Workflow saved to: {output_file}[/green]")
            elif verbose:
                # Auto-save with default name
                default_filename = "automated_workflow.json"
                workflow.save_to_file(default_filename)
                console.print(f"\n[green]✓ Workflow saved to: {default_filename}[/green]")

            return workflow

        else:
            console.print("\n[yellow]Workflow not ready to generate[/yellow]")
            return None

    except Exception as e:
        console.print(f"[red]Error in automated planning session: {e}[/red]")
        logger.exception("Automated planning session error")
        return None


async def demo_workflow_display():
    """Demonstrate loading and displaying the sample workflow format."""
    console.print("\n[bold]Sample Workflow Display[/bold]")

    try:
        # Load sample workflow
        sample_path = Path(__file__).parent.parent / "akd" / "planner-format-sample.json"

        if not sample_path.exists():
            console.print(f"[red]Sample workflow file not found: {sample_path}[/red]")
            return

        from akd.planner.format_builder import WorkflowFormat

        workflow = WorkflowFormat.from_file(str(sample_path))
        console.print("[green]Loaded sample workflow[/green]")

        # Display workflow
        workflow_json = workflow.to_json()
        syntax = Syntax(workflow_json, "json", theme="monokai", line_numbers=True)
        console.print("\n[bold]Sample Workflow Format:[/bold]")
        console.print(syntax)

        # Display summary
        console.print("\n[bold]Workflow Summary:[/bold]")
        console.print(f"Workflow Type: {workflow.workflow_type}")
        console.print(f"Version: {workflow.version}")
        console.print(f"Total Nodes: {len(workflow.nodes)}")
        console.print(f"Total Edges: {len(workflow.edges)}")

        # Show nodes with io_map
        console.print("\n[bold]Runtime Data Flow (io_map):[/bold]")
        for node in workflow.nodes:
            console.print(f"\nNode: [cyan]{node.type_}[/cyan]")
            if node.io_map:
                for field, jsonpath in node.io_map.items():
                    console.print(f"  {field} ← [yellow]{jsonpath}[/yellow]")
            else:
                console.print("  [dim](No io_map - receives user input)[/dim]")

    except Exception as e:
        console.print(f"[red]Error in workflow display: {e}[/red]")
        logger.exception("Workflow display error")


@app.command()
def interactive():
    """Run an interactive planning session."""
    asyncio.run(interactive_planning_session())


@app.command()
def automated(
    goal: str = typer.Argument(..., help="Research goal or question"),
    responses: Optional[str] = typer.Option(
        None, "-r", "--responses", help="Comma-separated list of hardcoded responses"
    ),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output file for generated workflow"),
    max_turns: int = typer.Option(10, "--max-turns", help="Maximum conversation turns"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Minimal output (non-verbose)"),
):
    """
    Run an automated planning session with hardcoded responses.

    Examples:
        python scripts/demo_planner.py automated "Find papers on AlphaFold"
        python scripts/demo_planner.py automated "Carbon recovery research" -r "existing studies,both" -o workflow.json
        python scripts/demo_planner.py automated "Drug discovery trends" --max-turns 5 --quiet
    """

    async def _automated_session():
        # Parse responses if provided
        response_list = None
        if responses:
            response_list = [r.strip() for r in responses.split(",")]

        workflow = await automated_planning_session(
            initial_request=goal,
            hardcoded_responses=response_list,
            output_file=output,
            max_turns=max_turns,
            verbose=not quiet,
        )

        if workflow:
            if quiet:
                console.print("[green]✓ Workflow generated successfully[/green]")
            return 0
        else:
            if quiet:
                console.print("[red]✗ Workflow generation failed[/red]")
            return 1

    exit_code = asyncio.run(_automated_session())
    sys.exit(exit_code or 0)


@app.command()
def sample():
    """Display the sample workflow format."""
    asyncio.run(demo_workflow_display())


@app.command()
def agents():
    """Display available agents in the registry."""
    print_header()
    console.print("\n[bold]Agent Registry:[/bold]")
    print_agent_registry()


@app.command()
def quick(
    goal: str = typer.Argument(..., help="Research goal or question"),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output file for generated workflow"),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", "-n", help="Run without prompts (auto-save if output specified)"
    ),
):
    """Quick workflow generation for a research goal."""

    async def _quick_plan():
        _register_ext_agents()
        try:
            if not non_interactive:
                console.print(f"[blue]Planning workflow for: {goal}[/blue]")

            session = await quick_plan(goal)
            response = await session.start()

            if not non_interactive:
                console.print(f"\n[bold cyan]Planner Response:[/bold cyan] {response.message}")

            if response.workflow_plan:
                if not non_interactive:
                    console.print("\n[bold]Quick Plan Generated:[/bold]")
                    console.print(f"Research Goal: {response.workflow_plan.research_goal}")
                    console.print(
                        f"Suggested Agents: {[a.agent_name for a in response.workflow_plan.suggested_agents]}"
                    )

                # Try to generate workflow automatically
                try:
                    workflow = await session.generate_workflow()
                    workflow_json = workflow.to_json()

                    if output:
                        workflow.save_to_file(output)
                        console.print(f"[green]Workflow saved to {output}[/green]")
                    elif non_interactive:
                        # Non-interactive mode: just print JSON
                        console.print(workflow_json)
                    else:
                        # Interactive mode: formatted display
                        syntax = Syntax(workflow_json, "json", theme="monokai")
                        console.print("\n[bold]Generated Workflow:[/bold]")
                        console.print(syntax)

                        # Show io_map summary
                        nodes_with_io_map = [node for node in workflow.nodes if node.io_map]
                        if nodes_with_io_map:
                            console.print(
                                f"\n[green]✓ Generated {len(nodes_with_io_map)} node(s) with runtime data flow (io_map)[/green]"
                            )

                except Exception as e:
                    if non_interactive:
                        console.print(f"Error: {e}", file=sys.stderr)
                        sys.exit(1)
                    else:
                        console.print(f"[yellow]Could not auto-generate workflow: {e}[/yellow]")
                        console.print("Use 'interactive' mode for full conversation.")
            else:
                if non_interactive:
                    console.print("Error: Planner needs more information", file=sys.stderr)
                    sys.exit(1)
                else:
                    console.print("[yellow]The planner needs more information to create a workflow.[/yellow]")
                    if response.question:
                        console.print(f"Question: {response.question.question}")
                    console.print("Use interactive mode for full conversation.")

        except Exception as e:
            console.print(f"[red]Error in quick planning: {e}[/red]")
            logger.exception("Quick planning error")

    asyncio.run(_quick_plan())


if __name__ == "__main__":
    init_observability()
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    app()
