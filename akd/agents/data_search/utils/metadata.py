"""Utilities for capturing execution metadata."""

import json
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def capture_git_info() -> Dict[str, Any]:
    """
    Capture current git commit, branch, and dirty status.

    Returns:
        Dictionary with git metadata
    """
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        is_dirty = len(dirty_output.strip()) > 0

        return {
            "commit_hash": commit_hash,
            "branch": branch,
            "is_dirty": is_dirty,
        }
    except Exception:
        return {
            "commit_hash": "unknown",
            "branch": "unknown",
            "is_dirty": None,
        }


def capture_prompts(*prompts_dirs: Path) -> Dict[str, str]:
    """
    Capture raw prompt template files from one or more directories.

    Args:
        *prompts_dirs: Directories containing prompt .md files

    Returns:
        Dictionary mapping template name to raw markdown content
    """
    prompts = {}

    for prompts_dir in prompts_dirs:
        if not prompts_dir.exists():
            continue

        for file in prompts_dir.glob("*.md"):
            # Use relative path as key to avoid conflicts
            key = f"{prompts_dir.name}/{file.name}"
            prompts[key] = file.read_text()

    return prompts


def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert text to filename-safe slug.

    Args:
        text: Text to slugify
        max_length: Maximum length of slug

    Returns:
        Filename-safe slug
    """
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:max_length]


def generate_run_id() -> str:
    """
    Generate a unique run ID for a search execution.

    Returns:
        Unique 8-character hex ID (e.g., "a1b2c3d4")
    """
    return uuid.uuid4().hex[:8]


def generate_search_id(query: str, run_id: str) -> str:
    """
    Generate a unique search ID from run ID and query.

    Args:
        query: Natural language query
        run_id: Unique run ID for this search

    Returns:
        Unique search ID (e.g., "a1b2c3d4_atmospheric_co2")
    """
    query_slug = slugify(query)
    return f"{run_id}_{query_slug}"


def build_output_filename(search_id: str) -> str:
    """
    Build output filename for auto-saved results.

    Args:
        search_id: Unique search ID

    Returns:
        Path to output file in captured_data/ directory
    """
    return f"captured_data/{search_id}.json"


def save_with_metadata(
    agent_output,  # DataSearchAgentOutputSchema
    agent_config,  # DataSearchAgentConfig
    output_file: str,
):
    """
    Save agent output with execution metadata.

    Args:
        agent_output: Agent output schema
        agent_config: Agent configuration
        output_file: Path to save JSON file
    """
    # Build complete output with metadata
    output_data = {
        "agent_output": agent_output.model_dump(),
        "execution_metadata": {
            "timestamp": datetime.now().isoformat(),
            "config": agent_config.model_dump(),
        },
    }

    # Add git info and prompts if metadata capture enabled
    if agent_config.capture_metadata:
        output_data["execution_metadata"]["git_info"] = capture_git_info()

        # Capture prompts from universal + CMR handler directories
        from pathlib import Path

        # Get component prompts directory
        base_dir = Path(__file__).parent.parent
        component_prompts_dir = base_dir / "components" / "prompts"
        cmr_prompts_dir = base_dir / "handlers" / "cmr" / "prompts"

        output_data["prompts"] = capture_prompts(
            component_prompts_dir,
            cmr_prompts_dir,
        )

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save to file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
