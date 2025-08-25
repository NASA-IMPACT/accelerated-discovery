"""
Prompt loading utilities for data search components.
"""

import os
from typing import Dict, Any
from pathlib import Path


def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template from the prompts directory.
    
    Args:
        template_name: Name of the template file (without .md extension)
        
    Returns:
        Template content as string
        
    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    prompts_dir = Path(__file__).parent / "prompts"
    template_path = prompts_dir / f"{template_name}.md"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    return template_path.read_text().strip()


def format_prompt_template(template: str, **kwargs: Any) -> str:
    """
    Format a prompt template with the given variables.
    
    Args:
        template: Template string with {variable} placeholders
        **kwargs: Variables to substitute in the template
        
    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs)


def load_and_format_prompt(template_name: str, **kwargs: Any) -> str:
    """
    Load and format a prompt template in one step.
    
    Args:
        template_name: Name of the template file (without .md extension)
        **kwargs: Variables to substitute in the template
        
    Returns:
        Formatted prompt string
    """
    template = load_prompt_template(template_name)
    return format_prompt_template(template, **kwargs)