"""
Production-Grade Components for Literature Search Agents

This module contains production-ready components for research workflow orchestration,
content management, and synthesis operations with comprehensive configuration support.
"""

from .triage import TriageComponent
from .clarification import ClarificationComponent  
from .instruction_builder import InstructionBuilderComponent
from .content_manager import ContentManager
from .synthesis import ResearchSynthesizer

# Legacy components (deprecated - use new components above)
from .research_synthesis import ResearchSynthesisComponent

__all__ = [
    # Core production components
    "TriageComponent",
    "ClarificationComponent",
    "InstructionBuilderComponent", 
    "ContentManager",
    "ResearchSynthesizer",
    
    # Legacy components (deprecated)
    "ResearchSynthesisComponent",
]