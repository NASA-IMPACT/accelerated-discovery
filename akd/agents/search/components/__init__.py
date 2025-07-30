"""
Embedded Components for Literature Search Agents

This module contains internal components that are embedded within literature search agents,
providing deep integration of research workflow capabilities.
"""

from .clarification import ClarificationComponent
from .instruction_builder import InstructionBuilderComponent
from .research_synthesis import ResearchSynthesisComponent
from .triage import TriageComponent

__all__ = [
    "TriageComponent",
    "ClarificationComponent",
    "InstructionBuilderComponent",
    "ResearchSynthesisComponent",
]
