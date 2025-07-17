"""
AKD Mapping Module

This module provides data transformation and mapping capabilities for the AKD framework.
It implements a waterfall mapping approach that progresses through increasingly 
sophisticated transformation strategies to handle data compatibility between agents.

The module includes:
- WaterfallMapper: Main orchestrator for multi-stage mapping
- MapperInput/Output: Input and output schemas for mapping operations
- Individual mapping strategies: Direct, Semantic, and LLM-based mapping
"""

from .mappers import (
    WaterfallMapper,
    MapperInput,
    MapperOutput,
    MapperConfig,
    BaseMappingStrategy,
    DirectFieldMapper,
    SemanticFieldMapper,
    LLMFallbackMapper,
)

__all__ = [
    "WaterfallMapper",
    "MapperInput", 
    "MapperOutput",
    "MapperConfig",
    "BaseMappingStrategy",
    "DirectFieldMapper",
    "SemanticFieldMapper", 
    "LLMFallbackMapper",
]