"""
Mapping System for AKD Framework

This module implements a data transformation system that uses a
waterfall approach to map data between different agent schemas. The system
progressively tries more sophisticated mapping strategies until successful
transformation is achieved. It is designed to be used in multi-agent workflows.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from pydantic import BaseModel, Field

from akd._base import AbstractBase, BaseConfig, InputSchema, OutputSchema
from akd.configs.project import get_project_settings
from akd.serializers import AKDSerializer


class MappingConfig(BaseConfig):
    """
    Configuration for data mapping operations.

    This configuration controls the behavior of the waterfall mapping system,
    including which strategies to enable, thresholds for matching, and fallback
    options for when primary strategies fail.
    """

    # Stage enablement flags
    enable_direct_matching: bool = Field(
        True, description="Enable exact field name matching as first strategy"
    )
    enable_semantic_matching: bool = Field(
        True, description="Enable fuzzy semantic field matching"
    )
    enable_llm_fallback: bool = Field(
        True, description="Enable LLM-assisted parsing as final fallback"
    )

    # Matching thresholds and quality controls
    semantic_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence score for semantic matches"
    )

    # LLM configuration
    llm_model: str = Field(
        default_factory=lambda: get_project_settings().model_config_settings.model_name,
        description="LLM model to use for fallback parsing",
    )

    # Performance and reliability settings
    max_retries: int = Field(
        2, ge=0, le=10, description="Maximum retry attempts for failed operations"
    )
    circuit_breaker_threshold: int = Field(
        5, ge=1, description="Number of failures before disabling a strategy"
    )
    enable_caching: bool = Field(
        True, description="Enable result caching for performance"
    )


class MappingInput(InputSchema):
    """
    Type-safe input specification for data mapping operations.

    Uses proper Pydantic models for both source and target, enabling type-safe
    transformations with full schema introspection and validation.
    """

    source_model: BaseModel = Field(
        description="Source Pydantic model instance to be transformed"
    )
    target_schema: Type[BaseModel] = Field(
        description="Target schema class for type-aware mapping"
    )
    mapping_hints: Optional[Dict[str, str]] = Field(
        None, description="Optional field mapping hints (source_field -> target_field)"
    )


class MappingOutput(OutputSchema):
    """
    Result of data mapping operation with comprehensive transformation details.

    Provides the transformed data as a validated Pydantic model instance along
    with metadata about the mapping process, including confidence scores,
    strategy used, and any unmapped fields.
    """

    mapped_model: BaseModel = Field(
        description="Successfully transformed and validated target model instance"
    )
    mapping_confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in the mapping quality"
    )
    used_strategy: str = Field(
        description="Name of the mapping strategy that succeeded"
    )
    unmapped_fields: List[str] = Field(
        default_factory=list,
        description="List of source fields that could not be mapped",
    )
    transformation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the transformation process",
    )


class BaseMappingStrategy(AbstractBase[MappingInput, MappingOutput]):
    """
    Abstract base class for all mapping strategies.

    Defines the interface that all mapping strategies must implement and provides
    common functionality like validation and result formatting with type-safe
    Pydantic model handling.
    """

    input_schema = MappingInput
    output_schema = MappingOutput
    config_schema = MappingConfig

    def __init__(self, config: Optional[MappingConfig] = None):
        super().__init__(config=config or MappingConfig())
        self.failure_count = 0
        self.is_disabled = False
        self.serializer = AKDSerializer()

    @abstractmethod
    async def map_models(
        self,
        source_model: BaseModel,
        target_schema: Type[BaseModel],
        mapping_hints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Map source model to target format.

        Args:
            source_model: Source Pydantic model instance
            target_schema: Target schema class for type-aware mapping
            mapping_hints: Optional field mapping hints

        Returns:
            Dictionary with 'mapped', 'confidence', and 'unmapped' keys
        """
        pass

    async def _arun(self, params: MappingInput, **kwargs) -> MappingOutput:
        """Execute the mapping strategy with proper error handling."""
        _ = kwargs  # Unused but required by interface

        # Get source field names for unmapped tracking
        source_fields = list(params.source_model.__class__.model_fields.keys())

        if self.is_disabled:
            # Create empty target model with default values
            try:
                empty_target = params.target_schema()
                return MappingOutput(
                    mapped_model=empty_target,
                    mapping_confidence=0.0,
                    used_strategy=f"{self.__class__.__name__} (disabled)",
                    unmapped_fields=source_fields,
                    transformation_metadata={"disabled": True},
                )
            except Exception:
                # If target schema requires fields, create minimal instance
                target_fields = self._get_schema_fields(params.target_schema)
                minimal_data = {field: None for field in target_fields}
                try:
                    empty_target = params.target_schema(**minimal_data)
                except Exception:
                    # Last resort: use dict representation
                    empty_target = params.target_schema.model_construct()

                return MappingOutput(
                    mapped_model=empty_target,
                    mapping_confidence=0.0,
                    used_strategy=f"{self.__class__.__name__} (disabled)",
                    unmapped_fields=source_fields,
                    transformation_metadata={"disabled": True},
                )

        try:
            result = await self.map_models(
                params.source_model, params.target_schema, params.mapping_hints
            )

            # Create target model instance from mapped data
            try:
                mapped_model = params.target_schema(**result["mapped"])
            except Exception as e:
                logger.warning(f"Failed to create target model instance: {e}")
                # Try with model_construct for partial data
                mapped_model = params.target_schema.model_construct(**result["mapped"])

            return MappingOutput(
                mapped_model=mapped_model,
                mapping_confidence=result["confidence"],
                used_strategy=self.__class__.__name__,
                unmapped_fields=result.get("unmapped", []),
                transformation_metadata=result.get("metadata", {}),
            )

        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.circuit_breaker_threshold:
                self.is_disabled = True
                logger.warning(
                    f"Disabling {self.__class__.__name__} after {self.failure_count} failures"
                )

            logger.error(f"Mapping strategy {self.__class__.__name__} failed: {e}")
            raise

    def _get_schema_fields(self, schema: Type[BaseModel]) -> List[str]:
        """Extract field names from Pydantic schema."""
        if hasattr(schema, "model_fields"):
            return list(schema.model_fields.keys())
        return []

    def _convert_model_to_dict(self, model: BaseModel) -> Dict[str, Any]:
        """Convert Pydantic model to dictionary using AKDSerializer."""
        return self.serializer._convert_pydantic_to_dict(model)
