"""
Mapping System for AKD Framework

This module implements a data transformation system that uses a
waterfall approach to map data between different agent schemas. The system
progressively tries more sophisticated mapping strategies until successful
transformation is achieved. It is designed to be used in multi-agent workflows.
"""

import hashlib
import json
from abc import abstractmethod
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Type

from langchain_openai import ChatOpenAI
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


class DirectFieldMapper(BaseMappingStrategy):
    """
    Maps data using exact field name matching.

    This strategy performs direct field-to-field mapping based on exact name
    correspondence. It's the fastest and most reliable strategy when field
    names align between source and target schemas.
    """

    async def map_models(
        self,
        source_model: BaseModel,
        target_schema: Type[BaseModel],
        mapping_hints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Map fields using exact name matching between Pydantic models.

        Args:
            source_model: Source Pydantic model instance
            target_schema: Target schema class for field validation
            mapping_hints: Optional explicit field mappings

        Returns:
            Mapping result with mapped fields and confidence score
        """

        # Convert source model to dictionary
        source_data = self._convert_model_to_dict(source_model)

        mapped = {}
        used_fields = set()

        # Apply explicit mapping hints first
        if mapping_hints:
            for source_field, target_field in mapping_hints.items():
                if source_field in source_data:
                    mapped[target_field] = source_data[source_field]
                    used_fields.add(source_field)

        # Direct field matching
        target_fields = self._get_schema_fields(target_schema)

        for target_field in target_fields:
            if target_field in source_data and target_field not in mapped:
                mapped[target_field] = source_data[target_field]
                used_fields.add(target_field)

        # Calculate confidence based on mapping success
        total_fields = len(source_data)
        mapped_fields = len(used_fields)
        confidence = mapped_fields / max(total_fields, 1)

        unmapped_fields = [f for f in source_data.keys() if f not in used_fields]

        return {
            "mapped": mapped,
            "confidence": confidence,
            "unmapped": unmapped_fields,
            "metadata": {
                "strategy": "direct_matching",
                "total_source_fields": total_fields,
                "mapped_fields": mapped_fields,
                "source_schema": source_model.__class__.__name__,
                "target_schema": target_schema.__name__,
            },
        }


class SemanticFieldMapper(BaseMappingStrategy):
    """
    Maps data using fuzzy semantic matching.

    This strategy uses string similarity and semantic knowledge to match fields
    with similar names or meanings. It handles variations in naming conventions
    and common field name patterns.
    """

    def __init__(self, config: Optional[MappingConfig] = None):
        super().__init__(config)

        # Basic semantic field groups for common patterns
        self.semantic_groups = {
            "identifiers": ["id", "identifier", "uid", "key", "_id", "pk"],
            "text_content": ["text", "content", "body", "description", "summary"],
            "titles": ["title", "name", "heading", "label"],
            "queries": ["query", "search", "question", "term"],
            "results": ["results", "data", "items", "documents", "papers"],
            "scores": ["score", "confidence", "relevance", "rating"],
            "counts": ["count", "total", "length", "size", "num_results"],
            "urls": ["url", "link", "href", "uri", "source"],
        }

        # Create reverse mapping for quick lookup
        self.field_to_group = {}
        for group, fields in self.semantic_groups.items():
            for field in fields:
                self.field_to_group[field.lower()] = group

    async def map_models(
        self,
        source_model: BaseModel,
        target_schema: Type[BaseModel],
        mapping_hints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Map fields using semantic similarity between Pydantic models.

        Args:
            source_model: Source Pydantic model instance
            target_schema: Target schema class for field validation
            mapping_hints: Optional explicit field mappings (applied first)

        Returns:
            Mapping result with semantically matched fields
        """

        # Convert source model to dictionary
        source_data = self._convert_model_to_dict(source_model)

        mapped = {}
        used_fields = set()

        # Apply mapping hints first (higher confidence)
        if mapping_hints:
            for source_field, target_field in mapping_hints.items():
                if source_field in source_data:
                    mapped[target_field] = source_data[source_field]
                    used_fields.add(source_field)

        # Semantic matching for remaining fields
        target_fields = self._get_schema_fields(target_schema)
        remaining_source_fields = [
            f for f in source_data.keys() if f not in used_fields
        ]

        for target_field in target_fields:
            if target_field not in mapped:
                best_match = await self._find_best_semantic_match(
                    target_field, remaining_source_fields, source_data
                )

                if best_match and best_match["confidence"] >= self.semantic_threshold:
                    source_field = best_match["source_field"]
                    mapped[target_field] = source_data[source_field]
                    used_fields.add(source_field)
                    remaining_source_fields.remove(source_field)

        # Calculate overall confidence
        total_fields = len(source_data)
        mapped_fields = len(used_fields)
        confidence = mapped_fields / max(total_fields, 1)

        unmapped_fields = [f for f in source_data.keys() if f not in used_fields]

        return {
            "mapped": mapped,
            "confidence": confidence,
            "unmapped": unmapped_fields,
            "metadata": {
                "strategy": "semantic_matching",
                "total_source_fields": total_fields,
                "mapped_fields": mapped_fields,
                "semantic_threshold": self.semantic_threshold,
                "source_schema": source_model.__class__.__name__,
                "target_schema": target_schema.__name__,
            },
        }

    async def _find_best_semantic_match(
        self, target_field: str, source_fields: List[str], source_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find the best semantic match for a target field."""

        if not source_fields:
            return None

        best_match = None
        best_confidence = 0.0

        for source_field in source_fields:
            confidence = self._calculate_semantic_similarity(target_field, source_field)

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = {
                    "source_field": source_field,
                    "confidence": confidence,
                    "value": source_data[source_field],
                }

        return best_match if best_confidence > 0.0 else None

    def _calculate_semantic_similarity(
        self, target_field: str, source_field: str
    ) -> float:
        """Calculate semantic similarity between two field names."""

        target_lower = target_field.lower()
        source_lower = source_field.lower()

        # Exact match (highest confidence)
        if target_lower == source_lower:
            return 1.0

        # String similarity using difflib
        string_similarity = SequenceMatcher(None, target_lower, source_lower).ratio()

        # Semantic group matching
        target_group = self.field_to_group.get(target_lower)
        source_group = self.field_to_group.get(source_lower)

        semantic_bonus = 0.0
        if target_group and source_group and target_group == source_group:
            semantic_bonus = 0.3

        # Substring matching
        substring_bonus = 0.0
        if target_lower in source_lower or source_lower in target_lower:
            substring_bonus = 0.2

        # Combined score with weights
        total_score = (string_similarity * 0.5) + semantic_bonus + substring_bonus

        return min(total_score, 1.0)


class LLMFallbackMapper(BaseMappingStrategy):
    """
    Maps data using LLM when other strategies fail.

    This strategy uses a language model to intelligently parse and transform
    complex data structures when direct and semantic matching are insufficient.
    It serves as the final fallback in the waterfall approach.
    """

    def __init__(self, config: Optional[MappingConfig] = None):
        super().__init__(config)

        # Initialize LLM client with graceful fallback
        project_settings = get_project_settings()
        api_key = project_settings.model_config_settings.api_keys.openai

        if api_key:
            self.llm = ChatOpenAI(
                model=self.llm_model, temperature=0.0, api_key=api_key
            )
        else:
            # Mock LLM for testing without API key
            self.llm = None
            logger.warning("No OpenAI API key found, LLM fallback will be disabled")

    async def map_models(
        self,
        source_model: BaseModel,
        target_schema: Type[BaseModel],
        mapping_hints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Map data using LLM-assisted parsing between Pydantic models.

        Args:
            source_model: Source Pydantic model instance
            target_schema: Target schema class for guided transformation
            mapping_hints: Optional explicit field mappings

        Returns:
            Mapping result from LLM transformation
        """

        # Convert source model to dictionary
        source_data = self._convert_model_to_dict(source_model)

        try:
            # Check if LLM is available
            if not self.llm:
                raise Exception("LLM not available (no API key)")

            # Create transformation prompt
            prompt = self._create_transformation_prompt(
                source_data, target_schema, mapping_hints
            )

            # Get LLM response
            response = await self.llm.ainvoke(prompt)

            # Parse the response
            parsed_data = self._parse_llm_response(response.content)

            # Validate against target schema if possible
            confidence = self._estimate_transformation_confidence(
                parsed_data, target_schema
            )

            return {
                "mapped": parsed_data,
                "confidence": confidence,
                "unmapped": [],
                "metadata": {
                    "strategy": "llm_assisted",
                    "model": self.llm_model,
                    "prompt_length": len(prompt),
                    "source_schema": source_model.__class__.__name__,
                    "target_schema": target_schema.__name__,
                },
            }

        except Exception as e:
            logger.warning(f"LLM fallback failed: {e}")

            # Final fallback - try to create minimal target model
            target_fields = self._get_schema_fields(target_schema)

            # Use source data where field names match, None for others
            fallback_data = {}
            for field in target_fields:
                if field in source_data:
                    fallback_data[field] = source_data[field]
                else:
                    # Try to find a reasonable default based on field type
                    field_info = target_schema.model_fields.get(field)
                    if field_info and hasattr(field_info, "default"):
                        fallback_data[field] = field_info.default
                    else:
                        fallback_data[field] = None

            return {
                "mapped": fallback_data,
                "confidence": 0.1,
                "unmapped": [f for f in source_data.keys() if f not in target_fields],
                "metadata": {
                    "strategy": "string_fallback",
                    "error": str(e),
                    "source_schema": source_model.__class__.__name__,
                    "target_schema": target_schema.__name__,
                },
            }

    def _create_transformation_prompt(
        self,
        source_data: Dict[str, Any],
        target_schema: Type[BaseModel],
        mapping_hints: Optional[Dict[str, str]],
    ) -> str:
        """Create prompt for LLM data transformation."""

        # Get target schema information
        target_fields = self._get_schema_fields(target_schema)
        schema_info = {
            "name": target_schema.__name__,
            "fields": target_fields,
            "description": getattr(target_schema, "__doc__", ""),
        }

        hints_text = ""
        if mapping_hints:
            hints_text = f"\nMapping hints: {json.dumps(mapping_hints, indent=2)}"

        prompt = f"""
Transform the following source data into the target schema format.

SOURCE DATA:
{json.dumps(source_data, indent=2)}

TARGET SCHEMA: {schema_info["name"]}
Description: {schema_info["description"]}
Required fields: {", ".join(target_fields)}
{hints_text}

INSTRUCTIONS:
1. Extract relevant information from the source data
2. Map it to the target schema fields as best as possible
3. Use intelligent inference for missing but derivable fields
4. Return ONLY valid JSON matching the target schema
5. If a field cannot be determined, omit it or use null
6. Be conservative but creative in your mappings

Return only the JSON object, no additional text:
"""

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with fallback handling."""

        response = response.strip()

        # Remove markdown formatting if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            start = response.find("{")
            end = response.rfind("}") + 1

            if start != -1 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

            # Final fallback
            return {"parsed_content": response}

    def _estimate_transformation_confidence(
        self, parsed_data: Dict[str, Any], target_schema: Type[BaseModel]
    ) -> float:
        """Estimate confidence in the LLM transformation."""

        if not target_schema:
            return 0.5

        target_fields = set(self._get_schema_fields(target_schema))
        parsed_fields = set(parsed_data.keys())

        if not target_fields:
            return 0.7  # Default confidence if no schema info

        # Calculate field coverage
        matching_fields = target_fields.intersection(parsed_fields)
        coverage = len(matching_fields) / len(target_fields)

        # Base confidence for LLM transformation
        base_confidence = 0.6

        # Adjust based on field coverage
        confidence = base_confidence + (coverage * 0.3)

        return min(confidence, 0.9)  # Cap at 0.9 for LLM transformations


class WaterfallMapper(AbstractBase[MappingInput, MappingOutput]):
    """
    Orchestrates multiple mapping strategies in waterfall pattern.

    This is the main entry point for type-safe data transformation. It progressively
    tries different mapping strategies until successful transformation is achieved,
    providing robust data compatibility between different agent schemas.
    """

    input_schema = MappingInput
    output_schema = MappingOutput
    config_schema = MappingConfig

    def __init__(self, config: Optional[MappingConfig] = None):
        super().__init__(config=config or MappingConfig())

        # Initialize mapping strategies
        self.direct_mapper = DirectFieldMapper(self.config)
        self.semantic_mapper = SemanticFieldMapper(self.config)
        self.llm_mapper = LLMFallbackMapper(self.config)

        # Cache for performance
        self._cache: Dict[str, MappingOutput] = {}
        self.serializer = AKDSerializer()

    async def _arun(self, params: MappingInput, **kwargs) -> MappingOutput:
        """
        Execute waterfall mapping with progressive strategy application.

        Args:
            params: Mapping input with source model and target schema

        Returns:
            Mapping result from the first successful strategy
        """
        _ = kwargs  # Unused but required by interface

        # Check cache if enabled
        cache_key = None
        if self.enable_caching:
            cache_key = self._generate_cache_key(params)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                # Update metadata to indicate cache hit
                cached_result.transformation_metadata["from_cache"] = True
                return cached_result

        # Progressive strategy application
        strategies = [
            ("direct", self.direct_mapper, self.enable_direct_matching),
            ("semantic", self.semantic_mapper, self.enable_semantic_matching),
            ("llm_fallback", self.llm_mapper, self.enable_llm_fallback),
        ]

        last_result = None

        for strategy_name, mapper, enabled in strategies:
            if not enabled or mapper.is_disabled:
                continue

            try:
                if self.debug:
                    logger.debug(f"Trying mapping strategy: {strategy_name}")

                result = await mapper.arun(params)

                # Check if result is acceptable
                if self._is_acceptable_result(result):
                    if self.debug:
                        logger.debug(
                            f"Successful mapping with {strategy_name}: confidence {result.mapping_confidence}"
                        )

                    # Cache successful result
                    if self.enable_caching and cache_key:
                        self._cache[cache_key] = result

                    return result

                last_result = result

            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue

        # If all strategies failed, return the last attempt
        if last_result:
            return last_result

        # Complete fallback - create minimal target model
        try:
            # Try to create empty target model
            empty_target = params.target_schema()
            source_fields = list(params.source_model.__class__.model_fields.keys())

            return MappingOutput(
                mapped_model=empty_target,
                mapping_confidence=0.0,
                used_strategy="none",
                unmapped_fields=source_fields,
                transformation_metadata={"all_strategies_failed": True},
            )
        except Exception:
            # If we can't create empty model, use model_construct
            empty_target = params.target_schema.model_construct()
            source_fields = list(params.source_model.__class__.model_fields.keys())

            return MappingOutput(
                mapped_model=empty_target,
                mapping_confidence=0.0,
                used_strategy="none",
                unmapped_fields=source_fields,
                transformation_metadata={
                    "all_strategies_failed": True,
                    "fallback_construct": True,
                },
            )

    def _is_acceptable_result(self, result: MappingOutput) -> bool:
        """Check if a mapping result is acceptable to stop the waterfall."""

        # Accept if we have a valid mapped model and reasonable confidence
        return result.mapped_model is not None and result.mapping_confidence > 0.1

    def _generate_cache_key(self, params: MappingInput) -> str:
        """Generate cache key for mapping parameters."""

        # Convert source model to dict for key generation
        source_dict = self.serializer._convert_pydantic_to_dict(params.source_model)

        key_data = {
            "source_schema": params.source_model.__class__.__name__,
            "source_keys": sorted(source_dict.keys()),
            "target_schema": params.target_schema.__name__,
            "mapping_hints": params.mapping_hints or {},
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
