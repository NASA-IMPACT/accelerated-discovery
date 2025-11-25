"""
LLM Reranker Adapter for CMR Handler.

This adapter provides a clean interface for using the LLM reranker tool with CMR collections,
handling all conversions between CMR collection format and SearchResultItem format.
"""

from typing import Any, Dict, List

from loguru import logger

from akd.structures import SearchResultItem
from akd.tools.reranker import (
    LLMRerankerTool,
    LLMRerankerToolConfig,
    ScoringCategory,
    ScoringCriterion,
    create_reranker,
)


class LLMRerankerAdapter:
    """
    Adapter for using LLM reranker with CMR collections.

    This class encapsulates:
    - Configuration management
    - Conversion between CMR collections and SearchResultItem format
    - LLM reranker invocation
    - Score extraction and logging
    """

    def __init__(
        self,
        config: LLMRerankerToolConfig,
        debug: bool = False,
    ):
        """
        Initialize the LLM reranker adapter.

        Args:
            config: LLM reranker configuration with scoring criteria and fields
            debug: Enable debug logging
        """
        self.config = config
        self.debug = debug

        # Create the reranker instance
        self.reranker: LLMRerankerTool = create_reranker(
            reranker_type="llm",
            config=config,
            debug=debug,
        )

        if debug:
            logger.info(
                f"Initialized LLMRerankerAdapter with model: {config.model_name}, "
                f"{len(config.scoring_criteria)} criteria",
            )

    @classmethod
    def from_default_cmr_config(
        cls,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        debug: bool = False,
    ) -> "LLMRerankerAdapter":
        """
        Create adapter with default CMR-specific scoring criteria.

        Args:
            model_name: LLM model to use for scoring
            temperature: Temperature for LLM responses
            debug: Enable debug logging

        Returns:
            LLMRerankerAdapter instance with default CMR configuration
        """
        config = cls._create_default_cmr_config(model_name, temperature)
        return cls(config=config, debug=debug)

    @staticmethod
    def _create_default_cmr_config(
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> LLMRerankerToolConfig:
        """
        Create default LLM reranker configuration with CMR-specific scoring criteria.

        Based on the criteria from run_llm_reranker.py example.

        Args:
            model_name: LLM model to use
            temperature: Temperature for LLM responses

        Returns:
            LLMRerankerToolConfig with CMR-specific criteria
        """
        return LLMRerankerToolConfig(
            model_name=model_name,
            temperature=temperature,
            fields_to_evaluate={
                "title": "The title or name of the dataset (Entry Title in CMR)",
                "content": "Description or abstract of the dataset",
                "spatial_resolution": "Ground sampling distance - lower values (e.g., 30m) are higher resolution",
                "temporal_resolution": "Revisit time or frequency of data collection (e.g., daily, monthly)",
                "processing_level": "Data processing level (L0/1A=raw, L1/1B=calibrated/geolocated, L2+=derived products with corrections)",
                "bounding_box": "Spatial extent of the dataset",
                "temporal": "Temporal coverage range of the dataset",
            },
            scoring_criteria=[
                ScoringCriterion(
                    name="Variable Accuracy",
                    description="This criterion assesses whether a dataset directly measures or derives the targeted variable (e.g., sea surface temperature, soil moisture, vegetation indices). The dataset should actually measure what is specified in the query.",
                    weight=0.25,
                    scoring_categories=[
                        ScoringCategory(
                            name="Direct Measurement",
                            description="The dataset directly measures the variable specified in the query",
                            value=3.0,
                        ),
                        ScoringCategory(
                            name="Indirectly Related",
                            description="The dataset measures a parameter that can be converted or correlated to the desired variable with additional processing",
                            value=2.0,
                        ),
                        ScoringCategory(
                            name="Unrelated",
                            description="The dataset measures unrelated phenomena",
                            value=1.0,
                        ),
                    ],
                ),
                ScoringCriterion(
                    name="Resolution",
                    description="Resolution considers both spatial and temporal resolution and how well they match the scale and dynamics of the scientific question. Higher spatial resolution (smaller pixel size) captures finer details. Temporal resolution (revisit time) determines how frequently data are collected; dynamic processes (e.g., wildfires, flooding) require daily or sub-daily observations. Unless otherwise specified, prefer higher spatial (30m-500m) and temporal (daily) resolutions.",
                    weight=0.20,
                    scoring_categories=[
                        ScoringCategory(
                            name="Appropriate Spatial & Temporal Resolution",
                            description="The dataset's spatial and temporal resolution align with the scale of the scientific question. Resolution is sufficient to capture features of interest without being overly fine",
                            value=3.0,
                        ),
                        ScoringCategory(
                            name="Acceptable but Not Optimal",
                            description="The dataset's resolution falls within a usable range. Spatial resolution may be too fine (data overload) or too coarse but still usable; temporal sampling may miss short-term events",
                            value=2.0,
                        ),
                        ScoringCategory(
                            name="Too Coarse or Too Infrequent",
                            description="The spatial or temporal resolution is insufficient to capture the phenomenon",
                            value=1.0,
                        ),
                    ],
                ),
                ScoringCriterion(
                    name="Processing Level",
                    description="Processing level reflects how much pre-processing has been applied to transform raw instrument data into geophysical variables. Higher levels (2-4) generally include calibration, georeferencing, atmospheric corrections and aggregation onto grids, making the data easier to use. Unless the user's scientific goal requires raw or unprocessed data, always prefer high processing levels (Level 2 or greater).",
                    weight=0.15,
                    scoring_categories=[
                        ScoringCategory(
                            name="High Processing Level",
                            description="Level 2 or above. Data is calibrated, geolocated and often atmospherically corrected or aggregated onto grids",
                            value=3.0,
                        ),
                        ScoringCategory(
                            name="Moderate Processing Level",
                            description="Level 1 or 1B. Data is calibrated but not atmospherically corrected; users must derive their own geophysical variables",
                            value=2.0,
                        ),
                        ScoringCategory(
                            name="Raw",
                            description="Level 0 or 1A. Raw data requiring extensive processing",
                            value=1.0,
                        ),
                    ],
                ),
                ScoringCriterion(
                    name="Cross-Cutting Potential",
                    description="Cross-cutting potential measures whether a dataset can be combined with other datasets to answer multi-facet scientific queries. Datasets that complement one another in spatial, temporal or spectral characteristics enable more comprehensive analysis. Prefer collections which are capable of addressing multiple research questions or topics.",
                    weight=0.10,
                    scoring_categories=[
                        ScoringCategory(
                            name="High Synergy",
                            description="The dataset complements other instruments in spatial or temporal coverage, enabling integrated analysis",
                            value=3.0,
                        ),
                        ScoringCategory(
                            name="Moderate Synergy",
                            description="The dataset can be combined with others but requires significant processing, such as reprojection or resampling",
                            value=2.0,
                        ),
                        ScoringCategory(
                            name="Limited Synergy",
                            description="The dataset is unique or incompatible with others and offers little benefit when combined",
                            value=1.0,
                        ),
                    ],
                ),
                ScoringCriterion(
                    name="Ease of Use",
                    description="Ease of use evaluates how simple it is to discover, access and work with a dataset. Factors include direct download availability, intuitive user interfaces, comprehensive documentation, and high-quality metadata. Prefer collections where the raw data is easy to access, and which have available supplementary files and good documentation.",
                    weight=0.10,
                    scoring_categories=[
                        ScoringCategory(
                            name="Easy Access and Well-Documented",
                            description="Data can be downloaded directly through standard protocols and is accompanied by user guides, quality information and comprehensive metadata",
                            value=3.0,
                        ),
                        ScoringCategory(
                            name="Moderate Usability",
                            description="Data are available but require authentication, subsetting tools or specialized software to access",
                            value=2.0,
                        ),
                        ScoringCategory(
                            name="Difficult Access/Poor Documentation",
                            description="Data require special requests, proprietary software or have incomplete/broken metadata or links, making them effectively inaccessible",
                            value=1.0,
                        ),
                    ],
                ),
            ],
        )

    def convert_collections_to_search_results(
        self,
        collections: List[Dict[str, Any]],
        query: str,
    ) -> List[SearchResultItem]:
        """
        Convert CMR collections to SearchResultItem format for reranker.

        Args:
            collections: List of CMR collection dictionaries
            query: The search query

        Returns:
            List of SearchResultItem objects
        """
        results = []
        for coll in collections:
            # Extract fields from collection
            title = coll.get("entry_title", "")
            summary = coll.get("summary", "")
            concept_id = coll.get("concept_id", "")

            # Prepare extra fields for reranker evaluation
            extra = {"concept_id": concept_id}

            # Add fields from query_approach_info if available
            query_approach_info = coll.get("query_approach_info", {})
            if query_approach_info:
                for field in ["spatial_resolution", "temporal_resolution", "processing_level", "bounding_box", "temporal"]:
                    if field in query_approach_info and query_approach_info[field]:
                        extra[field] = query_approach_info[field]

            # Also check collection itself for these fields (fallback)
            if not extra.get("processing_level"):
                processing_level = coll.get("processing_level_id", "") or coll.get("processing_level", "")
                if processing_level:
                    extra["processing_level"] = processing_level

            for field in ["spatial_resolution", "temporal_resolution", "bounding_box", "temporal"]:
                if field not in extra and coll.get(field):
                    extra[field] = coll[field]

            # Add any additional fields from the collection
            for key in ["approach_index", "query_index", "instrument", "platform"]:
                if key in coll:
                    extra[key] = coll[key]

            # CMR concept URL format
            url = f"https://cmr.earthdata.nasa.gov/search/concepts/{concept_id}.html"

            result = SearchResultItem(
                query=query,
                title=title,
                content=summary,
                url=url,
                extra=extra,
            )
            results.append(result)

        return results

    def convert_search_results_to_collections(
        self,
        results: List[SearchResultItem],
        original_collections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert SearchResultItem objects back to CMR collection format.

        Args:
            results: List of reranked SearchResultItem objects
            original_collections: Original collection dictionaries

        Returns:
            List of CMR collection dictionaries in reranked order
        """
        # Build mapping from concept_id to original collection
        concept_id_map = {coll.get("concept_id"): coll for coll in original_collections}

        # Convert results back to collections
        ranked_collections = []
        for result in results:
            concept_id = result.extra.get("concept_id")
            if concept_id and concept_id in concept_id_map:
                coll = concept_id_map[concept_id]
                # Preserve LLM reranker scores in the collection
                if "llm_reranker" in result.extra:
                    coll["llm_reranker"] = result.extra["llm_reranker"]
                if hasattr(result, "score"):
                    coll["reranker_score"] = result.score
                ranked_collections.append(coll)

        return ranked_collections

    async def rank_collections(
        self,
        collections: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Rank CMR collections using LLM reranker.

        This is the main method that orchestrates the full workflow:
        1. Convert collections to SearchResultItem format
        2. Run LLM reranker
        3. Convert back to collection format

        Args:
            collections: List of CMR collection dictionaries
            query: The search query

        Returns:
            List of collections in ranked order with reranker scores
        """
        if not collections:
            if self.debug:
                logger.warning("No collections provided for ranking")
            return []

        # Step 1: Convert to SearchResultItem format
        search_results = self.convert_collections_to_search_results(collections, query)

        if self.debug:
            logger.info(
                f"Converted {len(collections)} collections to SearchResultItem format",
            )

        # Step 2: Run reranker
        reranker_input = self.reranker.input_schema(
            query=query,
            results=search_results,
        )
        reranked_output = await self.reranker.arun(reranker_input)

        if self.debug:
            logger.info(
                f"Reranked {len(reranked_output.results)} results using {self.config.model_name}",
            )

        # Step 3: Convert back to collection format
        ranked_collections = self.convert_search_results_to_collections(
            reranked_output.results,
            collections,
        )

        if self.debug:
            logger.info(
                f"Converted {len(ranked_collections)} results back to collection format",
            )

        return ranked_collections

    def get_reranker_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the reranker configuration.

        Returns:
            Dictionary with reranker metadata
        """
        return {
            "reranker_type": "llm",
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "num_criteria": len(self.config.scoring_criteria),
            "criteria_names": [c.name for c in self.config.scoring_criteria],
            "criteria_weights": {c.name: c.weight for c in self.config.scoring_criteria},
        }
