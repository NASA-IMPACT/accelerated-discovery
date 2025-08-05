"""
Source validation tool for validating research sources against a controlled whitelist.

This tool processes raw links from search engines (Semantic Scholar, SearxNG),
extracts DOIs, fetches source metadata from CrossRef API, and validates
against a predefined source whitelist.
"""

from __future__ import annotations
from rapidfuzz.fuzz import token_set_ratio

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig
from akd.utils import get_akd_root

if TYPE_CHECKING:
    pass


class SourceInfo(BaseModel):
    """Schema for source information from CrossRef API."""

    title: str = Field(..., description="Source title")
    publisher: Optional[str] = Field(None, description="Publisher name")
    issn: Optional[List[str]] = Field(None, description="List of ISSNs")
    is_open_access: Optional[bool] = Field(None, description="Open access status")
    doi: str = Field(..., description="DOI of the article")
    url: Optional[str] = Field(None, description="Original URL")


class ValidationResult(BaseModel):
    """Schema for validation result."""

    source_info: Optional[SourceInfo] = Field(
        None,
        description="Source information from CrossRef",
    )
    is_whitelisted: bool = Field(..., description="Whether source is in whitelist")
    whitelist_category: Optional[str] = Field(
        None,
        description="Category from whitelist (e.g., ES, Bio, etc.)",
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors",
    )
    confidence_score: float = Field(
        ...,
        description="Confidence in validation (0.0-1.0)",
    )


class SourceValidatorInputSchema(InputSchema):
    """Input schema for source validation tool."""

    search_results: List[SearchResultItem] = Field(
        ...,
        description="List of search results to validate",
    )
    whitelist_file_path: Optional[str] = Field(
        None,
        description="Path to source whitelist JSON file. If None, uses default path.",
    )


class SourceValidatorOutputSchema(OutputSchema):
    """Output schema for source validation tool."""

    validated_results: List[ValidationResult] = Field(
        ...,
        description="List of validation results",
    )
    summary: Dict[str, Any] = Field(..., description="Summary statistics of validation")


class SourceValidatorConfig(BaseToolConfig):
    """Configuration for source validator tool."""

    crossref_base_url: str = Field(
        default="https://api.crossref.org/works",
        description="Base URL for CrossRef API",
    )
    whitelist_file_path: Optional[str] = Field(
        default_factory=lambda: str(
            get_akd_root() / "docs" / "pubs_whitelist.json",
        ),
        description="Path to source whitelist JSON file",
    )
    timeout_seconds: int = Field(
        default=30,
        description="Timeout for API requests in seconds",
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum number of concurrent API requests",
    )
    user_agent: str = Field(
        default="SourceValidator/1.0",
        description="User agent for API requests",
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    use_fuzzy_match: bool = Field(
        default=False,
        description="Enable fuzzy matching of journal titles",
    )

    fuzzy_threshold: int = Field(
        default=87,
        description="Fuzzy matching threshold (0-100) for journal titles",
    )


class SourceValidator(
    BaseTool[SourceValidatorInputSchema, SourceValidatorOutputSchema],
):
    """
    Tool for validating research sources against a source whitelist.

    This tool:
    1. Extracts DOIs from URLs in search results
    2. Fetches source metadata from CrossRef API
    3. Validates against a predefined source whitelist
    4. Returns validation results with confidence scores
    """

    input_schema = SourceValidatorInputSchema
    output_schema = SourceValidatorOutputSchema
    config_schema = SourceValidatorConfig

    def __init__(
        self,
        config: Optional[SourceValidatorConfig] = None,
        debug: bool = False,
    ):
        """Initialize the source validator tool."""
        config = config or SourceValidatorConfig()
        config.debug = debug
        super().__init__(config, debug)

        # Load whitelist on initialization
        self._whitelist = self._load_whitelist()

        # DOI extraction patterns
        self._doi_patterns = [
            # Standard DOI URLs
            r"(?:https?://)?(?:dx\.)?doi\.org/(?:10\.\d+/.+)",
            r"(?:https?://)?(?:www\.)?dx\.doi\.org/(?:10\.\d+/.+)",
            # DOI in URL path
            r"(?:https?://[^/]+)?.*?(?:doi/|DOI:|doi:|DOI/)(\d{2}\.\d+/.+?)(?:[&?#]|$)",
            # Bare DOI pattern
            r"\b(10\.\d+/.+?)(?:\s|$|[&?#])",
            # DOI in query parameters
            r"[\?&]doi=([^&\s]+)",
        ]

    def _load_whitelist(self) -> Dict[str, Any]:
        """Load source whitelist from JSON file."""
        whitelist_path = (
            self.config.whitelist_file_path
            or get_akd_root() / "docs" / "pubs_whitelist.json"
        )

        try:
            with open(whitelist_path, "r", encoding="utf-8") as f:
                whitelist_data = json.load(f)

            if self.debug:
                logger.info(
                    f"Loaded whitelist with {len(whitelist_data.get('data', {}))} categories",
                )

            return whitelist_data
        except Exception as e:
            logger.error(f"Failed to load whitelist from {whitelist_path}: {e}")
            return {"data": {}, "metadata": {}}

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """
        Extract DOI from URL using multiple patterns.

        Args:
            url: URL to extract DOI from

        Returns:
            Extracted DOI or None if not found
        """
        url_str = str(url).strip()

        for pattern in self._doi_patterns:
            matches = re.findall(pattern, url_str, re.IGNORECASE)
            if matches:
                doi = matches[0] if isinstance(matches[0], str) else matches[0][0]
                # Clean up the DOI
                doi = doi.strip().rstrip(".,;)")
                # Ensure DOI starts with 10.
                if not doi.startswith("10."):
                    continue
                return doi

        return None

    async def _fetch_crossref_metadata(
        self,
        session: aiohttp.ClientSession,
        doi: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch source metadata from CrossRef API.

        Args:
            session: aiohttp session
            doi: DOI to fetch metadata for

        Returns:
            Metadata dictionary or None if failed
        """
        url = f"{self.config.crossref_base_url}/{doi}"
        headers = {"User-Agent": self.config.user_agent, "Accept": "application/json"}

        try:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("message", {})
                elif response.status == 404:
                    if self.debug:
                        logger.warning(f"DOI not found in CrossRef: {doi}")
                    return None
                else:
                    logger.warning(
                        f"CrossRef API error {response.status} for DOI: {doi}",
                    )
                    return None

        except aiohttp.ClientError as e:
            logger.warning(f"Network error fetching CrossRef data for {doi}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching CrossRef data for {doi}: {e}")
            return None

    def _parse_crossref_response(
        self,
        data: Dict[str, Any],
        doi: str,
        original_url: str,
    ) -> SourceInfo:
        """
        Parse CrossRef API response into SourceInfo.

        Args:
            data: CrossRef API response data
            doi: Original DOI
            original_url: Original URL

        Returns:
            SourceInfo object
        """
        # Extract source title
        container_title = data.get("container-title", [])
        title = container_title[0] if container_title else "Unknown Source"

        # Extract publisher
        publisher = data.get("publisher")

        # Extract ISSNs
        issn_data = data.get("ISSN", [])
        issn_list = (
            issn_data
            if isinstance(issn_data, list)
            else [issn_data]
            if issn_data
            else []
        )

        # Extract open access information
        is_open_access = None
        license_info = data.get("license", [])
        if license_info:
            # Check for common open access license indicators
            for license_item in license_info:
                license_url = license_item.get("URL", "").lower()
                if any(
                    oa_indicator in license_url
                    for oa_indicator in ["creativecommons", "cc-by", "open", "public"]
                ):
                    is_open_access = True
                    break

        return SourceInfo(
            title=title,
            publisher=publisher,
            issn=issn_list,
            is_open_access=is_open_access,
            doi=doi,
            url=original_url,
        )

    def _validate_against_whitelist(
        self,
        source_info: SourceInfo,
    ) -> tuple[bool, Optional[str], float]:
        """
        Validate source against whitelist.

        Args:
            source_info: Source information from CrossRef

        Returns:
            Tuple of (is_whitelisted, category, confidence_score)
        """
        if not self._whitelist.get("data"):
            return False, None, 0.0

        source_title = source_info.title.lower().strip()

        # Search through all categories in whitelist
        for category_name, category_data in self._whitelist["data"].items():
            sources = category_data.get("journals", [])

            for source_entry in sources:
                if not source_entry or not isinstance(source_entry, dict):
                    continue

                whitelisted_title = (
                    (source_entry.get("Journal Name") or "").lower().strip()
                )
                if not whitelisted_title:
                    continue

                # Exact title match
                if source_title == whitelisted_title:
                    return True, category_name, 1.0
                
                 # if (
                #     whitelisted_title in source_title
                #     or source_title in whitelisted_title
                # ):
                #     # Check if it's a meaningful match (not just common words)
                #     if len(whitelisted_title) > 10 or len(source_title) > 10:
                #         return True, category_name, 0.8
                    

                if self.config.use_fuzzy_match:
                    fuzzy_score_set = token_set_ratio(source_title, whitelisted_title)
                    if fuzzy_score_set >= self.config.fuzzy_threshold:  # You can adjust this threshold
                        return True, category_name, fuzzy_score_set / 100.0  # Normalize to [0, 1]

               
        return False, None, 0.0

    async def _validate_single_result(
        self,
        session: aiohttp.ClientSession,
        result: Any,
    ) -> ValidationResult:
        """Validate a single search result."""
        validation_errors = []

        # Extract DOI
        doi = getattr(result, "doi", None) or self._extract_doi_from_url(
            str(result.url),
        )
        if not doi and hasattr(result, "pdf_url") and result.pdf_url:
            doi = self._extract_doi_from_url(str(result.pdf_url))

        if not doi:
            validation_errors.append("No DOI found in URL or result data")
            return ValidationResult(
                source_info=None,
                is_whitelisted=False,
                whitelist_category=None,
                validation_errors=validation_errors,
                confidence_score=0.0,
            )

        # Fetch metadata from CrossRef
        crossref_data = await self._fetch_crossref_metadata(session, doi)

        if not crossref_data:
            validation_errors.append(
                f"Failed to fetch metadata from CrossRef for DOI: {doi}",
            )
            return ValidationResult(
                source_info=None,
                is_whitelisted=False,
                whitelist_category=None,
                validation_errors=validation_errors,
                confidence_score=0.0,
            )

        # Parse source information
        source_info = self._parse_crossref_response(crossref_data, doi, str(result.url))

        # Validate against whitelist
        is_whitelisted, category, confidence = self._validate_against_whitelist(
            source_info,
        )

        return ValidationResult(
            source_info=source_info,
            is_whitelisted=is_whitelisted,
            whitelist_category=category,
            validation_errors=validation_errors,
            confidence_score=confidence,
        )

    async def _arun(
        self,
        params: SourceValidatorInputSchema,
        **kwargs,
    ) -> SourceValidatorOutputSchema:
        """
        Run the source validation tool.

        Args:
            params: Input parameters

        Returns:
            Validation results
        """
        # Update whitelist path if provided
        if params.whitelist_file_path:
            self.config.whitelist_file_path = params.whitelist_file_path
            self._whitelist = self._load_whitelist()

        validated_results = []

        # Create aiohttp session with semaphore for concurrent requests
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        ) as session:
            # Process results concurrently but with limited concurrency
            import asyncio

            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

            async def validate_with_semaphore(
                result: Any,
            ) -> ValidationResult:
                async with semaphore:
                    return await self._validate_single_result(session, result)

            # Process all results concurrently
            tasks = [
                validate_with_semaphore(result) for result in params.search_results
            ]
            validated_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            final_results = []
            for i, result in enumerate(validated_results):
                if isinstance(result, Exception):
                    logger.error(f"Error validating result {i}: {result}")
                    final_results.append(
                        ValidationResult(
                            source_info=None,
                            is_whitelisted=False,
                            whitelist_category=None,
                            validation_errors=[f"Validation error: {str(result)}"],
                            confidence_score=0.0,
                        ),
                    )
                else:
                    final_results.append(result)

            validated_results = final_results

        # Generate summary statistics
        total_results = len(validated_results)
        whitelisted_count = sum(1 for r in validated_results if r.is_whitelisted)
        error_count = sum(1 for r in validated_results if r.validation_errors)

        # Category breakdown
        category_counts = {}
        for result in validated_results:
            if result.whitelist_category:
                category_counts[result.whitelist_category] = (
                    category_counts.get(result.whitelist_category, 0) + 1
                )

        summary = {
            "total_processed": total_results,
            "whitelisted_count": whitelisted_count,
            "whitelisted_percentage": (whitelisted_count / total_results * 100)
            if total_results > 0
            else 0,
            "error_count": error_count,
            "category_breakdown": category_counts,
            "avg_confidence": sum(r.confidence_score for r in validated_results)
            / total_results
            if total_results > 0
            else 0,
        }

        if self.debug:
            logger.info(f"Validation summary: {summary}")

        return SourceValidatorOutputSchema(
            validated_results=validated_results,
            summary=summary,
        )


def create_source_validator(
    whitelist_file_path: Optional[str] = None,
    timeout_seconds: int = 30,
    max_concurrent_requests: int = 10,
    debug: bool = False,
) -> SourceValidator:
    """
    Factory function to create a source validator tool.

    Args:
        whitelist_file_path: Path to source whitelist JSON file
        timeout_seconds: Timeout for API requests
        max_concurrent_requests: Maximum concurrent requests
        debug: Enable debug logging

    Returns:
        Configured SourceValidator instance
    """
    config = SourceValidatorConfig(
        whitelist_file_path=whitelist_file_path,
        timeout_seconds=timeout_seconds,
        max_concurrent_requests=max_concurrent_requests,
        debug=debug,
    )
    return SourceValidator(config, debug=debug)
