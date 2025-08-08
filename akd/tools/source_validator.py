"""
ISSN-based source validation tool.

This tool processes raw links from search engines (Semantic Scholar, SearxNG),
extracts DOIs, fetches source metadata from the CrossRef API, and validates
the publication venue by comparing its ISSNs (print/electronic) against a
configured ISSN whitelist. This simplifies whitelist management by relying on
stable identifiers rather than fuzzy title matching.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

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

    title: str = Field(..., description="Source title (journal/proceedings)")
    publisher: Optional[str] = Field(None, description="Publisher name")
    issn: List[str] = Field(
        default_factory=list, description="List of normalized ISSNs"
    )
    is_open_access: Optional[bool] = Field(None, description="Open access status")
    doi: str = Field(..., description="DOI of the article")
    url: Optional[str] = Field(None, description="Original URL")


class ValidationResult(BaseModel):
    """Schema for validation result."""

    source_info: Optional[SourceInfo] = Field(
        None,
        description="Source information from CrossRef",
    )
    is_whitelisted: bool = Field(..., description="Whether venue ISSN is in whitelist")
    whitelist_category: Optional[str] = Field(
        None,
        description="Category from whitelist (if provided)",
    )
    matched_issn: Optional[str] = Field(
        None,
        description="The specific ISSN that matched the whitelist, if any",
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
        description=(
            "Path to ISSN whitelist JSON file. Accepts either a flat list of ISSNs, "
            "an object with 'issn' list, or a category map {category: [issns...]}. "
            "If None, uses default path."
        ),
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
            get_akd_root() / "docs" / "issn_whitelist.json",
        ),
        description=(
            "Path to ISSN whitelist JSON file. Can be a flat list of ISSNs, "
            "an object with key 'issn', or a category map {category: [issns...]}."
        ),
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

        # Load whitelist on initialization (ISSN-based)
        self._allowed_issn_set: Set[str] = set()
        self._issn_to_category: Dict[str, str] = {}
        self._load_whitelist()

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

    def _load_whitelist(self) -> None:
        """Load ISSN whitelist from JSON file into in-memory structures.

        Supported JSON formats:
        - ["1234-5678", "8765-4321"]
        - {"issn": ["1234-5678", ...]}
        - {"Bio": ["1234-5678"], "CS": ["8765-4321"]}
        - {"categories": {"Bio": ["1234-5678"]}}
        """
        whitelist_path = self.config.whitelist_file_path or str(
            get_akd_root() / "docs" / "issn_whitelist.json"
        )

        try:
            with open(whitelist_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            allowed_issn: Set[str] = set()
            issn_to_category: Dict[str, str] = {}

            def normalize_issn(issn: str) -> Optional[str]:
                return self._normalize_issn(issn)

            if isinstance(raw, list):
                for value in raw:
                    norm = normalize_issn(str(value)) if value is not None else None
                    if norm is not None:
                        allowed_issn.add(norm)
            elif isinstance(raw, dict):
                if "issn" in raw and isinstance(raw["issn"], list):
                    for value in raw["issn"]:
                        norm = normalize_issn(str(value)) if value is not None else None
                        if norm is not None:
                            allowed_issn.add(norm)
                if "categories" in raw and isinstance(raw["categories"], dict):
                    for category, values in raw["categories"].items():
                        if not isinstance(values, list):
                            continue
                        for value in values:
                            norm = (
                                normalize_issn(str(value))
                                if value is not None
                                else None
                            )
                            if norm is not None:
                                allowed_issn.add(norm)
                                issn_to_category[norm] = str(category)
                # Treat any other key as category map if value is list
                for key, value in raw.items():
                    if key in {"issn", "categories"}:
                        continue
                    if isinstance(value, list):
                        for v in value:
                            norm = normalize_issn(str(v)) if v is not None else None
                            if norm is not None:
                                allowed_issn.add(norm)
                                issn_to_category[norm] = str(key)
            else:
                raise ValueError("Unsupported ISSN whitelist format")

            self._allowed_issn_set = allowed_issn
            self._issn_to_category = issn_to_category

            if self.debug:
                logger.info(
                    "Loaded ISSN whitelist: %d ISSNs across %d categories",
                    len(self._allowed_issn_set),
                    len(set(issn_to_category.values())),
                )
        except FileNotFoundError:
            logger.warning(
                f"ISSN whitelist file not found at {whitelist_path}. Validation will allow none.",
            )
            self._allowed_issn_set = set()
            self._issn_to_category = {}
        except Exception as e:
            logger.error(f"Failed to load ISSN whitelist from {whitelist_path}: {e}")
            self._allowed_issn_set = set()
            self._issn_to_category = {}

    @staticmethod
    def _normalize_issn(issn: str) -> Optional[str]:
        """Normalize an ISSN string to the canonical form NNNN-NNNN.

        - Removes all non-alphanumeric characters
        - Uppercases 'x' to 'X'
        - Inserts hyphen after 4th character if length is 8
        - Returns None if normalization is not possible
        """
        if issn is None:
            return None
        candidate = re.sub(r"[^0-9xX]", "", issn).upper()
        if len(candidate) != 8:
            return None
        return f"{candidate[:4]}-{candidate[4:]}"

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
        publisher: Optional[str] = data.get("publisher")

        # Extract and normalize ISSNs from both 'ISSN' and 'issn-type'
        issn_values: List[str] = []
        if isinstance(data.get("ISSN"), list):
            issn_values.extend([str(v) for v in data.get("ISSN", [])])
        elif data.get("ISSN"):
            issn_values.append(str(data.get("ISSN")))

        issn_type = data.get("issn-type", [])
        if isinstance(issn_type, list):
            for item in issn_type:
                value = item.get("value") if isinstance(item, dict) else None
                if value:
                    issn_values.append(str(value))

        normalized_issn: List[str] = []
        for raw in issn_values:
            norm = self._normalize_issn(raw)
            if norm is not None and norm not in normalized_issn:
                normalized_issn.append(norm)

        # Extract open access information (best-effort)
        is_open_access: Optional[bool] = None
        license_info = data.get("license", [])
        if isinstance(license_info, list):
            for license_item in license_info:
                license_url = (
                    str(license_item.get("URL", "")).lower()
                    if isinstance(license_item, dict)
                    else ""
                )
                if any(
                    oa_indicator in license_url
                    for oa_indicator in ["creativecommons", "cc-by", "open", "public"]
                ):
                    is_open_access = True
                    break

        return SourceInfo(
            title=title,
            publisher=publisher,
            issn=normalized_issn,
            is_open_access=is_open_access,
            doi=doi,
            url=original_url,
        )

    def _validate_against_whitelist(
        self,
        source_info: SourceInfo,
    ) -> Tuple[bool, Optional[str], float, Optional[str]]:
        """Validate venue against the ISSN whitelist.

        Returns a tuple (is_whitelisted, category, confidence, matched_issn).
        """
        if not self._allowed_issn_set:
            return False, None, 0.0, None

        for issn in source_info.issn:
            if issn in self._allowed_issn_set:
                category = self._issn_to_category.get(issn)
                return True, category, 1.0, issn

        return False, None, 0.0, None

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
                matched_issn=None,
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
                matched_issn=None,
                validation_errors=validation_errors,
                confidence_score=0.0,
            )

        # Parse source information
        source_info = self._parse_crossref_response(crossref_data, doi, str(result.url))

        if not source_info.issn:
            validation_errors.append("No ISSN found in CrossRef metadata")

        # Validate against whitelist
        is_whitelisted, category, confidence, matched_issn = (
            self._validate_against_whitelist(
                source_info,
            )
        )

        return ValidationResult(
            source_info=source_info,
            is_whitelisted=is_whitelisted,
            whitelist_category=category,
            matched_issn=matched_issn,
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
            self._load_whitelist()

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
                            matched_issn=None,
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
        category_counts: Dict[str, int] = {}
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
