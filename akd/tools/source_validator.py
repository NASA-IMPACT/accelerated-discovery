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
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from akd._base import InputSchema, OutputSchema
from akd.structures import SearchResultItem
from akd.tools._base import BaseTool, BaseToolConfig
from akd.utils import get_akd_root


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
    allow_arxiv: bool = Field(
        default=True,
        description=(
            "When true, allow arXiv results to pass validation without ISSN checks."
        ),
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
        whitelist_path = self.config.whitelist_file_path or str(
            get_akd_root() / "docs" / "issn_whitelist.json"
        )

        try:
            with open(whitelist_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            allowed_issn: Set[str] = set()
            issn_to_category: Dict[str, str] = {}

            if isinstance(raw, list):
                for value in raw:
                    if value is not None:
                        norm = self._normalize_issn(str(value))
                        if norm:
                            allowed_issn.add(norm)
            elif isinstance(raw, dict):
                if "issn" in raw and isinstance(raw["issn"], list):
                    for value in raw["issn"]:
                        if value is not None:
                            norm = self._normalize_issn(str(value))
                            if norm:
                                allowed_issn.add(norm)
                if "categories" in raw and isinstance(raw["categories"], dict):
                    for category, values in raw["categories"].items():
                        if isinstance(values, list):
                            for value in values:
                                if value is not None:
                                    norm = self._normalize_issn(str(value))
                                    if norm:
                                        allowed_issn.add(norm)
                                        issn_to_category[norm] = str(category)
                for key, value in raw.items():
                    if key not in {"issn", "categories"} and isinstance(value, list):
                        for v in value:
                            if v is not None:
                                norm = self._normalize_issn(str(v))
                                if norm:
                                    allowed_issn.add(norm)
                                    issn_to_category[norm] = str(key)
            else:
                raise ValueError("Unsupported ISSN whitelist format")

            self._allowed_issn_set = allowed_issn
            self._issn_to_category = issn_to_category

            if self.debug:
                logger.info(
                    f"Loaded {len(self._allowed_issn_set)} ISSNs in {len(set(issn_to_category.values()))} categories"
                )
        except FileNotFoundError:
            logger.warning(f"ISSN whitelist not found: {whitelist_path}")
            self._allowed_issn_set = set()
            self._issn_to_category = {}
        except Exception as e:
            logger.error(f"Failed to load whitelist: {e}")
            self._allowed_issn_set = set()
            self._issn_to_category = {}

    @staticmethod
    def _normalize_issn(issn: str) -> Optional[str]:
        if not issn:
            return None
        candidate = re.sub(r"[^0-9xX]", "", issn).upper()
        return f"{candidate[:4]}-{candidate[4:]}" if len(candidate) == 8 else None

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        for pattern in self._doi_patterns:
            matches = re.findall(pattern, str(url).strip(), re.IGNORECASE)
            if matches:
                doi = matches[0] if isinstance(matches[0], str) else matches[0][0]
                doi = doi.strip().rstrip(".,;)")
                if doi.startswith("10."):
                    return doi
        return None

    async def _fetch_crossref_metadata(
        self,
        session: aiohttp.ClientSession,
        doi: str,
    ) -> Optional[Dict[str, Any]]:
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
                        logger.debug(f"DOI not found: {doi}")
                    return None
                else:
                    logger.warning(f"CrossRef error {response.status}: {doi}")
                    return None
        except Exception as e:
            if self.debug:
                logger.warning(f"CrossRef fetch failed {doi}: {e}")
            return None

    def _parse_crossref_response(
        self,
        data: Dict[str, Any],
        doi: str,
        original_url: str,
    ) -> SourceInfo:
        container_title = data.get("container-title", [])
        title = container_title[0] if container_title else "Unknown Source"
        publisher = data.get("publisher")

        issn_values = []
        if isinstance(data.get("ISSN"), list):
            issn_values.extend(str(v) for v in data["ISSN"])
        elif data.get("ISSN"):
            issn_values.append(str(data["ISSN"]))

        for item in data.get("issn-type", []):
            if isinstance(item, dict) and item.get("value"):
                issn_values.append(str(item["value"]))

        normalized_issn = []
        for raw in issn_values:
            norm = self._normalize_issn(raw)
            if norm and norm not in normalized_issn:
                normalized_issn.append(norm)

        is_open_access = None
        for license_item in data.get("license", []):
            if isinstance(license_item, dict):
                license_url = str(license_item.get("URL", "")).lower()
                if any(
                    oa in license_url
                    for oa in ["creativecommons", "cc-by", "open", "public"]
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
        if getattr(self.config, "allow_arxiv", False):
            url_lower = (source_info.url or "").lower()
            doi_lower = (source_info.doi or "").lower()
            is_arxiv = "arxiv.org" in url_lower or doi_lower.startswith(
                ("10.48550/arxiv", "10.48550/ARXIV")
            )
            if is_arxiv:
                if self.debug:
                    logger.debug(f"ACCEPT (arXiv): {source_info.title}")
                return True, "arXiv", 1.0, None

        if not self._allowed_issn_set:
            if self.debug:
                logger.debug(
                    f"REJECT (empty whitelist): {source_info.title} issns={source_info.issn}"
                )
            return False, None, 0.0, None

        for issn in source_info.issn:
            if issn in self._allowed_issn_set:
                category = self._issn_to_category.get(issn)
                if self.debug:
                    logger.debug(
                        f"ACCEPT: {source_info.title} matched_issn={issn} category={category}"
                    )
                return True, category, 1.0, issn

        if self.debug:
            logger.debug(f"REJECT: {source_info.title} issns={source_info.issn}")
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

        if self.debug:
            logger.debug(f"Validating DOI: {doi} from {result.url}")

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

        if self.debug:
            logger.debug(f"Parsed: {source_info.title} issns={source_info.issn}")

        if not source_info.issn:
            validation_errors.append("No ISSN found in CrossRef metadata")

        is_whitelisted, category, confidence, matched_issn = (False, None, 0.0, None)
        if self._allowed_issn_set:
            is_whitelisted, category, confidence, matched_issn = (
                self._validate_against_whitelist(source_info)
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
    ) -> SourceValidatorOutputSchema:
        if params.whitelist_file_path:
            self.config.whitelist_file_path = params.whitelist_file_path
            self._load_whitelist()

        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        ) as session:
            import asyncio

            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

            async def validate_with_semaphore(result: Any) -> ValidationResult:
                async with semaphore:
                    return await self._validate_single_result(session, result)

            tasks = [
                validate_with_semaphore(result) for result in params.search_results
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            validated_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Validation error {i}: {result}")
                    validated_results.append(
                        ValidationResult(
                            source_info=None,
                            is_whitelisted=False,
                            whitelist_category=None,
                            matched_issn=None,
                            validation_errors=[f"Validation error: {str(result)}"],
                            confidence_score=0.0,
                        )
                    )
                else:
                    validated_results.append(result)

        total = len(validated_results)
        whitelisted = sum(1 for r in validated_results if r.is_whitelisted)
        errors = sum(1 for r in validated_results if r.validation_errors)

        category_counts = {}
        for result in validated_results:
            if result.whitelist_category:
                category_counts[result.whitelist_category] = (
                    category_counts.get(result.whitelist_category, 0) + 1
                )

        summary = {
            "total_processed": total,
            "whitelisted_count": whitelisted,
            "whitelisted_percentage": (whitelisted / total * 100) if total > 0 else 0,
            "error_count": errors,
            "category_breakdown": category_counts,
            "avg_confidence": sum(r.confidence_score for r in validated_results) / total
            if total > 0
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
    config = SourceValidatorConfig(
        whitelist_file_path=whitelist_file_path,
        timeout_seconds=timeout_seconds,
        max_concurrent_requests=max_concurrent_requests,
        debug=debug,
    )
    return SourceValidator(config, debug=debug)
