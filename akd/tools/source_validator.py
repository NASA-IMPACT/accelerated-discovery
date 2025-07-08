"""
Source validation tool for validating research sources against a controlled whitelist.

This tool processes raw links from search engines (Semantic Scholar, SearxNG),
extracts DOIs, fetches source metadata from CrossRef API, and validates
against a predefined source whitelist.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import ftfy
import orjson
from crossref_commons.retrieval import get_publication_as_json
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz, process
from unidecode import unidecode

from akd._base import InputSchema, OutputSchema
from akd.tools._base import BaseTool, BaseToolConfig
from akd.utils import get_akd_root

if TYPE_CHECKING:
    from akd.structures import SearchResultItem


class SourceInfo(BaseModel):
    """Schema for source information from CrossRef API."""

    title: str = Field(..., description="Source title", min_length=1)
    publisher: Optional[str] = Field(None, description="Publisher name")
    issn: Optional[List[str]] = Field(None, description="List of ISSNs")
    is_open_access: Optional[bool] = Field(None, description="Open access status")
    doi: str = Field(..., description="DOI of the article", min_length=7)
    url: Optional[str] = Field(None, description="Original URL")

    @field_validator("doi")
    @classmethod
    def validate_doi(cls, v: str) -> str:
        """
        Validate DOI format.

        Args:
            v: DOI string to validate

        Returns:
            Validated DOI string

        Raises:
            ValueError: If DOI format is invalid
        """
        if not v.startswith("10.") or "/" not in v:
            raise ValueError('DOI must start with "10." and contain "/"')
        return v

    @field_validator("issn")
    @classmethod
    def validate_issn(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """
        Validate ISSN format with improved checksum validation.

        Args:
            v: List of ISSN strings to validate

        Returns:
            List of valid ISSN strings or None
        """
        if v is None:
            return v

        def is_valid_issn(issn: str) -> bool:
            """Validate ISSN format and checksum."""
            issn = issn.strip()

            # Basic format check (XXXX-XXXX)
            if not re.match(r"^\d{4}-\d{4}$", issn):
                return False

            # Remove hyphen for checksum calculation
            issn_digits = issn.replace("-", "")

            # ISSN checksum validation
            try:
                total = 0
                for i, digit in enumerate(issn_digits[:7]):
                    total += int(digit) * (8 - i)

                check_digit = issn_digits[7]
                calculated_check = 11 - (total % 11)

                if calculated_check == 10:
                    return check_digit.upper() == "X"
                elif calculated_check == 11:
                    return check_digit == "0"
                else:
                    return check_digit == str(calculated_check)

            except (ValueError, IndexError):
                return False

        validated_issns = []
        for issn in v:
            if is_valid_issn(issn):
                validated_issns.append(issn.strip())
            else:
                # Log invalid ISSN but don't fail validation
                logger.debug(f"Invalid ISSN format or checksum: {issn}")

        return validated_issns if validated_issns else None

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """
        Clean and validate title.

        Args:
            v: Title string to validate

        Returns:
            Cleaned title string

        Raises:
            ValueError: If title is empty
        """
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")

        # Use ftfy to clean the title
        cleaned = ftfy.fix_text(v.strip())
        return cleaned


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
        ge=0.0,
        le=1.0,
    )


class SourceValidatorInputSchema(InputSchema):
    """Input schema for source validation tool."""

    search_results: List["SearchResultItem"] = Field(
        ...,
        description="List of search results to validate",
        min_length=1,
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

    whitelist_file_path: Optional[str] = Field(
        default=None,
        description="Path to source whitelist JSON file",
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum number of concurrent CrossRef API requests",
        gt=0,
        le=50,
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

    # Class-level cache for whitelist data to avoid reloading
    _whitelist_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        config: Optional[SourceValidatorConfig] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the source validator tool.

        Args:
            config: Configuration for the tool
            debug: Enable debug logging

        Raises:
            RuntimeError: If initialization fails
        """
        if config is None:
            config = SourceValidatorConfig()

        # Set default whitelist path if not provided
        if config.whitelist_file_path is None:
            config.whitelist_file_path = str(
                get_akd_root() / "docs" / "pubs_whitelist.json"
            )

        super().__init__(config, debug)

        # Load and validate whitelist on initialization with proper error handling
        try:
            self._whitelist = self._load_whitelist()
            if not self._whitelist.get("data"):
                raise ValueError("Whitelist data is empty or invalid")
        except Exception as e:
            logger.error(f"Failed to initialize source validator: {e}")
            raise RuntimeError(f"Source validator initialization failed: {e}") from e

        # Create searchable index for rapid fuzzy matching
        self._journal_index = self._build_journal_index()

        # Pre-compile DOI extraction patterns for performance
        self._compiled_doi_patterns = self._compile_doi_patterns()

    def _compile_doi_patterns(self) -> List[re.Pattern[str]]:
        """
        Compile optimized DOI extraction patterns for performance.

        Returns:
            List of compiled regex patterns for DOI extraction
        """
        return [
            # Combined standard DOI URLs (consolidating similar patterns)
            re.compile(
                r"(?:https?://)?(?:(?:dx\.|www\.)?doi\.org|dx\.doi\.org)/(10\.\d+/[^\s&?#]+)",
                re.IGNORECASE,
            ),
            # DOI in URL path (consolidated path patterns)
            re.compile(
                r"(?:doi[:/]|DOI[:/]|/doi/|/DOI/)(10\.\d+/[^\s&?#]+)",
                re.IGNORECASE,
            ),
            # DOI in query parameters (consolidated)
            re.compile(r"[\?&]doi=([^&\s#]+)", re.IGNORECASE),
            # URL-encoded DOI patterns
            re.compile(r"(?:doi\.org%2F|doi%3A)(10\.[\d%]+%2F[^&\s#]+)", re.IGNORECASE),
            # Publisher-specific patterns
            re.compile(r"/article/(?:pii/)?[^/]*/?(10\.\d+/[^\s&?#]+)", re.IGNORECASE),
            # Bare DOI pattern (most restrictive, used last)
            re.compile(r"\b(10\.\d{4,}/[^\s&?#]{6,})(?=[\s&?#]|$)", re.IGNORECASE),
        ]

    @classmethod
    def from_params(
        cls,
        whitelist_file_path: Optional[str] = None,
        max_concurrent_requests: int = 10,
        debug: bool = False,
    ) -> "SourceValidator":
        """
        Create a source validator from specific parameters.

        Args:
            whitelist_file_path: Path to source whitelist JSON file
            max_concurrent_requests: Maximum number of concurrent CrossRef API requests
            debug: Enable debug logging

        Returns:
            Configured SourceValidator instance
        """
        config = SourceValidatorConfig(
            whitelist_file_path=whitelist_file_path,
            max_concurrent_requests=max_concurrent_requests,
        )
        return cls(config, debug=debug)

    @classmethod
    def clear_whitelist_cache(cls) -> None:
        """Clear the whitelist cache. Useful for testing or when whitelist files change."""
        cls._whitelist_cache.clear()

    def _load_whitelist(self) -> Dict[str, Any]:
        """
        Load source whitelist from JSON file with caching and proper error handling.

        Returns:
            Whitelist data dictionary

        Raises:
            FileNotFoundError: If whitelist file doesn't exist
            ValueError: If whitelist structure is invalid
        """
        whitelist_path = self.config.whitelist_file_path

        if not whitelist_path:
            raise ValueError("Whitelist file path not configured")

        # Check cache first
        if whitelist_path in self._whitelist_cache:
            if self.debug:
                logger.info(f"Using cached whitelist for: {whitelist_path}")
            return self._whitelist_cache[whitelist_path]

        try:
            with open(whitelist_path, "rb") as f:  # orjson requires binary mode
                whitelist_data = orjson.loads(f.read())

            # Validate whitelist structure
            if not isinstance(whitelist_data, dict):
                raise ValueError("Whitelist must be a JSON object")

            if "data" not in whitelist_data:
                raise ValueError('Whitelist must contain "data" key')

            if not isinstance(whitelist_data["data"], dict):
                raise ValueError('Whitelist "data" must be an object')

            # Cache the validated whitelist
            self._whitelist_cache[whitelist_path] = whitelist_data

            if self.debug:
                categories_count = len(whitelist_data.get("data", {}))
                logger.info(
                    f"Loaded and cached whitelist with {categories_count} categories"
                )

            return whitelist_data

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Whitelist file not found: {whitelist_path}"
            ) from e
        except (orjson.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Invalid JSON in whitelist file: {whitelist_path}. Error: {e}"
            ) from e

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """
        Extract DOI from URL using pre-compiled patterns.

        Args:
            url: URL to extract DOI from

        Returns:
            Extracted DOI or None if not found
        """
        if not url:
            return None

        url_str = str(url).strip()
        if not url_str:
            return None

        # Try with original URL first
        for pattern in self._compiled_doi_patterns:
            match = pattern.search(url_str)
            if match:
                doi = match.group(1).strip().rstrip(".,;)")

                # Handle URL-encoded DOIs
                if "%2F" in doi or "%3A" in doi:
                    doi = urllib.parse.unquote(doi)

                # Validate DOI format with improved checks
                if self._is_valid_doi_format(doi):
                    return doi

        # If no DOI found, try URL-decoding the entire URL and search again
        try:
            decoded_url = urllib.parse.unquote(url_str)
            if decoded_url != url_str:
                for pattern in self._compiled_doi_patterns:
                    match = pattern.search(decoded_url)
                    if match:
                        doi = match.group(1).strip().rstrip(".,;)")
                        if self._is_valid_doi_format(doi):
                            return doi
        except Exception as e:
            logger.debug(f"Error during URL decoding: {e}")

        return None

    def _is_valid_doi_format(self, doi: str) -> bool:
        """
        Validate DOI format with comprehensive checks according to DOI standards.

        Args:
            doi: DOI string to validate

        Returns:
            True if DOI format is valid, False otherwise
        """
        if not doi or not isinstance(doi, str):
            return False

        # Basic structure check
        if not doi.startswith("10."):
            return False

        if "/" not in doi:
            return False

        # Minimum length check (e.g., "10.1/a" is theoretical minimum)
        if len(doi) < 6:
            return False

        # Split into prefix and suffix
        parts = doi.split("/", 1)
        if len(parts) != 2:
            return False

        prefix, suffix = parts

        # Validate prefix: must be "10." followed by 4 or more digits
        if not re.match(r"^10\.\d{4,}$", prefix):
            return False

        # Validate suffix: must not be empty and contain valid characters
        if not suffix or len(suffix.strip()) == 0:
            return False

        # Check for invalid characters in suffix (very permissive as per DOI spec)
        # DOI suffix can contain most printable characters except spaces and some control chars
        if re.search(r"[\s\x00-\x1f\x7f]", suffix):
            return False

        # Additional check: suffix should have reasonable length
        if len(suffix) > 1000:  # Extremely long suffixes are suspicious
            return False

        return True

    async def _fetch_crossref_metadata_simple(
        self, doi: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch CrossRef metadata using crossref-commons library with improved error handling.

        Args:
            doi: DOI to look up

        Returns:
            CrossRef metadata dictionary or None if not found
        """
        import asyncio
        import concurrent.futures

        try:
            # Use ThreadPoolExecutor for better control over thread safety
            # instead of asyncio.to_thread() which may have issues with crossref-commons
            def fetch_doi_sync() -> Optional[Dict[str, Any]]:
                """Synchronous wrapper for crossref-commons call."""
                try:
                    result = get_publication_as_json(doi)
                    return result
                except Exception as e:
                    logger.warning(f"CrossRef request failed for DOI {doi}: {e}")
                    return None

            # Use ThreadPoolExecutor for better thread management
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_doi_sync)
                try:
                    # Add timeout to prevent hanging requests
                    result = await asyncio.wait_for(
                        asyncio.wrap_future(future),
                        timeout=30.0,  # 30 second timeout for CrossRef requests
                    )

                    if result and self.debug:
                        logger.info(
                            f"Successfully fetched CrossRef metadata for DOI: {doi}"
                        )

                    return result

                except asyncio.TimeoutError:
                    logger.warning(f"CrossRef request timeout after 30s for DOI: {doi}")
                    future.cancel()
                    return None

        except Exception as e:
            logger.warning(f"CrossRef API error for DOI {doi}: {e}")
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

        Raises:
            ValueError: If required data is missing or invalid
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
        is_open_access = self._determine_open_access_status(data)

        return SourceInfo(
            title=title,
            publisher=publisher,
            issn=issn_list,
            is_open_access=is_open_access,
            doi=doi,
            url=original_url,
        )

    def _determine_open_access_status(self, data: Dict[str, Any]) -> Optional[bool]:
        """
        Determine open access status from CrossRef data.

        Args:
            data: CrossRef API response data

        Returns:
            True if open access, False if not, None if unknown
        """
        license_info = data.get("license", [])
        if license_info:
            # Check for legitimate open access license indicators
            open_access_indicators = [
                "creativecommons.org",
                "cc-by",
                "cc-zero",
                "cc0",
                "cc-by-sa",
                "cc-by-nc",
                "cc-by-nc-sa",
                "cc-by-nc-nd",
                "cc-by-nd",
                "opensource.org",
                "gnu.org/licenses",
                "apache.org/licenses",
                "mit-license",
                "bsd-license",
                "/publicdomain/",
                "unlicense.org",
            ]
            for license_item in license_info:
                license_url = license_item.get("URL", "").lower()
                if any(
                    indicator in license_url for indicator in open_access_indicators
                ):
                    return True

        return None

    def _normalize_journal_title(self, title: str) -> str:
        """
        Normalize journal title using ftfy, unidecode and common abbreviations.

        Args:
            title: Raw journal title

        Returns:
            Normalized title
        """
        if not title:
            return ""

        # Use ftfy to fix text encoding issues first
        cleaned = ftfy.fix_text(title)

        # Use unidecode for unicode normalization
        normalized = unidecode(cleaned).lower().strip()

        # Common journal abbreviations (reduced set - most important ones)
        abbreviations = {
            "&": "and",
            "j.": "journal",
            "rev.": "review",
            "res.": "research",
            "lett.": "letters",
            "sci.": "science",
            "phys.": "physics",
            "geophys.": "geophysical",
            "astrophys.": "astrophysical",
            "astron.": "astronomical",
            "proc.": "proceedings",
        }

        for abbrev, full in abbreviations.items():
            normalized = normalized.replace(abbrev, full)

        # Clean up punctuation and whitespace
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    def _calculate_similarity_score(self, str1: str, str2: str) -> float:
        """
        Calculate similarity using rapidfuzz with weighted scoring for better accuracy.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not str1 or not str2:
            return 0.0

        # Normalize both strings
        norm1 = self._normalize_journal_title(str1)
        norm2 = self._normalize_journal_title(str2)

        if norm1 == norm2:
            return 1.0

        # Use rapidfuzz for multiple similarity metrics with weighted scoring
        ratio = fuzz.ratio(norm1, norm2) / 100
        partial = fuzz.partial_ratio(norm1, norm2) / 100
        token_sort = fuzz.token_sort_ratio(norm1, norm2) / 100
        token_set = fuzz.token_set_ratio(norm1, norm2) / 100

        # Weighted average with emphasis on token_set for journal matching
        # token_set is most important for journal names with reordered words
        weighted_score = (
            ratio * 0.2 + partial * 0.2 + token_sort * 0.25 + token_set * 0.35
        )

        return weighted_score

    def _validate_against_whitelist(
        self,
        source_info: SourceInfo,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Validate source against whitelist using rapidfuzz for efficient fuzzy matching.

        Args:
            source_info: Source information from CrossRef

        Returns:
            Tuple of (is_whitelisted, category, confidence_score)
        """
        if not self._journal_index:
            return False, None, 0.0

        source_title = self._normalize_journal_title(source_info.title)
        if not source_title:
            return False, None, 0.0

        # Use rapidfuzz for efficient fuzzy matching across all journals
        try:
            result = process.extractOne(
                source_title,
                self._journal_index.keys(),
                scorer=fuzz.token_set_ratio,
                score_cutoff=75,  # Minimum score threshold
            )

            if result:
                matched_title, score, _ = result
                original_title, category = self._journal_index[matched_title]
                confidence = score / 100.0

                if self.debug:
                    logger.info(
                        f'Matched "{source_info.title}" -> "{original_title}" '
                        f"(score: {score}, category: {category})"
                    )

                return True, category, confidence

        except Exception as e:
            logger.warning(f"Error during fuzzy matching: {e}")

        return False, None, 0.0

    async def _validate_single_result(
        self,
        result: "SearchResultItem",
    ) -> ValidationResult:
        """
        Validate a single search result.

        Args:
            result: Search result to validate

        Returns:
            ValidationResult with detailed validation information
        """
        validation_errors: List[str] = []

        # Extract DOI with multiple fallback strategies
        doi = None
        if hasattr(result, "doi") and result.doi:
            doi = result.doi

        if not doi:
            doi = self._extract_doi_from_url(str(result.url))

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
        crossref_data = await self._fetch_crossref_metadata_simple(doi)

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
        try:
            source_info = self._parse_crossref_response(
                crossref_data, doi, str(result.url)
            )
        except Exception as e:
            validation_errors.append(f"Failed to parse CrossRef response: {e}")
            return ValidationResult(
                source_info=None,
                is_whitelisted=False,
                whitelist_category=None,
                validation_errors=validation_errors,
                confidence_score=0.0,
            )

        # Validate against whitelist
        try:
            is_whitelisted, category, confidence = self._validate_against_whitelist(
                source_info
            )
        except Exception as e:
            validation_errors.append(f"Error during whitelist validation: {e}")
            is_whitelisted, category, confidence = False, None, 0.0
            if self.debug:
                logger.error(f"Whitelist validation error for DOI {doi}: {e}")

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
        **kwargs: Any,
    ) -> SourceValidatorOutputSchema:
        """
        Run the source validation tool.

        Args:
            params: Input parameters
            **kwargs: Additional keyword arguments

        Returns:
            Validation results
        """
        # Validate input
        if not params.search_results:
            return SourceValidatorOutputSchema(
                validated_results=[],
                summary={
                    "total_processed": 0,
                    "whitelisted_count": 0,
                    "whitelisted_percentage": 0.0,
                    "error_count": 0,
                    "category_breakdown": {},
                    "avg_confidence": 0.0,
                },
            )

        validated_results: List[ValidationResult] = []

        # Process results with controlled concurrency using crossref-commons
        # No need for HTTP session management as crossref-commons handles it internally
        try:
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

            async def validate_with_semaphore(
                result: "SearchResultItem",
            ) -> ValidationResult:
                async with semaphore:
                    return await self._validate_single_result(result)

            # Process all results concurrently
            tasks = [
                validate_with_semaphore(result) for result in params.search_results
            ]

            # Handle results and exceptions properly
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error validating result {i}: {result}")
                    validated_results.append(
                        ValidationResult(
                            source_info=None,
                            is_whitelisted=False,
                            whitelist_category=None,
                            validation_errors=[f"Validation error: {str(result)}"],
                            confidence_score=0.0,
                        ),
                    )
                else:
                    validated_results.append(result)

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            # Create error results for all inputs
            validated_results = [
                ValidationResult(
                    source_info=None,
                    is_whitelisted=False,
                    whitelist_category=None,
                    validation_errors=[f"Validation error: {str(e)}"],
                    confidence_score=0.0,
                )
                for _ in params.search_results
            ]

        # Generate summary statistics
        summary = self._generate_summary_statistics(validated_results)

        if self.debug:
            logger.info(f"Validation summary: {summary}")

        return SourceValidatorOutputSchema(
            validated_results=validated_results,
            summary=summary,
        )

    def _generate_summary_statistics(
        self, validated_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from validation results.

        Args:
            validated_results: List of validation results

        Returns:
            Dictionary containing summary statistics
        """
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

        # Calculate average confidence (only for non-error results)
        confidence_scores = [
            r.confidence_score for r in validated_results if not r.validation_errors
        ]
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

        return {
            "total_processed": total_results,
            "whitelisted_count": whitelisted_count,
            "whitelisted_percentage": (whitelisted_count / total_results * 100)
            if total_results > 0
            else 0.0,
            "error_count": error_count,
            "category_breakdown": category_counts,
            "avg_confidence": avg_confidence,
        }

    def _build_journal_index(self) -> Dict[str, Tuple[str, str]]:
        """
        Build a searchable index of journal titles for fast fuzzy matching.

        Returns:
            Dictionary mapping normalized titles to (original_title, category)
        """
        index: Dict[str, Tuple[str, str]] = {}

        for category_name, category_data in self._whitelist.get("data", {}).items():
            sources = category_data.get("journals", [])

            for source_entry in sources:
                if not source_entry or not isinstance(source_entry, dict):
                    continue

                original_title = source_entry.get("Journal Name", "")
                if not original_title:
                    continue

                normalized_title = self._normalize_journal_title(original_title)
                if normalized_title:
                    index[normalized_title] = (original_title, category_name)

        if self.debug:
            logger.info(f"Built journal index with {len(index)} entries")

        return index


def create_source_validator(
    whitelist_file_path: Optional[str] = None,
    max_concurrent_requests: int = 10,
    debug: bool = False,
) -> SourceValidator:
    """
    Factory function to create a source validator tool.

    Args:
        whitelist_file_path: Path to source whitelist JSON file
        max_concurrent_requests: Maximum concurrent CrossRef API requests
        debug: Enable debug logging

    Returns:
        Configured SourceValidator instance
    """
    config = SourceValidatorConfig(
        whitelist_file_path=whitelist_file_path,
        max_concurrent_requests=max_concurrent_requests,
    )
    return SourceValidator(config, debug=debug)
