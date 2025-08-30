import os
from typing import Literal, Optional, Union

import aiohttp
import httpx
from loguru import logger
from pydantic import Field, HttpUrl, field_validator
from rapidfuzz import fuzz

from akd.structures import SearchResultItem

from ._base import (
    ArticleResolverConfig,
    BaseArticleResolver,
    ResolverInputSchema,
    ResolverOutputSchema,
)


class CrossRefDoiResolverInputSchema(ResolverInputSchema):
    """
    Input schema for CrossRef DOI resolver.
    Inherits from ResolverInputSchema to ensure compatibility with search results.
    """

    # makes title  required for CrossRef DOI resolution
    title: str = Field(
        ...,
        description="Title of the article to resolve DOI",
    )

    # overirde to make url optional for CrossRef DOI resolution
    url: Optional[HttpUrl] = Field(
        None,
        description="URL of the article",
    )


class CrossRefDoiResolverOutputSchema(CrossRefDoiResolverInputSchema, ResolverOutputSchema):
    """
    Output schema for CrossRef DOI resolver.
    Inherits from ResolverOutputSchema to ensure compatibility with search results.
    """
    pass 


class CrossRefDoiResolverConfig(ArticleResolverConfig):
    """Configuration for the CrossRef DOI resolver."""

    validate_resolved_url: bool = Field(
        default=False,
        description="Whether to validate that the resolved URL is accessible (forced False in CrossRef config)",
    )

    cross_ref_api_url: str = Field(
        default=os.getenv("CROSS_REF_API_URL", "https://api.crossref.org/works"),
        description="Base URL for CrossRef API",
    )

    timeout_seconds: int = Field(
        default=30,
        description="Timeout for API requests in seconds",
    )

    title_fuzzy_threshold: int = Field(
        default=85,
        description="Fuzzy matching threshold for title matching (should be high for accuracy)",
    )

    author_fuzzy_threshold: int = Field(
        default=70,
        description="Fuzzy matching threshold for author matching",
    )

    max_authors_to_check: int = Field(
        default=5,
        description="Maximum number of authors to check for matching (0 = no limit)",
    )

    author_priority_strategy: Literal["first", "last", "first_and_last", "all"] = Field(
        default="first_and_last",
        description="Strategy for selecting which authors to prioritize when capping",
    )

    title_fuzzy_method: Literal["token_set", "token_sort", "ratio", "partial_ratio"] = (
        Field(
            default="ratio",
            description="Fuzzy score method for titles - ratio is more strict than token_set",
        )
    )

    max_candidates: int = Field(
        default=10,
        description="Maximum number of results to process from CrossRef API response",
    )

    author_fuzzy_method: Literal[
        "token_set",
        "token_sort",
        "ratio",
        "partial_ratio",
    ] = Field(
        default="token_set",
        description="Fuzzy score method for authors - token_set works well for name variations",
    )

    user_agent: Optional[str] = Field(
        default="CrossRefResolver/1.0",
        description="User agent for HTTP requests",
    )

    @field_validator("validate_resolved_url")
    @classmethod
    def force_validate_url_false(cls, v):
        return False  # Always return False regardless of input


class CrossRefDoiResolver(BaseArticleResolver):
    """
    Resolver that queries the CrossRef API using title and authors
    and resolves to a DOI and URL if a match is found.
    """

    config_schema = CrossRefDoiResolverConfig

    _fuzzy_scorer_map = {
        "token_set": fuzz.token_set_ratio,
        "token_sort": fuzz.token_sort_ratio,
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
    }

    def validate_url(self, url: Union[str, HttpUrl]) -> bool:
        """CrossRefResolver accepts any URL"""
        return True

    def build_author_names(self, author_entries: list[dict]) -> list[str]:
        names = []
        for entry in author_entries:
            given = entry.get("given", "").strip()
            family = entry.get("family", "").strip()
            if given or family:
                names.append(f"{given} {family}".strip())
        return names

    async def get_authors_from_search_result(
        self,
        result: SearchResultItem,
    ) -> list[str]:
        """
        Extract authors from search result. Utility function to be used if needed.
        """
        return (
            result.extra.get("authors")
            if result.extra and result.extra.get("authors")
            else result.authors
            if result.authors
            else []
        )

    def normalize_title(self, title: str) -> str:
        """Normalize title for comparison by removing common variations"""
        import re

        # Convert to lowercase and strip
        title = title.lower().strip()
        # Remove extra whitespace
        title = re.sub(r"\s+", " ", title)
        # Remove common punctuation at the end
        title = re.sub(r"[.,:;!?]+$", "", title)
        return title

    def select_priority_authors(self, authors: list[str]) -> list[str]:
        """
        Select priority authors based on the configured strategy.

        Args:
            authors: Full list of authors

        Returns:
            Filtered list of authors based on priority strategy
        """
        if not authors:
            return []

        max_authors = self.config.max_authors_to_check
        if max_authors == 0 or len(authors) <= max_authors:
            return authors

        strategy = self.config.author_priority_strategy

        if strategy == "first":
            # Take first N authors (most common approach)
            return authors[:max_authors]
        elif strategy == "last":
            # Take last N authors (sometimes senior/corresponding authors are last)
            return authors[-max_authors:]
        elif strategy == "first_and_last":
            # Take first author(s) and last author(s) - captures both first and senior authors
            if max_authors == 1:
                return [authors[0]]  # Just first if only one allowed
            elif max_authors == 2:
                return [authors[0], authors[-1]] if len(authors) > 1 else authors
            else:
                # Split between first and last authors
                first_count = max_authors // 2
                last_count = max_authors - first_count

                if len(authors) <= max_authors:
                    return authors

                selected = authors[:first_count]
                # Avoid duplicates if lists overlap
                last_authors = authors[-last_count:] if last_count > 0 else []
                for author in last_authors:
                    if author not in selected:
                        selected.append(author)

                return selected[:max_authors]  # Ensure we don't exceed limit
        else:  # strategy == "all"
            return authors

    def calculate_author_match_score(
        self,
        input_authors: list[str],
        result_authors: list[str],
    ) -> float:
        """
        Calculate a match score between input and result authors.
        Returns a score from 0-100 based on how well the author lists match.
        """
        if not input_authors or not result_authors:
            return 0.0 if input_authors or result_authors else 100.0

        # Apply priority selection to both input and result authors
        input_priority = self.select_priority_authors(input_authors)
        result_priority = self.select_priority_authors(result_authors)

        if self.debug:
            logger.debug(
                f"Priority authors - Input: {input_priority} (from {len(input_authors)} total) | Result: {result_priority} (from {len(result_authors)} total)",
            )

        author_scorer = self._fuzzy_scorer_map.get(
            self.config.author_fuzzy_method,
            fuzz.token_set_ratio,
        )
        threshold = self.config.author_fuzzy_threshold

        input_authors_norm = [a.lower().strip() for a in input_priority]
        result_authors_norm = [a.lower().strip() for a in result_priority]

        matched_count = 0
        total_score = 0

        for inp_author in input_authors_norm:
            best_match_score = 0
            for res_author in result_authors_norm:
                score = author_scorer(inp_author, res_author)
                best_match_score = max(best_match_score, score)

            total_score += best_match_score
            if best_match_score >= threshold:
                matched_count += 1

            if self.debug and best_match_score > 0:
                logger.debug(
                    f"  Author '{inp_author}' best match score: {best_match_score:.1f}",
                )

        # Score combines both match percentage and average similarity
        match_percentage = (matched_count / len(input_authors_norm)) * 100
        avg_similarity = (
            total_score / len(input_authors_norm) if input_authors_norm else 0
        )

        # Weighted combination: 70% based on threshold matches, 30% on average similarity
        final_score = (match_percentage * 0.7) + (avg_similarity * 0.3)

        if self.debug:
            logger.debug(
                f"Author scoring: {matched_count}/{len(input_authors_norm)} above threshold, avg similarity: {avg_similarity:.1f}, final score: {final_score:.1f}",
            )

        return final_score

    def find_best_match_from_items(
        self,
        items: list[dict],
        input_title: str,
        input_authors: Optional[list[str]] = None,
    ) -> tuple[Optional[dict], float]:
        """
        Find the best matching item from CrossRef API results.

        Args:
            items: List of CrossRef API result items
            input_title: The input title to match against
            input_authors: Optional list of input authors

        Returns:
            Tuple of (best_match_item, best_score) or (None, 0) if no suitable match
        """
        title_scorer = self._fuzzy_scorer_map.get(
            self.config.title_fuzzy_method,
            fuzz.ratio,
        )
        title_threshold = self.config.title_fuzzy_threshold
        title_norm = self.normalize_title(input_title)

        best_match = None
        best_score = 0

        for item in items[
            : self.max_candidates
        ]:  # Limit to first max_candidates results
            result_title = item.get("title", [""])[0]
            if not result_title:
                if self.debug:
                    logger.debug("Skipping item with no title")
                continue

            result_title_norm = self.normalize_title(result_title)
            title_score = title_scorer(title_norm, result_title_norm)

            if self.debug:
                logger.debug(
                    f"Title fuzzy score: {title_score:.1f} | Input: '{input_title}' | Result: '{result_title}'",
                )

            if title_score < title_threshold:
                if self.debug:
                    logger.debug(
                        f"Title score {title_score:.1f} below threshold {title_threshold}",
                    )
                continue

            # Get authors from CrossRef result
            result_authors = self.build_author_names(item.get("author", []))

            # Calculate author match score
            author_score = self.calculate_author_match_score(
                input_authors or [],
                result_authors,
            )

            if self.debug:
                logger.debug(
                    f"Author match score: {author_score:.1f} | Input authors: {input_authors} | Result authors: {result_authors}",
                )

            # If we have input authors, require a reasonable author match
            # If no input authors, rely solely on title match
            min_author_score = (
                50.0 if input_authors else 0.0
            )  # Require at least 50% author match if authors provided

            if author_score < min_author_score:
                if self.debug:
                    logger.debug(
                        f"Author score {author_score:.1f} below minimum {min_author_score}",
                    )
                continue

            # Combined score (weighted average: title is more important)
            combined_score = (title_score * 0.7) + (author_score * 0.3)

            if combined_score > best_score:
                best_score = combined_score
                best_match = item

                if self.debug:
                    logger.debug(
                        f"New best match: combined score {combined_score:.1f} (title: {title_score:.1f}, author: {author_score:.1f})",
                    )

        return best_match, best_score

    async def resolve(
        self,
        params: CrossRefDoiResolverInputSchema,
    ) -> Optional[CrossRefDoiResolverOutputSchema]:
        # if doi is already present, return it
        if getattr(params, "doi", None) and params.doi != 'None':
            if self.debug:
                logger.debug(f"DOI already present: {params.doi}")
            return CrossRefDoiResolverOutputSchema(**params.model_dump())

        title = params.title.strip()
        authors = await self.get_authors_from_search_result(params)

        if not title:
            if self.debug:
                logger.debug("CrossRefResolver requires title to resolve DOI")
            return None

        query_params = {
            "query.title": title,
            "rows": self.max_candidates,
        }
        if authors:
            priority_authors = self.select_priority_authors(authors)
            query_params["query.author"] = " ".join(priority_authors)

        doi = None

        try:
            async with self.session or httpx.AsyncClient() as session:
                response = await session.get(
                    self.cross_ref_api_url,
                    params=query_params,
                    headers=self.headers,
                    timeout=self.timeout_seconds,
                )
                if response.status_code != 200:
                    logger.error(f"CrossRef API error: {response.status_code}")
                    return None
                payload = response.json()

                items = payload.get("message", {}).get("items", []) or []
                if not items:
                    if self.debug:
                        logger.debug("No items returned from CrossRef API")
                    return None

                # Find the best match from items
                best_match, best_score = self.find_best_match_from_items(
                    items,
                    title,
                    authors,
                )

                if best_match:
                    doi = best_match.get("DOI")
                    if doi:
                        doi = doi.strip()
                        if self.debug:
                            logger.debug(
                                f"Resolved DOI: {doi} for title: '{title}' with combined score: {best_score:.1f}",
                            )
                    else:
                        if self.debug:
                            logger.debug("Best match found but no DOI available")
                else:
                    if self.debug:
                        logger.debug(
                            f"No suitable match found for title: '{title}' with authors: {authors}",
                        )

                result = CrossRefDoiResolverOutputSchema(**params.model_dump())
                result.doi = doi
                result.resolvers.append(self.__class__.__name__)
                return result

        except Exception as e:
            logger.error(f"Error during CrossRef resolution: {e}")
            return None
