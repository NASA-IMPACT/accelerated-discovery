from __future__ import annotations

from loguru import logger
from pydantic import HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema
from .ads import ADSResolver
from .arxiv import ArxivResolver
from .crossref_doi import CrossRefDoiResolver
from .specialized import DOIResolver, PDFUrlResolver
from .unpaywall import UnpaywallResolver


class CompositeResolver(BaseArticleResolver):
    """
    Composite resolver that tries multiple resolvers in sequence.
    This allows for a waterfall approach where if one resolver fails,
    the next one is tried until a successful resolution is found.
    """

    def __init__(
        self,
        *resolvers: BaseArticleResolver,
        debug: bool = False,
    ) -> None:
        """
        Initialize CompositeResolver with a chain of resolvers.

        If no resolvers are provided, uses a default chain optimized for scientific literature:
        1. CrossRef - finds DOI from title when missing
        2. DOI normalization - cleans up DOI format
        3. ArXiv - creates DOI for arXiv papers, converts to PDF
        4. ADS - extracts DOI from ADS pages
        5. DOI normalization (again) - ensures final DOI is canonical
        6. Unpaywall - uses the DOI to find OA versions
        7. PDF fallback - passes through existing PDF URLs

        Args:
            *resolvers: Variable number of resolver instances to chain together.
                       If empty, uses default scientific resolver chain.
            debug: Enable debug logging
        """
        super().__init__(debug=debug)

        # Use default resolver chain if none provided
        self.resolvers = resolvers or (
            CrossRefDoiResolver(debug=debug),  # 1. DOI from title
            DOIResolver(debug=debug),  # 2. Normalize DOI
            ArxivResolver(debug=debug),  # 3. ArXiv DOI + PDF
            ADSResolver(debug=debug),  # 4. ADS DOI/PDF
            DOIResolver(debug=debug),  # 5. Re-normalize DOI
            UnpaywallResolver(debug=debug),  # 6. OA versions
            PDFUrlResolver(debug=debug),  # 7. PDF fallback
        )

    def validate_url(self, url: HttpUrl | str) -> bool:
        """Composite resolver accepts any URL that at least one sub-resolver accepts"""
        return True

    def __str__(self) -> str:
        """String representation showing resolver chain."""
        resolver_names = [r.__class__.__name__ for r in self.resolvers]
        return f"CompositeResolver({len(self.resolvers)} resolvers: {' → '.join(resolver_names)})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return self.__str__()

    async def resolve(self, params: ResolverInputSchema) -> ResolverOutputSchema | None:
        """This method is not used in composite resolver - see _arun instead"""
        result = ResolverOutputSchema(**params.model_dump())
        result.resolvers.append(self.__class__.__name__)
        return result

    async def _arun(
        self,
        params: ResolverInputSchema,
        **kwargs,  # noqa: ARG002
    ) -> ResolverOutputSchema:
        """
        Run all resolvers and merge their contributions to create complete, consistent metadata.

        Strategy:
        1. All resolvers run (no early stopping) to maximize metadata collection
        2. For each field (url, doi, title, authors, etc.): fill if missing, don't overwrite
        3. Track URLs by quality type (PDF, OA, DOI) for smart selection
        4. Select best URL for crawling based on content accessibility
        5. Store alternative URLs and track data sources

        Goal: Produce fully enriched SearchResultItem with best available data from all sources.
        """
        result = ResolverOutputSchema(**params.model_dump())

        # Track URLs by quality/type for smart selection
        urls = {
            "pdf": None,  # Best: Direct PDF access
            "oa_host": None,  # Good: OA HTML with full text
            "doi": None,  # Reliable: DOI redirect (always resolves)
            "original": params.url,  # Fallback: Original search result
        }

        # Track which resolver provided each field (for debugging/transparency)
        field_sources = {}

        for resolver in self.resolvers:
            resolver_name = resolver.__class__.__name__
            try:
                if self.debug:
                    logger.debug(f"Running resolver={resolver_name}")

                # Pass cumulative result so resolvers can build on each other
                enriched = await resolver.arun(result)

                if not enriched:
                    if self.debug:
                        logger.debug(f"  {resolver_name} returned None, skipping")
                    continue

                # COLLECT URLs BY TYPE/QUALITY
                url_source = enriched.extra.get("url_source")
                url_type = enriched.extra.get("url_type")

                if url_source == "UnpaywallResolver":
                    if url_type == "pdf":
                        urls["pdf"] = enriched.url
                    elif url_type == "oa_host":
                        urls["oa_host"] = enriched.url

                elif url_source == "DOIResolver":
                    urls["doi"] = enriched.url

                elif url_source == "ArxivResolver" and enriched.pdf_url:
                    urls["pdf"] = enriched.pdf_url

                elif url_source == "PDFUrlResolver" and url_type == "pdf":
                    urls["pdf"] = enriched.url

                # MERGE METADATA FIELDS (fill missing, preserve existing)
                metadata_fields = ["doi", "title", "authors", "published_date", "category", "tags"]

                for field in metadata_fields:
                    current_value = getattr(result, field)
                    new_value = getattr(enriched, field)

                    # Fill if missing
                    if not current_value and new_value:
                        setattr(result, field, new_value)
                        field_sources[field] = resolver_name
                        if self.debug:
                            logger.debug(f"  Filled '{field}' from {resolver_name}")

                # Handle pdf_url specially (keep best one)
                if enriched.pdf_url and not result.pdf_url:
                    result.pdf_url = enriched.pdf_url
                    field_sources["pdf_url"] = resolver_name

                # Merge extra metadata (accumulate, don't replace)
                for key, value in enriched.extra.items():
                    # Always update these tracking fields
                    if key in ["is_url_resolved", "url_source", "url_type"]:
                        result.extra[key] = value
                    # For other fields, only add if not present (preserve earlier data)
                    elif key not in result.extra:
                        result.extra[key] = value

                # Track all resolvers that contributed
                for res_name in enriched.resolvers:
                    if res_name not in result.resolvers:
                        result.resolvers.append(res_name)

                # Update cumulative result for next resolver
                result = enriched

            except Exception as e:
                if self.debug:
                    logger.error(f"Error in resolver={resolver_name}: {e}")
                continue

        # SMART URL SELECTION FOR CRAWLING
        # Priority: PDF > OA Host > DOI > Original
        # Rationale: PDF has full content, OA has full HTML, DOI always resolves, Original uncertain
        selected_url = None
        selection_reason = None

        if urls["pdf"]:
            selected_url = urls["pdf"]
            selection_reason = "pdf_direct_access"
            result.pdf_url = urls["pdf"]  # Ensure pdf_url field is set
        elif urls["oa_host"]:
            selected_url = urls["oa_host"]
            selection_reason = "oa_host_full_text"
        elif urls["doi"]:
            selected_url = urls["doi"]
            selection_reason = "doi_fallback_reliable"
        else:
            selected_url = urls["original"]
            selection_reason = "original_search_result"

        # Set the selected URL as primary
        result.url = selected_url
        result.extra["url_selection_reason"] = selection_reason

        # Store alternative URLs for downstream fallback attempts
        alternative_urls = {k: str(v) for k, v in urls.items() if v and str(v) != str(selected_url)}
        if alternative_urls:
            result.extra["alternative_urls"] = alternative_urls

        # Store field sources for transparency
        result.extra["field_sources"] = field_sources

        if self.debug:
            logger.debug(
                f"Composite resolver complete: "
                f"url={selected_url} (reason: {selection_reason}), "
                f"doi={result.doi}, "
                f"title={'✓' if result.title else '✗'}, "
                f"authors={len(result.authors or [])}, "
                f"pdf_url={'✓' if result.pdf_url else '✗'}",
            )
            if alternative_urls:
                logger.debug(f"Alternative URLs: {list(alternative_urls.keys())}")

        return result
