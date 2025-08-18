import aiohttp
from loguru import logger
from typing import Optional, Union
from pydantic import HttpUrl
from rapidfuzz import fuzz


from ._base import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema, ArticleResolverConfig
from pydantic import Field

from akd.structures import SearchResultItem


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


class CrossRefDoiResolverOutputSchema(ResolverOutputSchema):
    """
    Output schema for CrossRef DOI resolver.
    Inherits from ResolverOutputSchema to ensure compatibility with search results.
    """

    # override to make url optional 
    url: Optional[HttpUrl] = Field(
        None,
        description="URL of the article",
    )


class CrossRefDoiResolverConfig(ArticleResolverConfig):
    """Configuration for the CrossRef DOI resolver."""

    def __init__(self, **kwargs):
        kwargs["validate_resolved_url"] = False
        super().__init__(**kwargs)

    validate_resolved_url: bool = Field(
        default=False,
        description="Whether to validate that the resolved URL is accessible (forced False in CrossRef config)",
    )

    cross_ref_api_url: str = Field(
        default="https://api.crossref.org/works",
        description="Base URL for CrossRef API",
    )

    timeout_seconds: int = Field(
        default=30,
        description="Timeout for API requests in seconds",
    )

    fuzzy_threshold: int = Field(
        default=70,
        description="Fuzzy matching threshold for title matching",
    )

    fuzzy_score_method: str = Field(
        default="token_set",
        description="Fuzzy score method: ratio, token_sort, token_set",
    )

    user_agent: Optional[str] = Field(
        default="CrossRefResolver/1.0",
        description="User agent for HTTP requests",
    )

    


class CrossRefDoiResolver(BaseArticleResolver):
    """
    Resolver that queries the CrossRef API using title and authors
    and resolves to a DOI and URL if a match is found.
    """

    config_schema = CrossRefDoiResolverConfig

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

    def get_fuzzy_scorer(self):
        method = self.config.fuzzy_score_method
        return {
            "token_set": fuzz.token_set_ratio,
            "token_sort": fuzz.token_sort_ratio,
            "ratio": fuzz.ratio,
        }.get(method, fuzz.token_set_ratio)
    

    async def get_authors_from_search_result(self, result: SearchResultItem) -> list[str]:
        """
        Extract authors from search result. Utility function to be used if needed.
        """
        return result.extra.get("authors") if result.extra and result.extra.get("authors") else result.authors if result.authors else []
    

    async def resolve(self, 
                      params: CrossRefDoiResolverInputSchema
                      ) -> Optional[CrossRefDoiResolverOutputSchema]:
        
        # if doi is already present, return it
        if getattr(params, "doi", None):
            if self.debug:
                logger.debug(f"DOI already present: {params.doi}")
            return CrossRefDoiResolverOutputSchema(**params.model_dump())
        
        title = params.title.strip()
        authors = await self.get_authors_from_search_result(params)
    

        if not title:
            if self.debug:
                logger.debug("CrossRefResolver requires title to resolve DOI")
            return None

        scorer     = self.get_fuzzy_scorer()
        threshold  = self.config.fuzzy_threshold
        title_norm = title.lower().strip()
        input_authors_norm = [a.lower().strip() for a in (authors or [])][:3]  # cap at 3

        query_params = {
            "query.title": title,
            "rows": 10,
        }
        if input_authors_norm:
            query_params["query.author"] = " ".join(input_authors_norm)

        doi = None

        try:
            async with (self.session or aiohttp.ClientSession()) as session:
                async with session.get(
                    self.config.cross_ref_api_url,
                    params=query_params,
                    headers={"User-Agent": self.config.user_agent},
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"CrossRef API error: {resp.status}")
                        return None
                    payload = await resp.json()

                items = payload.get("message", {}).get("items", []) or []
                if not items:
                    return None

                # --- matching ---
                for item in items[:10]:
                    result_title = (item.get("title") or [""])[0].lower().strip()
                    if scorer(title_norm, result_title) < threshold:
                        continue

                    # authors from CrossRef
                    result_authors = self.build_author_names(item.get("author", []))
                    result_authors_norm = [a.lower().strip() for a in result_authors][:3]

                    # if no input authors, title match is enough
                    authors_ok = True
                    if input_authors_norm:
                        # each input author must fuzzily match at least one result author
                        authors_ok = all(
                            any(scorer(inp, cand) >= threshold for cand in result_authors_norm)
                            for inp in input_authors_norm
                        )

                    if not authors_ok:
                        continue

                    doi = (item.get("DOI") or None).strip()
                    if doi:
                        if self.debug:
                            logger.debug(f"Resolved DOI: {doi} for title: {title}")
                        break

                if self.debug and not doi:
                    logger.debug(f"No matching DOI found for title: {title} with authors: {authors}")

                result = CrossRefDoiResolverOutputSchema(**params.model_dump())
                result.doi = doi
                result.resolvers.append(self.__class__.__name__)
                return result
            
        except Exception as e:
            logger.error(f"Error during CrossRef resolution: {e}")
            return None
        
        
