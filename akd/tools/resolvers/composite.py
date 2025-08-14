from loguru import logger
from pydantic import HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema


class ResearchArticleResolver(BaseArticleResolver):
    """
    Composite resolver that tries multiple resolvers in sequence.
    This allows for a waterfall approach where if one resolver fails,
    the next one is tried until a successful resolution is found.
    """

    def __init__(self, *resolvers: BaseArticleResolver, debug: bool = False) -> None:
        super().__init__(debug=debug)
        self.resolvers = resolvers

    def validate_url(self, url: HttpUrl | str) -> bool:
        """Composite resolver accepts any URL that at least one sub-resolver accepts"""
        return True

    async def resolve(self, params: ResolverInputSchema) -> ResolverOutputSchema | None:
        """This method is not used in composite resolver - see _arun instead"""
        result = ResolverOutputSchema(**params.model_dump())
        result.resolvers.append(self.__class__.__name__)
        return result
    
    async def _arun(
        self,
        params: ResolverInputSchema,
        **kwargs,
        ) -> ResolverOutputSchema:
        """
        Run resolvers sequentially.
        Stop only when we have BOTH:
          1) a transformed URL (different from params.url), and
          2) a DOI.
        Otherwise keep going, merging partial improvements.
        """
        output = ResolverOutputSchema(**params.model_dump())

        orig_url_str = str(params.url) if getattr(params, "url", None) else None

        def has_transformed_url(url) -> bool:
            if not orig_url_str and url:
                return True
            return bool(orig_url_str and url and str(url) != orig_url_str)

        def has_doi(doi) -> bool:
            return bool(doi)

        for resolver in self.resolvers:
            resolver_name = resolver.__class__.__name__
            try:
                if self.debug:
                    logger.debug(f"Trying resolver={resolver_name} for url={params.url}")

                result = await resolver.arun(params)
                if not result:
                    continue

                output.resolvers.extend(result.resolvers)

                if result.url and (output.url is None or has_transformed_url(result.url)):
                    output.url = result.url
                    
                if getattr(result, "doi", None) and not getattr(output, "doi", None):
                    output.doi = result.doi

                if not output.authors and result.authors:
                    output.authors = result.authors

                # Stop only when BOTH conditions are satisfied
                if has_transformed_url(getattr(output, "url", None)) and has_doi(getattr(output, "doi", None)):
                    if self.debug:
                        logger.debug(
                            f"Stopping after {resolver_name}: "
                            f"transformed URL={output.url}, DOI={output.doi}"
                        )
                    break

                # Otherwise, continue to try next resolver

            except Exception as e:
                if self.debug:
                    logger.error(f"Error using resolver={resolver_name}: {e}")
                continue

        return output
