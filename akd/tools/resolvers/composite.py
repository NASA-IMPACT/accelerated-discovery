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
        Stop only when BOTH are satisfied:
          1) URL is transformed (different from params.url, or any URL if original was None)
          2) DOI is present

        Once the URL is transformed, do NOT replace it again.
        """
        output = ResolverOutputSchema(**params.model_dump())
        orig_url_str = str(params.url) if getattr(params, "url", None) else None

        def has_transformed_url(url) -> bool:
            if not url:
                return False
            if not orig_url_str:
                return True  # any URL counts if we started with None
            return str(url) != orig_url_str

        
        # Track satisfaction across resolvers
        url_ok = False
        doi_ok = False 

        for resolver in self.resolvers:
            if url_ok and doi_ok:
                break

            resolver_name = resolver.__class__.__name__
            try:
                if self.debug:
                    logger.debug(f"Trying resolver={resolver_name} for url={params.url}")

                result = await resolver.arun(params)

                if not result:
                    continue

                # TODO:: this will call most of the resolvers: doi until resolved by crossref, so we should do something about this 
                if not url_ok and getattr(result, "url", None):
                    if output.url is None or has_transformed_url(result.url):
                        output.url = result.url
                        url_ok = has_transformed_url(output.url)
                        output.resolvers.extend(result.resolvers)

                if not doi_ok and getattr(result, "doi", None):
                    if not getattr(output, "doi", None):
                        output.doi = result.doi
                        doi_ok = True
                        output.resolvers.extend(result.resolvers)

                if not getattr(output, "authors", None) and getattr(result, "authors", None):
                    output.authors = result.authors

                if url_ok and doi_ok:
                    if self.debug:
                        logger.debug(
                            f"Stopping after {resolver_name}: "
                            f"Transformed URL={output.url}, DOI={output.doi}"
                        )
                    break

            except Exception as e:
                if self.debug:
                    logger.error(f"Error using resolver={resolver_name}: {e}")
                continue

        return output
