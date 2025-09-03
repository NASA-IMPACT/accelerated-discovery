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
        result = resolver_input = ResolverOutputSchema(**params.model_dump())

        is_url_resolved = False
        is_doi_resolved = bool(getattr(params, "doi", None) and params.doi != 'None') 

        for resolver in self.resolvers:
            resolver_name = resolver.__class__.__name__
            try:
                if self.debug:
                    logger.debug(f"Trying resolver={resolver_name} for url={resolver_input.url}")
                
                
                result = await resolver.arun(resolver_input)

                if not result:
                    continue


                # TODO:: this will call most of the resolvers: doi until resolved by crossref, so we should do something about this

                if "is_url_resolved" in result.extra:
                    is_url_resolved = result.extra["is_url_resolved"]

                is_doi_resolved = bool(getattr(result, "doi", None) and result.doi != 'None')

                resolver_input = result

                if is_url_resolved and is_doi_resolved:
                    if self.debug:
                        logger.debug(
                            f"Stopping after {resolver_name}: "
                            f"Transformed URL={params.url} to {result.url}, "
                            f"Transformed DOI={params.doi} to {result.doi}",
                        )
                    break

            except Exception as e:
                if self.debug:
                    logger.error(f"Error using resolver={resolver_name}: {e}")
                continue


        return result 
