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

    async def resolve(self, url: str | HttpUrl) -> str:
        """This method is not used in composite resolver - see _arun instead"""
        return str(url)

    async def _arun(
        self,
        params: ResolverInputSchema,
        **kwargs,
    ) -> ResolverOutputSchema:
        """
        Try each resolver in sequence until one succeeds.
        Returns the result from the first successful resolver.
        """
        output = ResolverOutputSchema(
            url=params.url,
            resolver=self.__class__.__name__,
        )

        for resolver in self.resolvers:
            resolver_name = resolver.__class__.__name__
            try:
                if self.debug:
                    logger.debug(
                        f"Trying resolver={resolver_name} for url={params.url}",
                    )

                result = await resolver.arun(params)

                if not result.url:
                    continue

                # If resolver transformed the URL, use it and stop
                if str(result.url) != str(params.url):
                    if self.debug:
                        logger.debug(
                            f"Resolver={resolver_name} transformed URL: {params.url} -> {result.url}",
                        )
                    output = result
                    break
                else:
                    # Resolver validated original URL - keep as fallback
                    if self.debug:
                        logger.debug(
                            f"Resolver={resolver_name} returned same URL: {params.url}",
                        )
                    output = result

            except Exception as e:
                if self.debug:
                    logger.error(f"Error using resolver={resolver_name}: {str(e)}")
                continue

        return output
