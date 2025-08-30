from typing import Optional, Union

from pydantic import HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema


class IdentityResolver(BaseArticleResolver):
    """
    Identity resolver that returns the URL as-is without any transformation.
    This is useful as a fallback resolver that always succeeds.
    """

    def validate_url(self, url: Union[str, HttpUrl]) -> bool:
        """Identity resolver accepts any URL"""
        return True

    async def resolve(
        self,
        params: ResolverInputSchema,
    ) -> Optional[ResolverOutputSchema]:
        """Return the primary URL as-is with preserved metadata"""
        result = ResolverOutputSchema(**params.model_dump())
        
        if not result.extra:
            result.extra = {}
        result.extra["is_url_resolved"] = True
        result.resolvers.append(self.__class__.__name__)
        return result
