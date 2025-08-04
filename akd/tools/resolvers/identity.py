from typing import Optional, Union

from pydantic import HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema


class IdentityResolver(BaseArticleResolver):
    """
    Identity resolver that returns the URL as-is without any transformation.
    This is useful as a fallback resolver that always succeeds.
    """

    def validate_url(self, url: Union[str, HttpUrl]) -> bool:
        """Identity resolver accepts any URL"""
        return True

    async def resolve(self, params: ResolverInputSchema) -> Optional[HttpUrl]:
        """Return the primary URL as-is"""
        return HttpUrl(str(params.url))
