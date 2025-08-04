from typing import Optional

from pydantic import HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema


class ArxivResolver(BaseArticleResolver):
    """
    Resolver for arXiv papers that converts arXiv URLs to direct PDF links.
    """

    def validate_url(self, url: HttpUrl | str):
        """Check if this URL is from arXiv"""
        if "arxiv.org" not in str(url):
            raise RuntimeError("Not a valid arxiv url")
        return True

    async def resolve(self, params: ResolverInputSchema) -> Optional[HttpUrl]:
        """Convert arXiv URL to PDF URL"""
        url = str(params.url)
        try:
            self.validate_url(url)
            paper_id = url.split("/")[-1]
            return HttpUrl(f"https://arxiv.org/pdf/{paper_id}.pdf")
        except RuntimeError:
            # Not a valid arxiv URL
            return None
