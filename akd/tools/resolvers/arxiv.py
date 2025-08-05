from typing import Optional

from pydantic import HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema


class ArxivResolver(BaseArticleResolver):
    """
    Resolver for arXiv papers that converts arXiv URLs to direct PDF links.
    """

    def validate_url(self, url: HttpUrl | str):
        """Check if this URL is from arXiv"""
        if "arxiv.org" not in str(url):
            raise RuntimeError("Not a valid arxiv url")
        return True

    async def resolve(
        self,
        params: ResolverInputSchema,
    ) -> Optional[ResolverOutputSchema]:
        """Convert arXiv URL to PDF URL"""
        url = str(params.url)
        try:
            self.validate_url(url)
            paper_id = url.split("/")[-1]
            pdf_url = HttpUrl(f"https://arxiv.org/pdf/{paper_id}.pdf")

            return ResolverOutputSchema(
                url=pdf_url,
                title=params.title,
                query=params.query,
                resolver=self.__class__.__name__,
            )
        except RuntimeError:
            # Not a valid arxiv URL
            return None
