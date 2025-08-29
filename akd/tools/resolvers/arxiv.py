from typing import Optional

from pydantic import HttpUrl

from ._base import BaseArticleResolver, ResolverInputSchema, ResolverOutputSchema


class ArxivResolver(BaseArticleResolver):
    """
    Resolver for arXiv papers that converts arXiv URLs to direct PDF links
    and resolves DOI when not provided using the standard arXiv DOI format.
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
        """Convert arXiv URL to PDF URL and resolve DOI when not provided"""
        url = str(params.url)
        try:
            self.validate_url(url)
            paper_id = url.split("/")[-1]
            pdf_url = HttpUrl(f"https://arxiv.org/pdf/{paper_id}.pdf")

            # Resolve DOI if not provided using standard arXiv DOI format
            doi = params.doi or f"10.48550/arXiv.{paper_id}"

            result = ResolverOutputSchema(**params.model_dump())
            if not result.extra:
                result.extra = {}
            if not hasattr(result.extra, "original_url"):
                result.extra["original_url"] = []

            result.extra["original_url"].append(params.url)
            result.url = pdf_url
            result.doi = doi
            result.resolvers.append(self.__class__.__name__)
            return result
        except RuntimeError:
            # Not a valid arxiv URL
            return None
