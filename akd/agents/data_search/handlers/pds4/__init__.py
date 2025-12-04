"""PDS4 handler stub - not yet implemented."""

from pydantic import BaseModel

from .._base import BaseHandler


class PDS4HandlerConfig(BaseModel):
    """Configuration for PDS4 handler (stub)."""

    pass


class PDS4Handler(BaseHandler):
    """
    Stub handler for PDS4 repository.

    This is a placeholder that returns "not implemented" responses.
    """

    def __init__(self, config: PDS4HandlerConfig | None = None, debug: bool = False):
        """Initialize PDS4 handler stub."""
        self.config = config or PDS4HandlerConfig()
        self.debug = debug

    async def process_decomposition(
        self,
        original_query: str,
        topic: dict,
        decomposition: dict,
    ) -> dict:
        """
        Process a single scientific decomposition (stub).

        Returns:
            Stub response indicating PDS4 is not yet implemented
        """
        return {
            "status": "not_implemented",
            "message": "PDS4 handler is not yet implemented",
            "data_results": [],
            "total_results_from_cmr": 0,
            "total_results_after_filtering": 0,
        }


__all__ = [
    "PDS4Handler",
    "PDS4HandlerConfig",
]
