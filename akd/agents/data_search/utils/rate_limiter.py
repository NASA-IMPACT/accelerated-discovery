"""
Rate limiting utilities for OpenAI API calls.

Prevents triggering OpenAI's concurrent request throttling by limiting
the number of simultaneous API calls.
"""

import asyncio
from typing import Optional


class OpenAIRateLimiter:
    """
    Rate limiter to control concurrent OpenAI API requests.

    OpenAI throttles when too many concurrent requests come from the same API key.
    This semaphore-based limiter prevents triggering that throttling.
    """

    def __init__(self, max_concurrent_requests: int = 2):
        """
        Initialize rate limiter.

        Args:
            max_concurrent_requests: Maximum number of concurrent API calls allowed.
                Default is 2 based on observed throttling at 3+ concurrent requests.
        """
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.max_concurrent = max_concurrent_requests

    async def __aenter__(self):
        """Acquire semaphore (wait if at limit)."""
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release semaphore."""
        self.semaphore.release()

    def available_slots(self) -> int:
        """Get number of available request slots."""
        return self.semaphore._value


# Global rate limiter instance
# Can be configured by setting max_concurrent_requests before first use
_global_limiter: Optional[OpenAIRateLimiter] = None


def get_rate_limiter(max_concurrent_requests: int = 2) -> OpenAIRateLimiter:
    """
    Get or create the global rate limiter instance.

    Args:
        max_concurrent_requests: Maximum concurrent requests (only used on first call)

    Returns:
        Shared rate limiter instance
    """
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = OpenAIRateLimiter(max_concurrent_requests)
    return _global_limiter


def reset_rate_limiter():
    """Reset the global rate limiter (mainly for testing)."""
    global _global_limiter
    _global_limiter = None
