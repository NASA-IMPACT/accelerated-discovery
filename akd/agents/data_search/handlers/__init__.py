"""
Repository-specific handlers for data search operations.

This package contains handlers for different NASA data repositories
and external data sources. Each handler implements the repository-specific
logic for processing scientific decompositions into data results.
"""

from enum import Enum

from akd.agents.data_search.components.repository_router import NASARepositoryEnum

from ._base import BaseHandler
from .cmr import CMRHandler, CMRHandlerConfig
from .pds4 import PDS4Handler, PDS4HandlerConfig


class HandlerStatus(str, Enum):
    """Status of handler implementation."""

    IMPLEMENTED = "implemented"  # Fully functional handler
    STUB = "stub"  # Placeholder that returns "not implemented"


# Handler registry - maps repository enum to handler class
HANDLERS: dict = {
    NASARepositoryEnum.CMR: CMRHandler,
    NASARepositoryEnum.PDS4: PDS4Handler,
}

# Handler status tracking - determines which handlers are available for routing
HANDLER_STATUS: dict = {
    NASARepositoryEnum.CMR: HandlerStatus.IMPLEMENTED,
    NASARepositoryEnum.PDS4: HandlerStatus.IMPLEMENTED,
}


def get_implemented_repositories() -> list[NASARepositoryEnum]:
    """
    Get list of NASA repositories with implemented (non-stub) handlers.

    This is used by the RepositoryRouterComponent to dynamically determine
    which NASA repositories are available for routing.

    Returns:
        List of NASA repository enum values with IMPLEMENTED status
    """
    return [
        repo
        for repo, status in HANDLER_STATUS.items()
        if status == HandlerStatus.IMPLEMENTED
    ]


def get_all_repositories() -> list[NASARepositoryEnum]:
    """
    Get list of all registered NASA repositories (both implemented and stubs).

    Returns:
        List of all NASA repository enum values with handlers
    """
    return list(HANDLERS.keys())


__all__ = [
    "BaseHandler",
    "HANDLERS",
    "HANDLER_STATUS",
    "HandlerStatus",
    "get_implemented_repositories",
    "get_all_repositories",
    "CMRHandler",
    "CMRHandlerConfig",
    "PDS4Handler",
    "PDS4HandlerConfig",
]
