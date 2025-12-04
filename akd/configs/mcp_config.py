"""MCP endpoint configuration.

Simple configuration for MCP (Model Context Protocol) server endpoint.
Uses environment variable MCP_ENDPOINT if set, otherwise defaults to localhost.
"""

import os

DEFAULT_MCP_ENDPOINT = "http://localhost:8080/mcp/cmr/mcp/"


def get_mcp_endpoint() -> str:
    """
    Get MCP endpoint from environment or default.

    The endpoint can be configured via the MCP_ENDPOINT environment variable.
    If not set, defaults to localhost development endpoint.

    Returns:
        str: MCP server endpoint URL

    Example:
        # Use default (localhost)
        endpoint = get_mcp_endpoint()

        # Or set via environment
        export MCP_ENDPOINT=http://production-mcp:8080/mcp/cmr/mcp/
    """
    return os.getenv("MCP_ENDPOINT", DEFAULT_MCP_ENDPOINT)
