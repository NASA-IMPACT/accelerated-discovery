"""
Centralized configuration for data search components.
"""

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class Environment(str, Enum):
    """Supported environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class WebSocketConfig(BaseModel):
    """WebSocket connection configuration."""

    host: str = Field(default="localhost", description="WebSocket host")
    port: int = Field(default=8003, description="WebSocket port")
    max_reconnect_attempts: int = Field(
        default=5,
        description="Maximum reconnection attempts",
    )
    base_reconnect_delay: int = Field(
        default=1000,
        description="Base reconnection delay in milliseconds",
    )
    connection_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds",
    )

    @property
    def url(self) -> str:
        """Get the WebSocket URL."""
        return f"ws://{self.host}:{self.port}"

    @property
    def http_base_url(self) -> str:
        """Get the HTTP base URL."""
        return f"http://{self.host}:{self.port}"


class MCPConfig(BaseModel):
    """MCP server configuration."""

    endpoint: HttpUrl = Field(
        default="http://localhost:8080/mcp/cmr/mcp/",
        description="MCP server endpoint URL",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_delay: float = Field(
        default=1.0,
        description="Base delay between retries in seconds",
    )


class LLMConfig(BaseModel):
    """LLM service configuration."""

    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    model_name: str = Field(default="gpt-4", description="Model name to use")
    temperature: float = Field(default=0.1, description="Model temperature")
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for LLM calls",
    )
    base_delay: float = Field(
        default=1.0,
        description="Base delay for exponential backoff",
    )


class SearchConfig(BaseModel):
    """Search behavior configuration."""

    max_collections_to_search: int = Field(
        default=10,
        description="Maximum collections to search for granules",
    )
    max_granules_per_collection: int = Field(
        default=100,
        description="Maximum granules per collection",
    )
    collection_search_page_size: int = Field(
        default=20,
        description="Page size for collection searches",
    )
    granule_search_page_size: int = Field(
        default=50,
        description="Page size for granule searches",
    )
    enable_parallel_search: bool = Field(
        default=True,
        description="Enable parallel searches",
    )
    min_collection_relevance_score: float = Field(
        default=0.3,
        description="Minimum collection relevance score",
    )


class DataSearchConfig(BaseModel):
    """Centralized configuration for the entire data search system."""

    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Current environment",
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # Component configurations
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    class Config:
        """Pydantic configuration."""

        env_prefix = "DATA_SEARCH_"
        env_nested_delimiter = "__"

    @classmethod
    def from_environment(cls, env: Optional[Environment] = None) -> "DataSearchConfig":
        """
        Create configuration from environment variables and environment-specific defaults.

        Args:
            env: Target environment. If None, will try to detect from ENV var.

        Returns:
            Configured DataSearchConfig instance
        """
        if env is None:
            env_str = os.getenv("ENV", "development").lower()
            env = (
                Environment(env_str)
                if env_str in Environment
                else Environment.DEVELOPMENT
            )

        # Base configuration
        config_data = {
            "environment": env,
            "debug": env == Environment.DEVELOPMENT,
        }

        # Environment-specific overrides
        if env == Environment.DEVELOPMENT:
            config_data.update(
                {
                    "websocket": {
                        "host": "localhost",
                        "port": 8003,
                        "max_reconnect_attempts": 5,
                    },
                    "mcp": {
                        "endpoint": "http://localhost:8080/mcp/cmr/mcp/",
                        "timeout_seconds": 30.0,
                    },
                    "search": {
                        "max_collections_to_search": 5,  # Smaller for dev
                        "enable_parallel_search": True,
                    },
                },
            )
        elif env == Environment.STAGING:
            config_data.update(
                {
                    "websocket": {
                        "host": os.getenv("WEBSOCKET_HOST", "staging.example.com"),
                        "port": int(os.getenv("WEBSOCKET_PORT", "8003")),
                        "max_reconnect_attempts": 3,
                    },
                    "mcp": {
                        "endpoint": os.getenv(
                            "MCP_ENDPOINT",
                            "http://staging-mcp.example.com/mcp/",
                        ),
                        "timeout_seconds": 45.0,
                    },
                    "search": {
                        "max_collections_to_search": 10,
                        "enable_parallel_search": True,
                    },
                },
            )
        elif env == Environment.PRODUCTION:
            config_data.update(
                {
                    "debug": False,
                    "websocket": {
                        "host": os.getenv("WEBSOCKET_HOST", "api.example.com"),
                        "port": int(os.getenv("WEBSOCKET_PORT", "443")),
                        "max_reconnect_attempts": 10,
                    },
                    "mcp": {
                        "endpoint": os.getenv(
                            "MCP_ENDPOINT",
                            "https://mcp.example.com/mcp/",
                        ),
                        "timeout_seconds": 60.0,
                        "max_retries": 5,
                    },
                    "search": {
                        "max_collections_to_search": 15,
                        "enable_parallel_search": True,
                    },
                },
            )

        # Apply environment variable overrides
        config_data["llm"] = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model_name": os.getenv("LLM_MODEL_NAME", "gpt-4"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
        }

        return cls(**config_data)

    def get_frontend_config(self) -> dict:
        """
        Get configuration values safe to send to frontend.

        Returns:
            Dictionary with frontend-safe configuration
        """
        return {
            "environment": self.environment.value,
            "websocket": {
                "url": self.websocket.url,
                "maxReconnectAttempts": self.websocket.max_reconnect_attempts,
                "baseReconnectDelay": self.websocket.base_reconnect_delay,
                "connectionTimeout": self.websocket.connection_timeout,
            },
            "search": {
                "maxCollections": self.search.max_collections_to_search,
                "enableParallelSearch": self.search.enable_parallel_search,
            },
        }


# Global configuration instance
config = DataSearchConfig.from_environment()


def get_config() -> DataSearchConfig:
    """Get the global configuration instance."""
    return config


def reload_config(env: Optional[Environment] = None) -> DataSearchConfig:
    """Reload configuration from environment."""
    global config
    config = DataSearchConfig.from_environment(env)
    return config
