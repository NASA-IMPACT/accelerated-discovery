import os
from enum import Enum, auto
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field


class Environment(Enum):
    """Application environment"""

    LOCAL = auto()
    DEV = auto()
    STAGING = auto()
    PRODUCTION = auto()


def get_environment() -> Environment:
    """Get current environment from ENV variable or default to LOCAL"""
    env_str = os.getenv("APP_ENV", "local").upper()
    try:
        return Environment[env_str]
    except KeyError:
        return Environment.LOCAL


class ModelProvider(str, Enum):
    """Enum for supported model providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VLLM = "vllm"


class ApiKey(BaseModel):
    openai: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ollama: Optional[str] = os.getenv("OLLAMA_API_KEY")
    vllm: Optional[str] = os.getenv("VLLM_API_KEY")


class ModelConfig(BaseModel):
    """Decoupled model configuration"""

    provider: ModelProvider = Field(default=ModelProvider.OPENAI)
    model_name: str = Field(default=os.getenv("MODEL_NAME_GLOBAL", "gpt-4o-mini"))
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=120_000_000)
    api_keys: ApiKey = Field(default=ApiKey())
    default_no_answer: str = Field(default="Answer not found")


class BaseConfig(BaseModel):
    debug: bool = False
    model_config_: ModelConfig = ModelConfig()


class LocalConfig(BaseConfig):
    pass


class DevConfig(BaseConfig):
    pass


class StagingConfig(BaseConfig):
    pass


class ProdConfig(BaseConfig):
    pass


def get_config() -> BaseConfig:
    """Retrieve configuration based on the environment."""
    env = os.getenv("APP_ENV", "local").upper().strip()
    logger.info(f"Loading configuration for environment: {env}")
    if env == "LOCAL":
        return LocalConfig()
    elif env == "DEV":
        return DevConfig()
    elif env == "STAGING":
        return StagingConfig()
    elif env == "PRODUCTION":
        return ProdConfig()
    else:
        raise ValueError(f"Invalid environment: {env}")


CONFIG = get_config()
