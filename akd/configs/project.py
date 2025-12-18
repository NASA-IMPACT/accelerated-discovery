import os
from enum import Enum, auto
from functools import lru_cache
from typing import Optional

from pydantic import AnyUrl, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(Enum):
    """Application environment"""

    LOCAL = auto()
    DEV = auto()
    STAGING = auto()
    PRODUCTION = auto()


class ModelProvider(str, Enum):
    """Enum for supported model providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VLLM = "vllm"


class ApiKeys(BaseModel):
    openai: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ollama: Optional[str] = os.getenv("OLLAMA_API_KEY")
    vllm: Optional[str] = os.getenv("VLLM_API_KEY")


class ModelConfigSettings(BaseSettings):
    provider: ModelProvider = ModelProvider.OPENAI
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 120_000
    api_keys: ApiKeys = ApiKeys()
    base_url: AnyUrl | None = Field(
        default=AnyUrl(os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")),
        description="Base URL for the model provider API",
    )
    default_no_answer: str = "Answer not found"


class GuardrailSettings(BaseModel):
    """Project-level guardrail defaults for the @guardrail decorator.

    These settings provide defaults when not explicitly specified in the decorator.
    Can be overridden via environment variables using GUARDRAILS__ prefix:
        GUARDRAILS__FAIL_ON_INPUT_RISK=true
        GUARDRAILS__FAIL_ON_OUTPUT_RISK=false
    """

    input_fields: list[str] = Field(
        default=["query", "content", "text", "message", "user_input"],
        description="Default input fields to extract text from for guardrail checks",
    )
    output_fields: list[str] = Field(
        default=["response", "answer", "result", "content", "text"],
        description="Default output fields to extract text from for guardrail checks",
    )
    fail_on_input_risk: bool = Field(
        default=False,
        description="Raise InputGuardrailTriggered exception when input risk detected",
    )
    fail_on_output_risk: bool = Field(
        default=False,
        description="Raise OutputGuardrailTriggered exception when output risk detected",
    )
    log_warnings: bool = Field(
        default=True,
        description="Log warnings when risks detected but not failing",
    )


class ProjectSettings(BaseSettings):
    env: Environment = Environment.LOCAL
    model_config_settings: ModelConfigSettings = ModelConfigSettings()
    guardrails: GuardrailSettings = GuardrailSettings()

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.prod"),
        env_file_encoding="utf-8",
        extra="allow",
        env_nested_delimiter="__",
    )

    # Add a validator to convert string to Environment enum
    @field_validator("env", mode="before")
    def validate_env(cls, value):
        if isinstance(value, str):
            return Environment[value.upper()]  # Convert string to enum (e.g., "local" -> Environment.LOCAL)
        return value


@lru_cache
def get_project_settings() -> ProjectSettings:
    return ProjectSettings()


CONFIG = get_project_settings()
