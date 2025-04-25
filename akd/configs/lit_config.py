from __future__ import annotations

import tomllib
from typing import List, Literal, Optional

from pydantic import BaseModel, ValidationError, model_validator
from pydantic.fields import Field
from pydantic.networks import HttpUrl
from pydantic_settings import BaseSettings
from typing_extensions import Self

from akd.tools.scrapers.web_scrapers import WebpageScraperToolConfig

from ..structures import SearchResultItem


class SearxNGSettings(BaseModel):
    base_url: HttpUrl = "http://localhost:8080"
    max_results: int = 10
    engines: List[str] = Field(default_factory=lambda: ["google", "arxiv", "google_scholar"])
    max_pages: int = 25
    results_per_page: int = 10
    score_cutoff: float = 0.25
    debug: bool = False

    @model_validator(mode="after")
    def validate_engines(self) -> Self:
        if not self.engines:
            raise ValueError("At least one engine must be specified")
        return self


class WebpageScraperSettings(BaseModel):
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        description="User agent string to use for requests.",
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for HTTP requests.",
    )
    max_content_length: int = Field(
        default=100_000_000,
        description="Maximum content length in bytes to process.",
    )
    debug: bool = Field(
        default=True,
        description="Boolean flag for debug mode",
    )


class LitAgentSettings(BaseSettings):
    search: SearxNGSettings
    scraper: WebpageScraperSettings

    @classmethod
    def from_toml(cls, toml_file_path: str) -> LitAgentSettings:
        with open(toml_file_path, "rb") as f:
            config = tomllib.load(f)
        return cls(**config)


def get_lit_agent_settings(toml_file_path: str) -> LitAgentSettings:
    try:
        lit_agent_settings = LitAgentSettings.from_toml(toml_file_path)
        return lit_agent_settings
    except ValidationError as e:
        print(f"Lit agent config validation error: {e}")
        raise
