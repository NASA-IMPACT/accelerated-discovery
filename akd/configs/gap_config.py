from __future__ import annotations

import tomllib

from pydantic import ValidationError
from pydantic_settings import BaseSettings

from akd.tools.scrapers import DoclingScraperConfig
from akd.tools.search import SemanticScholarSearchToolConfig
from akd.agents._base import BaseAgentConfig


class GapAgentSettings(BaseSettings):
    docling_scraper_config: DoclingScraperConfig
    s2_tool_config: SemanticScholarSearchToolConfig
    api_key: str
    model_name: str
    temperature: float = 0.0
    system_prompt: str = None

    @classmethod
    def from_toml(cls, toml_file_path: str) -> GapAgentSettings:
        with open(toml_file_path, "rb") as f:
            config = tomllib.load(f)
        return cls(**config)


def get_gap_agent_settings(toml_file_path: str) -> GapAgentSettings:
    try:
        gap_agent_settings = GapAgentSettings.from_toml(toml_file_path)
        return gap_agent_settings
    except ValidationError as e:
        print(f"Gap agent config validation error: {e}")
        raise
