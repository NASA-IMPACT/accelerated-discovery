from __future__ import annotations

import tomllib

from pydantic import ValidationError
from pydantic_settings import BaseSettings

from akd.tools.fact_reasoner_tool import FactReasonerToolConfig
from akd.tools.scrapers.web_scrapers import WebpageScraperToolConfig
from akd.tools.search import SearxNGSearchToolConfig
from akd.tools.vector_database import VectorDBToolConfig


class LitAgentSettings(BaseSettings):
    search: SearxNGSearchToolConfig
    scraper: WebpageScraperToolConfig
    vectordb_search: VectorDBToolConfig
    fact_reasoner: FactReasonerToolConfig

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
