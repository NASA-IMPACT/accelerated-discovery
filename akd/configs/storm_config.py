from typing import Optional

from pydantic import Field
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class StormSettings(BaseSettings):
    model_config = SettingsConfigDict()
    FAST_LLM: str = Field('gpt-4o-mini', alias='FAST_LLM')
    LONG_CTX_LLM: str = Field('gpt-4o', alias='LONG_CTX_LLM')
    OPENAI_API_BASE_FAST_LLM: Optional[str] = Field(None, alias='OPENAI_API_BASE_FAST_LLM')
    OPENAI_API_BASE_LONG_CTX_LLM: Optional[str] = Field(None, alias='OPENAI_API_BASE_LONG_CTX_LLM')
    SERVER: Optional[str] = Field("openai", alias='SERVER')
    RITS_API_KEY: Optional[str] = Field(None, alias='RITS_API_KEY')
    SEARCH_ENGINE: Optional[str] = Field('duckduckgo', alias='SEARCH_ENGINE')
    GOOGLE_API_KEY: Optional[str] = Field(None, alias='GOOGLE_API_KEY')
    GOOGLE_CSE_ID: Optional[str] = Field(None, alias='GOOGLE_CSE_ID')
    EMBEDDING_MODEL_ID: Optional[str] = Field('nasa-impact/nasa-smd-ibm-st-v2', alias='EMBEDDING_MODEL_ID')
    TOP_N_WIKI_RESULTS: Optional[int] = Field(1, alias='TOP_N_WIKI_RESULTS')
    MAX_NUM_TURNS: Optional[int] = Field(3, alias='MAX_NUM_TURNS')
    RETRY_ATTEMPTS: Optional[int] = Field(3, alias='RETRY_ATTEMPTS')
    NUM_EDITORS: Optional[int] = Field(3, alias='NUM_EDITORS')


def get_storm_settings() -> StormSettings:
    return StormSettings()


STORM_SETTINGS = get_storm_settings()
