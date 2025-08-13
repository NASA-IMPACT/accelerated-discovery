from akd.agents import BaseAgentConfig
from akd.agents.extraction import EstimationExtractionAgent
from akd.agents.intents import IntentAgent
from akd.agents.query import FollowUpQueryAgent, QueryAgent
from akd.agents.relevancy import MultiRubricRelevancyAgent, RelevancyAgent
from akd.agents.search import ControlledSearchAgent
from akd.configs.project import CONFIG
from akd.configs.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    FOLLOWUP_QUERY_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    MULTI_RUBRIC_RELEVANCY_SYSTEM_PROMPT,
    QUERY_SYSTEM_PROMPT,
)


def create_intent_agent(
    config: BaseAgentConfig | None = None,
    debug: bool = False,
) -> IntentAgent:
    config = config or BaseAgentConfig(
        api_key=CONFIG.model_config_settings.api_keys.openai,
        model_name=CONFIG.model_config_settings.model_name,
        temperature=CONFIG.model_config_settings.temperature,
        system_prompt=INTENT_SYSTEM_PROMPT,
    )
    return IntentAgent(config, debug=debug)


def create_extraction_agent(
    config: BaseAgentConfig | None = None,
    debug: bool = False,
) -> EstimationExtractionAgent:
    config = config or BaseAgentConfig(
        api_key=CONFIG.model_config_settings.api_keys.openai,
        model_name=CONFIG.model_config_settings.model_name,
        temperature=CONFIG.model_config_settings.temperature,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
    )
    return EstimationExtractionAgent(config, debug=debug)


def create_query_agent(
    config: BaseAgentConfig | None = None,
    debug: bool = False,
) -> QueryAgent:
    config = config or BaseAgentConfig(
        api_key=CONFIG.model_config_settings.api_keys.openai,
        model_name=CONFIG.model_config_settings.model_name,
        temperature=CONFIG.model_config_settings.temperature,
        system_prompt=QUERY_SYSTEM_PROMPT,
    )
    return QueryAgent(config, debug=debug)


def create_multi_rubric_relevancy_agent(
    config: BaseAgentConfig | None = None,
    debug: bool = False,
) -> MultiRubricRelevancyAgent:
    config = config or BaseAgentConfig(
        api_key=CONFIG.model_config_settings.api_keys.openai,
        model_name=CONFIG.model_config_settings.model_name,
        temperature=CONFIG.model_config_settings.temperature,
        system_prompt=MULTI_RUBRIC_RELEVANCY_SYSTEM_PROMPT,
    )
    return MultiRubricRelevancyAgent(config, debug=debug)


def create_followupquery_agent(
    config: BaseAgentConfig | None = None,
    debug: bool = False,
) -> FollowUpQueryAgent:
    """Create a FollowUpQueryAgent instance."""
    config = config or BaseAgentConfig(
        api_key=CONFIG.model_config_settings.api_keys.openai,
        model_name=CONFIG.model_config_settings.model_name,
        temperature=CONFIG.model_config_settings.temperature,
        system_prompt=FOLLOWUP_QUERY_SYSTEM_PROMPT,
    )
    return FollowUpQueryAgent(config, debug=debug)


def create_relevancy_agent(
    config: BaseAgentConfig | None = None,
    debug: bool = False,
) -> RelevancyAgent:
    """Create a basic RelevancyAgent instance."""
    config = config or BaseAgentConfig(
        api_key=CONFIG.model_config_settings.api_keys.openai,
        model_name=CONFIG.model_config_settings.model_name,
        temperature=CONFIG.model_config_settings.temperature,
        system_prompt=DEFAULT_SYSTEM_PROMPT,  # Use default for basic relevancy
    )
    return RelevancyAgent(config, debug=debug)


def create_lit_agent(
    config: BaseAgentConfig | None = None,
    debug: bool = False,
) -> ControlledSearchAgent:
    """Create a ControlledSearchAgent with default configuration."""
    from akd.agents.search import ControlledSearchAgentConfig

    # Use the new agent's config system
    agent_config = ControlledSearchAgentConfig()
    if config:
        # Map basic config properties to new config if provided
        agent_config.debug = debug

    return ControlledSearchAgent(config=agent_config, debug=debug)
