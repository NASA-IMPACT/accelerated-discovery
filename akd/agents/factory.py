from typing import List, Optional, Union

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import create_model, BaseModel, ConfigDict

from akd.configs.project import CONFIG
from akd.configs.prompts import INTENT_SYSTEM_PROMPT, EXTRACTION_SYSTEM_PROMPT, QUERY_SYSTEM_PROMPT
from akd.structures import ExtractionSchema, SingleEstimation

from .extraction import ExtractionInputSchema
from .intents import IntentAgent
from .query import QueryAgent


def create_intent_agent(config: Optional[ConfigDict] = None) -> IntentAgent:
    config = config or ConfigDict(
        client=ChatOpenAI(
                    api_key=CONFIG.model_config_settings.api_keys.openai,
                    model=CONFIG.model_config_settings.model_name,
                    temperature=0.0,
            ),
        system_prompt=ChatPromptTemplate.from_messages(
                    [
                        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                        MessagesPlaceholder(variable_name="memory"),
                    ]
                ),
    )
    return IntentAgent(config)


def create_extraction_agent(
    extraction_output_schema: Union[ExtractionSchema, List[SingleEstimation]],
    config: Optional[ConfigDict] = None,
) -> BaseModel:
    """
    Dynamically build the agent based on the type
    of extraction schema provided.
    """

    _ExtractionOutputSchema = create_model(
        "_ExtractionOutputSchema",
        answer=(extraction_output_schema, "Answer from the extraction"),
        __base__=BaseModel,
        __doc__="Extracted information from the research",
    )
    return BaseModel(
        ConfigDict(
            client=ChatOpenAI(
                    api_key=CONFIG.model_config_settings.api_keys.openai,
                    model=CONFIG.model_config_settings.model_name,
            ),
            system_prompt=ChatPromptTemplate.from_messages(
                    [
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        MessagesPlaceholder(variable_name="memory"),
                    ]
                ),
            input_schema=ExtractionInputSchema,
            output_schema=_ExtractionOutputSchema,
        ),
    )


def create_query_agent(config: Optional[ConfigDict] = None) -> QueryAgent:
    config = config or ConfigDict(
        client=ChatOpenAI(
                    api_key=CONFIG.model_config_settings.api_keys.openai,
                    model=CONFIG.model_config_settings.model_name,
            ),
        system_prompt=ChatPromptTemplate.from_messages(
                    [
                        {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                        MessagesPlaceholder(variable_name="memory"),
                    ]
                )
    )
    return QueryAgent(config)
