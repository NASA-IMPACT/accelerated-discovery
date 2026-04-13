from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

from akd.configs.storm_config import STORM_SETTINGS, StormSettings

class StormConfig:

    def __init__(self):
        self.fast_llm = None
        self.long_context_llm = None
        self.search_engine_wrapper = None
        self.wikipedia_retriever = None
        self.EMBEDDING_MODEL_ID = None
        self.MAX_NUM_TURNS = 3
        self.RETRY_ATTEMPTS = 3
        self.NUM_EDITORS = 3

    def initialize(self, config: dict):
        
        self.EMBEDDING_MODEL_ID = config.EMBEDDING_MODEL_ID
        self.MAX_NUM_TURNS = config.MAX_NUM_TURNS
        self.RETRY_ATTEMPTS = config.RETRY_ATTEMPTS
        self.NUM_EDITORS = config.NUM_EDITORS

        if config.SERVER == 'openai':
            print("initialising OpenAI Chat")
            self.fast_llm = ChatOpenAI(model=config.FAST_LLM)
            self.long_context_llm = ChatOpenAI(model=config.LONG_CTX_LLM)
        else:
            print("initialising vllm / ollama")
            self.fast_llm = ChatOpenAI(
                model=config.FAST_LLM,
                temperature=0,
                max_retries=2,
                api_key='/',
                base_url=config.OPENAI_API_BASE_FAST_LLM,
                default_headers={'RITS_API_KEY': config.RITS_API_KEY} if config.RITS_API_KEY is not None else None
            )
            self.long_context_llm = ChatOpenAI(
                model=config.LONG_CTX_LLM,
                temperature=0,
                max_retries=2,
                api_key='/',
                base_url=config.OPENAI_API_BASE_LONG_CTX_LLM,
                default_headers={'RITS_API_KEY': config.RITS_API_KEY} if config.RITS_API_KEY is not None else None
            )

        if config.SEARCH_ENGINE == 'duckduckgo':
            self.search_engine_wrapper = DuckDuckGoSearchAPIWrapper()
        elif config.SEARCH_ENGINE == 'google':
            self.search_engine_wrapper = GoogleSearchAPIWrapper()

        self.wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, top_k_results=config.TOP_N_WIKI_RESULTS)

storm_config = StormConfig()
storm_config.initialize(STORM_SETTINGS)


def initialise_storm_config(config: StormSettings):
    storm_config.initialize(config)

