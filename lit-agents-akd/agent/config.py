from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

from dotenv import load_dotenv
import os

load_dotenv()

TOP_N_WIKI_RESULTS = int(os.getenv('TOP_N_WIKI_RESULTS', 1))
MAX_NUM_TURNS = int(os.getenv('MAX_NUM_TURNS', 5))
RETRY_ATTEMPTS = int(os.getenv('RETRY_ATTEMPTS', 3))
NUM_EDITORS = int(os.getenv('NUM_EDITORS', 3))

FAST_LLM = os.getenv('FAST_LLM', 'meta-llama/Llama-3.1-8B-Instruct')
LONG_CTX_LLM = os.getenv('LONG_CTX_LLM', 'meta-llama/Llama-3.1-8B-Instruct')

SERVER = os.getenv('SERVER', 'openai')
RITS_API_KEY = os.getenv('RITS_API_KEY', None)

OPENAI_API_BASE_FAST_LLM = os.getenv('OPENAI_API_BASE_FAST_LLM', None)
OPENAI_API_BASE_LONG_CTX_LLM = os.getenv('OPENAI_API_BASE_LONG_CTX_LLM', None)

SEARCH_ENGINE = os.getenv('SEARCH_ENGINE', 'duckduckgo')

EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', "nasa-impact/nasa-smd-ibm-st-v2")

if SERVER == 'openai':
    print("initialising OpenAI Chat")
    fast_llm = ChatOpenAI(model=FAST_LLM)
    long_context_llm = ChatOpenAI(model=LONG_CTX_LLM)
elif SERVER == 'ollama':
    print("initialising Ollama")
    fast_llm = llm = ChatOllama(
        model=FAST_LLM,
        temperature=0,
        # other params...
    )
    long_context_llm = llm = ChatOllama(
        model=LONG_CTX_LLM,
        temperature=0,
        # other params...
    )
else:
    print("initialising vllm")
    fast_llm = ChatOpenAI(
        model=FAST_LLM,
        temperature=0,
        max_retries=2,
        api_key='/',
        base_url=OPENAI_API_BASE_FAST_LLM,
        default_headers={'RITS_API_KEY': RITS_API_KEY} if RITS_API_KEY is not None else None
    )
    long_context_llm = ChatOpenAI(
        model=LONG_CTX_LLM,
        temperature=0,
        max_retries=2,
        api_key='/',
        base_url=OPENAI_API_BASE_LONG_CTX_LLM,
        default_headers={'RITS_API_KEY': RITS_API_KEY} if RITS_API_KEY is not None else None
    )


if SEARCH_ENGINE == 'duckduckgo':
    search_engine_wrapper = DuckDuckGoSearchAPIWrapper()    
elif SEARCH_ENGINE == 'google':
    search_engine_wrapper = GoogleSearchAPIWrapper()

wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, top_k_results=TOP_N_WIKI_RESULTS)
