import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast

# Add the parent directory (the project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from akd.tools.search.code_search import (
    CodeSearchToolInputSchema,
    LocalRepoCodeSearchTool,
    LocalRepoCodeSearchToolConfig,
    SDECodeSearchTool,
    SDECodeSearchToolConfig,
    CompositeCodeSearchTool,
    CompositeCodeSearchToolConfig,
)
from akd.agents.search import ControlledSearchAgent, ControlledSearchAgentConfig, LitSearchAgentInputSchema
from akd.agents.query import QueryAgent, FollowUpQueryAgent
from akd.agents._base import BaseAgentConfig
from pprint import pprint
from akd.utils import get_akd_root
from akd.configs.prompts import QUERY_SYSTEM_PROMPT, MULTI_RUBRIC_RELEVANCY_SYSTEM_PROMPT
from akd.agents.relevancy import MultiRubricRelevancyAgent
from akd.tools.reranker import RerankerToolConfig

from akd.agents.search import CodeSearchAgent, CodeSearchAgentConfig

# Haystack imports
from haystack import Document
from haystack.components.evaluators import DocumentMRREvaluator

evaluator = DocumentMRREvaluator()

# Set OpenAI API key
import dotenv

dotenv.load_dotenv()


# custom accuracy evaluator
# if any of the ground truth urls are in the retrieved urls, then the accuracy is 1, otherwise 0
# returns a column of accuracies
def evaluator_acc(all_ground_truth, all_retrieved):
    accuracies = []
    for i in range(len(all_ground_truth)):
        truth_urls = [u.content for u in all_ground_truth[i]]
        retrieved_urls = [u.content for u in all_retrieved[i]]
        # if any of the truth urls are in the retrieved urls, then the accuracy is 1, otherwise 0
        if any(truth_url in retrieved_urls for truth_url in truth_urls):
            accuracies.append(1)
        else:
            accuracies.append(0)
    return accuracies


# -------------------- Setup --------------------

_cfg_sde = SDECodeSearchToolConfig(
    base_url="https://d2kqty7z3q8ugg.cloudfront.net/api/code/search", debug=False, search_mode="vector", page_size=100
)
_search_tool_sde = SDECodeSearchTool(config=_cfg_sde)

# Run the code search tool once using /examples/code_search_test.py to download initial embeddings dump
# Drop the 'embeddings' column
df = pd.read_csv(str(get_akd_root() / "docs" / "repositories_with_embeddings_v6.csv"))
if "embeddings" in df.columns:
    df = df.drop(columns=["embeddings"])
df.to_csv(str(get_akd_root() / "docs" / "repositories_with_embeddings_v6_granite.csv"), index=False)


_search_cfg_local = LocalRepoCodeSearchToolConfig(
    data_file=str(get_akd_root() / "docs" / "repositories_with_embeddings_v6_granite.csv"),
    embedding_model_name="ibm-granite/granite-embedding-english-r2",
)

_search_tool_local = LocalRepoCodeSearchTool(config=_search_cfg_local)

_search_cfg_composite = CompositeCodeSearchToolConfig(
    rrf_keys=["url"],
    debug=False,
    reranker_config=RerankerToolConfig(model_name="ibm-granite/granite-embedding-reranker-english-r2"),
)

_search_tool = CompositeCodeSearchTool(config=_search_cfg_composite, tools=[_search_tool_local, _search_tool_sde])

# _search_tool = _search_tool_local

_search_agent_config = CodeSearchAgentConfig(
    debug=True,
    search_result_reranking=True,
    reranker_config=RerankerToolConfig(model_name="ibm-granite/granite-embedding-reranker-english-r2"),
)
_search_agent = CodeSearchAgent(config=_search_agent_config, search_tool=_search_tool)


# -------------------- Load Gold Dataset --------------------

# Import val df
val_df = pd.read_csv(str(get_akd_root() / "validation" / "data" / "code_validation_data_v2.csv"))

# -------------------- Evaluation Loop --------------------


async def process_single_query(i, query, gold_urls, k):
    """Process a single query asynchronously"""
    try:
        print(f"Query {i}: {query}")
        output = await _search_agent._arun(LitSearchAgentInputSchema(query=query, max_results=10, top_k=10))
        predicted_urls = [str(r.url).strip() for r in output.results[:k]]

        # print predicted urls in a line by line
        for url in predicted_urls:
            print(url)

        return {
            "query": query,
            "gold_urls": gold_urls,
            "predicted_urls": predicted_urls,
            "ground_truth_docs": [Document(content=gold_url) for gold_url in gold_urls],
            "retrieved_docs": [Document(content=url) for url in predicted_urls],
        }
    except Exception as e:
        print(f"Error on query {i}: {e}")
        return {
            "query": query,
            "gold_urls": gold_urls,
            "predicted_urls": [],
            "ground_truth_docs": [Document(content=gold_url) for gold_url in gold_urls],
            "retrieved_docs": [],
        }


async def run_validation(k, batch_size=10):
    output_df = pd.DataFrame(columns=["query", "gold_urls", "predicted_urls", "rr", "accuracy"])
    all_ground_truth = []
    all_retrieved = []

    # Process queries in batches
    for batch_start in range(0, len(val_df), batch_size):
        batch_end = min(batch_start + batch_size, len(val_df))
        print(f"\nProcessing batch {batch_start}-{batch_end - 1}")

        # Prepare batch tasks
        tasks = []
        for i in range(batch_start, batch_end):
            query = val_df.iloc[i]["question"]
            repo_url_value = val_df.iloc[i]["url"]
            if isinstance(repo_url_value, str):
                gold_urls = [url.strip() for url in ast.literal_eval(repo_url_value)]
            else:
                gold_urls = [url.strip() for url in repo_url_value]

            tasks.append(process_single_query(i, query, gold_urls, k))

        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks)

        # Collect results
        for result in batch_results:
            all_ground_truth.append(result["ground_truth_docs"])
            all_retrieved.append(result["retrieved_docs"])
            output_df = output_df._append(
                {
                    "query": result["query"],
                    "gold_urls": result["gold_urls"],
                    "predicted_urls": result["predicted_urls"],
                    "rr": None,
                    "accuracy": None,
                },
                ignore_index=True,
            )

    # Calculate MRR using Haystack evaluator
    result = evaluator.run(ground_truth_documents=all_ground_truth, retrieved_documents=all_retrieved)

    result_accuracy = evaluator_acc(all_ground_truth, all_retrieved)

    # Update output_df with individual MRR scores
    output_df["rr"] = result["individual_scores"]
    output_df["accuracy"] = pd.Series(result_accuracy)

    mrr = result["score"]
    accuracy = np.mean(result_accuracy)
    print(f"\nMRR@{k}: {mrr:.4f}, Accuracy@{k}: {accuracy:.4f}")
    return mrr, accuracy


if __name__ == "__main__":

    async def main():
        results = await run_validation(10)
        return results

    results = asyncio.run(main())
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)
    print(results)
