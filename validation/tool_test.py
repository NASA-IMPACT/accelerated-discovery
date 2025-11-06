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

# -------------------- Load Gold Dataset --------------------

# Import val df
val_df = pd.read_csv(str(get_akd_root() / "validation" / "data" / "code_validation_data_v2.csv"))
val_df = val_df[:17]

# -------------------- Evaluation Loop --------------------


async def run_validation(k):
    output_df = pd.DataFrame(columns=["query", "gold_urls", "predicted_urls", "rr", "accuracy"])
    all_ground_truth = []
    all_retrieved = []

    for i in range(len(val_df)):
        query = val_df.iloc[i]["question"]
        repo_url_value = val_df.iloc[i]["url"]
        if isinstance(repo_url_value, str):
            # If it's a string representation of a list, parse it
            gold_urls = [url.strip() for url in ast.literal_eval(repo_url_value)]
        else:
            # If it's already a list
            gold_urls = [url.strip() for url in repo_url_value]

        try:
            print(f"Query {i}: {query}")
            output = await _search_tool._arun(
                CodeSearchToolInputSchema(queries=[query], query=query, max_results=10, top_k=10)
            )
        except Exception as e:
            print(f"Error on query {i}: {e}")
            all_ground_truth.append([Document(content=gold_url) for gold_url in gold_urls])
            all_retrieved.append([])
            output_df = output_df._append(
                {"query": query, "gold_urls": gold_urls, "predicted_urls": [], "rr": 0}, ignore_index=True
            )
            continue

        predicted_urls = [str(r.url).strip() for r in output.results[:k]]

        # Prepare documents for Haystack evaluator
        all_ground_truth.append([Document(content=gold_url) for gold_url in gold_urls])
        all_retrieved.append([Document(content=url) for url in predicted_urls])

        # print predicted urls in a line by line
        for url in predicted_urls:
            print(url)

        output_df = output_df._append(
            {"query": query, "gold_urls": gold_urls, "predicted_urls": predicted_urls, "rr": None, "accuracy": None},
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
        results = await asyncio.gather(run_validation(5), run_validation(10))
        return results

    results = asyncio.run(main())
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)
    print(results)
