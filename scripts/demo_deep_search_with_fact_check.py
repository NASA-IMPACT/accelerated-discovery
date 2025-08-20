"""
Demo script for running a full research and fact-checking workflow.

This script demonstrates a complete pipeline:
1.  Run the DeepLitSearchAgent to find relevant literature and generate a report.
2.  Use the TextSplitterTool to chunk the retrieved source documents.
3.  Use the VectorDBTool to index the chunks into a persistent ChromaDB.
    This step is so that we can also use the same documents to verify the answer.
4.  Use the FactCheckTool to verify the generated report.
"""

import asyncio
import sys

import markdown
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from akd.agents.search import (
    DeepLitSearchAgent,
    DeepLitSearchAgentConfig,
    LitSearchAgentInputSchema,
)
from akd.configs.project import get_project_settings
from akd.tools.fact_check import FactCheckInputSchema, FactCheckTool
from akd.tools.text_splitter import (
    TextSplitterInputSchema,
    TextSplitterTool,
    TextSplitterToolConfig,
)
from akd.tools.vector_db_tool import VectorDBTool


def process_report(report):
    """Processes a report (in markdown) to remove headers and reference lists."""
    html = markdown.markdown(report)
    soup = BeautifulSoup(html, "html.parser")

    # Ignore header values
    content_tags = soup.find_all(["p", "li"])

    prose_fragments = []
    for tag in content_tags:
        is_reference_link = (
            tag.name == "li" and tag.find("a") and len(tag.contents) == 1
        )

        if is_reference_link:
            continue
        else:
            prose_fragments.append(tag.get_text(strip=True))

    cleaned_markdown = "\n".join(prose_fragments)
    return cleaned_markdown


async def main():
    # Check for API keys
    settings = get_project_settings()
    if not settings.model_config_settings.api_keys.openai:
        print(
            "No OpenAI API key found. Please set OPENAI_API_KEY environment variable.",
        )
        return

    # Configure and run the agent
    agent_config = DeepLitSearchAgentConfig(
        max_research_iterations=1,
        use_semantic_scholar=False,  # avoid rate limits
        enable_full_content_scraping=True,
        debug=True,
    )
    agent = DeepLitSearchAgent(config=agent_config)

    research_query = "What evidence is there for water on Mars?"
    input_params = LitSearchAgentInputSchema(query=research_query, max_results=3)

    print(f"--- Starting research for: '{research_query}' ---")
    research_output = await agent._arun(input_params)

    report_results = [
        res
        for res in research_output.results
        if res.get("url") == "deep-research://report"
    ]
    source_results = [
        res
        for res in research_output.results
        if res.get("url") != "deep-research://report"
    ]

    if not source_results:
        print("No source documents were found. Exiting.")
        return

    if not report_results:
        print("No research report was generated. Exiting.")
        return

    research_report = report_results[0]

    print(f"\n--- Found {len(source_results)} source documents and research report ---")

    # Convert search results to Langchain Document objects for processing
    documents_to_process = [
        Document(
            page_content=res["content"],
            metadata={"source": res["url"], "title": res["title"]},
        )
        for res in source_results
        if res.get("content")
    ]

    splitter_config = TextSplitterToolConfig(chunk_size=2000)
    splitter_tool = TextSplitterTool(config=splitter_config, debug=True)
    splitter_input = TextSplitterInputSchema(documents=documents_to_process)

    print(
        f"\n--- Splitting {len(documents_to_process)} documents into smaller chunks ---",
    )
    split_output = await splitter_tool._arun(splitter_input)
    chunks = split_output.chunks

    print(f"Created {len(chunks)} chunks.")

    vector_db_tool = VectorDBTool()

    print(f"\n--- Indexing {len(chunks)} chunks into ChromaDB ---")

    vector_db_tool.index(chunks)

    print(f"Indexing complete! Database is located at: {vector_db_tool.config.db_path}")

    print("\n--- Fact-checking the generated research report ---")
    fact_check_tool = FactCheckTool()

    # Clean markdown formatting from report
    report = research_report.get("content", "")

    plaintext_report = process_report(report)

    fact_check_input = FactCheckInputSchema(
        question=research_report.get("query", research_query),
        answer=plaintext_report,
    )

    print(plaintext_report)
    fact_check_result = await fact_check_tool.arun(params=fact_check_input)

    print("\n--- Fact-Check Complete ---")
    score = fact_check_result.fact_reasoner_score.get("factuality_score", 0)
    num_supported = len(fact_check_result.supported_atoms)
    num_not_supported = len(fact_check_result.not_supported_atoms)

    print(f"Factuality Score: {score:.2%}")
    print(f"Supported Atoms: {num_supported}")
    print(f"Not Supported Atoms: {num_not_supported}")
    print(f"Graph ID: {fact_check_result.graph_id}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nDemo failed with an unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
