from typing import Dict

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_openai import ChatOpenAI

from akd.agents.search.aspect_search import (
    AspectSearchAgent,
    AspectSearchConfig,
    AspectSearchInputSchema,
)

from .structures import ResearchState
from .tools import (
    get_draft_outline,
    get_draft_outline_from_sketch,
    get_refined_outline,
    section_writer,
    writer,
)


async def initialize_research(state: ResearchState, fast_llm: ChatOpenAI) -> Dict:
    """
    Initializes the research process by generating a draft outline for the given topic.

    Args:
        state (ResearchState): Current research state.
        fast_llm (ChatOpenAI): A small LLM capable of structured output.

    Returns:
        Dict: Updated research state.
    """
    topic = state["topic"]
    outline_sketch = state["outline_sketch"]
    if outline_sketch:
        outline = get_draft_outline_from_sketch(
            topic=topic,
            outline_sketch=outline_sketch,
            fast_llm=fast_llm,
        )
    else:
        outline = get_draft_outline(topic=topic, fast_llm=fast_llm)
    return {
        **state,
        "outline": outline,
    }


async def conduct_interviews(
    state: ResearchState,
    aspect_search_config: AspectSearchConfig,
) -> Dict:
    """
    Conducts interviews between an SME by generating perspectives focusing on different aspects of the topic.

    Args:
        state (ResearchState): Current research state.
        aspect_search_config (AspectSearchConfig): Configuration for the aspect search agent.

    Returns:
        Dict: Updated research state.
    """
    topic = state["topic"]
    aspect_agent = AspectSearchAgent(aspect_search_config)
    aspect_output = await aspect_agent.arun(AspectSearchInputSchema(topic=topic))
    for i in range(len(aspect_output.interview_results)):
        aspect_output.interview_results[i].pop("search_results")
    return {
        **state,
        "perspectives": aspect_output.perspectives,
        "interview_results": aspect_output.interview_results,
        "references": aspect_output.references,
        "search_results": aspect_output.search_results,
    }


async def refine_outline(state: ResearchState, long_context_llm: ChatOpenAI) -> Dict:
    """
    Refines the article outline using interview conversations.

    Args:
        state (ResearchState): Current research state.
        long_context_llm (ChatOpenAI): An LLM capable of handling long context.

    Returns:
        Dict: Updated research state.
    """

    def format_conversation(interview_state):
        messages = interview_state["messages"]
        convo = "\n".join(f"{m.name}: {m.content}" for m in messages)
        return f"Conversation with {interview_state['editor'].name}\n\n" + convo

    conversations = "\n\n".join(
        [
            format_conversation(interview_state)
            for interview_state in state["interview_results"]
        ],
    )
    updated_outline = await get_refined_outline(
        topic=state["topic"],
        old_outline=state["outline"].as_str,
        conversations=conversations,
        long_context_llm=long_context_llm,
    )
    return {**state, "outline": updated_outline}


async def index_references(state: ResearchState, vector_store: VectorStore) -> Dict:
    """
    Indexes reference documents for retrieval.

    Args:
        state (ResearchState): Current research state.
        vector_store (VectorStore): In memory vector store to index documents for the current topic.

    Returns:
        ResearchState: Updated research state.
    """
    reference_docs = [
        Document(page_content=v, metadata={"source": k})
        for k, v in state["references"].items()
    ]
    await vector_store.aadd_documents(reference_docs)
    return state


async def write_sections(
    state: ResearchState,
    long_context_llm: ChatOpenAI,
    retriever: VectorStoreRetriever,
) -> Dict:
    """
    Writes content for each section of a research document, using a long-context LLM and a retriever.

    Args:
        state (ResearchState): Current research state.
        long_context_llm (ChatOpenAI):An LLM capable of handling long context.
        retriever (VectorStoreRetriever): Vectorstore retriever for fetching relevant documents.

    Returns:
        Dict: Updated research state.
    """
    outline = state["outline"]
    sections = await section_writer(
        outline=outline,
        sections=outline.sections,
        topic=state["topic"],
        long_context_llm=long_context_llm,
        retriever=retriever,
    )
    return {
        **state,
        "sections": sections,
    }


async def write_article(state: ResearchState, long_context_llm: ChatOpenAI) -> Dict:
    """
    Writes content for each section of a research document, using a long-context LLM and a retriever.

    Args:
        state (ResearchState): Current research state.
        long_context_llm (ChatOpenAI): An LLM capable of handling long context.
        retriever (VectorStoreRetriever): Vectorstore retriever for fetching relevant documents.

    Returns:
        Dict: Updated research state.
    """
    topic = state["topic"]
    sections = state["sections"]
    draft = "\n\n".join([section.as_str for section in sections])
    article = await writer(topic=topic, draft=draft, long_context_llm=long_context_llm)
    return {
        **state,
        "article": article,
    }
