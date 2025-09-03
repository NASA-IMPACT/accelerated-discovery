from langchain_community.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from akd.agents.search.aspect_search import (
    AspectSearchAgent,
    AspectSearchConfig,
    AspectSearchInputSchema,
)

from .structures import ResearchState
from .tools import get_draft_outline, get_refined_outline, section_writer, writer


async def initialize_research(state: ResearchState, fast_llm: ChatOpenAI):
    topic = state["topic"]
    print(f"\n💬: {topic}\n")
    outline = get_draft_outline(topic, fast_llm=fast_llm)
    print("\n🤖: Here is a highlight of your article's initial outline.\n")
    print(f"{outline.as_str} ...")
    return {
        **state,
        "outline": outline,
    }


async def conduct_interviews(
    state: ResearchState,
    aspect_search_config: AspectSearchConfig,
):
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
    }


async def refine_outline(state: ResearchState, long_context_llm: ChatOpenAI):
    def format_conversation(interview_state):
        messages = interview_state["messages"]
        convo = "\n".join(f"{m.name}: {m.content}" for m in messages)
        return f"Conversation with {interview_state['editor'].name}\n\n" + convo

    convos = "\n\n".join(
        [
            format_conversation(interview_state)
            for interview_state in state["interview_results"]
        ],
    )
    updated_outline = await get_refined_outline(
        topic=state["topic"],
        old_outline=state["outline"].as_str,
        conversations=convos,
        long_context_llm=long_context_llm,
    )
    print(
        "\n🤖: Here is a highlight of your article's refined outline using the interviews for context.\n",
    )
    print(f"{updated_outline.as_str} ...")
    return {**state, "outline": updated_outline}


async def index_references(state: ResearchState, vector_store: VectorStore):
    print("\n🤖: Indexing references")
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
):
    outline = state["outline"]
    print("\n🤖: Writing each section")
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


async def write_article(state: ResearchState, long_context_llm: ChatOpenAI):
    topic = state["topic"]
    sections = state["sections"]
    print("\n🤖: Writing the article!")
    draft = "\n\n".join([section.as_str for section in sections])
    article = await writer(topic=topic, draft=draft, long_context_llm=long_context_llm)
    print("\n🤖: Done. Print your article below!")
    return {
        **state,
        "article": article,
    }
