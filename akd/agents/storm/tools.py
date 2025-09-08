from functools import partial
from typing import Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from .prompts import (
    GEN_INITIAL_OUTLINE_PROMPT,
    REFINE_OUTLINE_PROMPT,
    SECTION_WRITER_PROMPT,
    WRITER_PROMPT,
)
from .structures import ArticleSection, Outline, OutlineSection

# =============================================================================
# Initialise Research
# =============================================================================


def get_draft_outline(topic: str, fast_llm: ChatOpenAI) -> Outline:
    """
    Generates a structured draft outline for a given topic.

    Args:
        topic (str): The user-defined topic.
        fast_llm (ChatOpenAI): A small LLM capable of structured output.

    Returns:
        Outline: An outline generated for the topic containing sections, subsections and corresponsing descriptions.
    """
    generate_outline_direct = (
        GEN_INITIAL_OUTLINE_PROMPT
        | fast_llm.with_structured_output(
            Outline,
        )
    )
    return generate_outline_direct.invoke({"topic": topic})


# =============================================================================
# Refine Outline
# =============================================================================


async def get_refined_outline(
    topic: str,
    old_outline: str,
    conversations: str,
    long_context_llm: ChatOpenAI,
) -> Outline:
    """
    Refines the draft outline based on the aspect search conversations and context.

    Args:
        topic (str): The user-defined topic.
        old_outline (str): The initial draft outline to be refined.
        conversations (str): Conversation history from aspect search.
        long_context_llm (ChatOpenAI): An LLM capable of handling long context.

    Returns:
        Outline: A refined outline object.
    """
    refine_outline_chain = (
        REFINE_OUTLINE_PROMPT
        | long_context_llm.with_structured_output(
            Outline,
        )
    )
    return await refine_outline_chain.ainvoke(
        {
            "topic": topic,
            "old_outline": old_outline,
            "conversations": conversations,
        },
    )


# =============================================================================
# Write Article
# =============================================================================


async def retrieve(inputs: Dict, retriever: VectorStoreRetriever) -> Dict:
    """
    Retrieves relevant documents based on a topic and section from the in-memory vector store.

    Args:
        inputs (Dict): Dictionary containing "topic" and "section" keys.
        retriever (VectorStoreRetriever): Vectorstore retriever for fetching relevant documents.

    Returns:
        Dict: Original input dictionary augmented with a formatted string of retrieved documents.
    """
    docs = await retriever.ainvoke(inputs["topic"] + ": " + inputs["section"])
    formatted = "\n".join(
        [
            f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ],
    )
    return {"docs": formatted, **inputs}


async def section_writer(
    outline: Outline,
    sections: List[OutlineSection],
    topic: str,
    long_context_llm: ChatOpenAI,
    retriever: VectorStoreRetriever,
) -> List[ArticleSection]:
    """
    Generates article sections based on a given outline and topic by retrieving relevant context.

    Args:
        outline (Outline): The full outline structure for the article.
        sections (List[OutlineSection]): List of sections to be written.
        topic (str): The user-defined topic.
        long_context_llm (ChatOpenAI): An LLM capable of handling long context.
        retriever (VectorStoreRetriever): Vectorstore retriever for fetching relevant documents.

    Returns:
        List[ArticleSection]: A list of generated article sections.
    """
    section_writer = (
        partial(retrieve, retriever=retriever)
        | SECTION_WRITER_PROMPT
        | long_context_llm.with_structured_output(ArticleSection)
    )
    sections = await section_writer.abatch(
        [
            {
                "outline": outline.as_str,
                "section": section.section_title,
                "topic": topic,
            }
            for section in sections
        ],
    )
    return sections


async def writer(topic: str, draft: str, long_context_llm: ChatOpenAI) -> str:
    """
    Compiles a full article using the topic and draft.

    Args:
        topic (str): The user-defined topic.
        draft (str): The draft of the article.
        long_context_llm (ChatOpenAI): An LLM capable of handling long context.

    Returns:
        str: The generated article as a string.
    """
    writer = WRITER_PROMPT | long_context_llm | StrOutputParser()
    article = await writer.ainvoke({"topic": topic, "draft": draft})
    return article
