from functools import partial

from langchain_core.output_parsers import StrOutputParser

from .prompts import (
    GEN_INITIAL_OUTLINE_PROMPT,
    REFINE_OUTLINE_PROMPT,
    SECTION_WRITER_PROMPT,
    WRITER_PROMPT,
)
from .structures import ArticleSection, Outline, Section

# =============================================================================
# Initialise Research
# =============================================================================


def get_draft_outline(topic, fast_llm):
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


async def get_refined_outline(topic, old_outline, conversations, long_context_llm):
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


async def retrieve(inputs: dict, retriever):
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
    sections: list[Section],
    topic: str,
    long_context_llm,
    retriever,
):
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


async def writer(topic: str, draft: str, long_context_llm):
    writer = WRITER_PROMPT | long_context_llm | StrOutputParser()
    article = await writer.ainvoke({"topic": topic, "draft": draft})
    return article
