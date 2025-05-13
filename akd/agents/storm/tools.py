from typing import Optional
import json
import time

from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.runnables import chain as as_runnable
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from .utils.prompts import *
from .utils.outline_utils import *
from .utils.interview_utils import *
from .utils.article_utils import *
from .config import storm_config
from .state import InterviewState


fast_llm = storm_config.fast_llm
long_context_llm = storm_config.long_context_llm
wikipedia_retriever = storm_config.wikipedia_retriever
EMBEDDING_MODEL_ID = storm_config.EMBEDDING_MODEL_ID
search_engine_wrapper = storm_config.search_engine_wrapper


# ======================
# Initialise Research
# ======================

@as_runnable
async def survey_subjects(topic: str):
    # Generate perspectives
    gen_perspectives_chain = gen_perspectives_prompt | fast_llm.with_structured_output(
        Perspectives
    )
    # Expand topics
    expand_chain = gen_related_topics_prompt | fast_llm.with_structured_output(
            RelatedSubjects
    )
    related_subjects = await expand_chain.ainvoke({"topic": topic})
    retrieved_docs = await wikipedia_retriever.abatch(
        related_subjects.topics, return_exceptions=True
    )
    all_docs = []
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        all_docs.extend(docs)
    formatted = format_docs(all_docs)
    return await gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})


def get_draft_outline(topic):
    # Generate draft outline
    generate_outline_direct = direct_gen_outline_prompt | fast_llm.with_structured_output(
        Outline
    )
    return generate_outline_direct.invoke({"topic": topic})


async def get_perspectives(topic):
    return await survey_subjects.ainvoke(topic)


# ======================
# Interview Editors
# ======================


def tag_with_name(ai_message: AIMessage, name: str):
    ai_message.name = name
    return ai_message


def swap_roles(state: InterviewState, name: str):
    converted = []
    for i in range(len(state["messages"])):
        # Name must be formatted for OpenAI
        clean_name = format_name(state["messages"][i].name)
        state["messages"][i].name = clean_name
        message = state["messages"][i]
        if isinstance(message, AIMessage) and message.name != name:
            message = HumanMessage(**message.model_dump(exclude={"type"}))
        converted.append(message)
    return {"messages": converted}


@as_runnable
async def generate_question(state: InterviewState):
    editor = state["editor"]
    print(f"{editor.name} is speaking now")
    gn_chain = (
        RunnableLambda(swap_roles).bind(name=editor.name)
        | gen_qn_prompt.partial(persona=editor.persona)
        | fast_llm
        | RunnableLambda(tag_with_name).bind(name=editor.name)
    )
    result = await gn_chain.ainvoke(state)
    return {"messages": [result]}


# Search Engine
# TODO: Customise search results

@tool
def search_engine(query: str, top_n: int = 5):
    """Search and return the top 5 results."""
    results = search_engine_wrapper.results(query, top_n)
    # TODO: Replace, this is temporary to avoid ddg rate limits
    time.sleep(10)
    return [{"content": r["snippet"], "url": r["link"]} for r in results]


async def gen_answer(
    state: InterviewState,
    config: Optional[RunnableConfig] = None,
    name: str = "Subject_Matter_Expert",
    max_str_len: int = 15000,
):
    gen_queries_chain = gen_queries_prompt | fast_llm.with_structured_output(Queries, include_raw=True, method='function_calling')
    gen_answer_chain = gen_answer_prompt | fast_llm.with_structured_output(
        AnswerWithCitations, include_raw=True
    ).with_config(run_name="GenerateAnswer")


    swapped_state = swap_roles(state, name)
    queries = await gen_queries_chain.ainvoke(swapped_state)
    query_results = await search_engine.abatch(
        queries["parsed"].queries, config, return_exceptions=True
    )
    successful_results = [
        res for res in query_results if not isinstance(res, Exception)
    ]
    all_query_results = {
        res["url"]: res["content"] for results in successful_results for res in results
    }
    dumped = json.dumps(all_query_results)[:max_str_len]
    ai_message: AIMessage = queries["raw"]
    tool_call = queries["raw"].tool_calls[0]
    tool_id = tool_call["id"]
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])
    generated = await gen_answer_chain.ainvoke(swapped_state)
    cited_urls = set(generated["parsed"].cited_urls)
    cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
    formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
    return {"messages": [formatted_message], "references": cited_references}


# ======================
# Refine Outline
# ======================


async def get_refined_outline(topic, old_outline, conversations):
    refine_outline_chain = refine_outline_prompt | long_context_llm.with_structured_output(
        Outline
    )

    return await refine_outline_chain.ainvoke(
        {
            "topic": topic,
            "old_outline": old_outline,
            "conversations": conversations,
        }
    )


# ======================
# Write Article
# ======================


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID, model_kwargs={'device': "cpu"})
vectorstore = InMemoryVectorStore(embedding=embeddings)
retriever = vectorstore.as_retriever(k=3)

async def retrieve(inputs: dict):
    docs = await retriever.ainvoke(inputs["topic"] + ": " + inputs["section"])
    formatted = "\n".join(
        [
            f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )
    return {"docs": formatted, **inputs}


async def section_writer(outline:Outline, sections: list[Section], topic: str):
    section_writer = (
        retrieve
        | section_writer_prompt
        | long_context_llm.with_structured_output(WikiSection)
    )
    sections = await section_writer.abatch(
        [
            {
                "outline": outline.as_str,
                "section": section.section_title,
                "topic": topic,
            }
            for section in sections
        ]
    )
    return sections


async def writer(topic: str, draft: str):
    writer = writer_prompt | long_context_llm | StrOutputParser()
    article = await writer.ainvoke({"topic": topic, "draft": draft})
    return article