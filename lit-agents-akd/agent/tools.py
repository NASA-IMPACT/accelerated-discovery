from typing import Optional
import json

from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.runnables import chain as as_runnable
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


from agent.utils.prompts import *
from agent.utils.outline_utils import *
from agent.utils.model_utils import *
from agent.utils.interview_utils import *
from agent.utils.article_utils import *
from agent.config import fast_llm, long_context_llm, wikipedia_retriever, EMBEDDING_MODEL_ID, search_engine_wrapper
from agent.state import InterviewState


# ======================
# Initialise Research
# ======================

@as_runnable
async def survey_subjects(topic: str):
    gen_related_prompt = format_messages(gen_related_topics_prompt.invoke(topic).to_messages())
    response = await fast_llm.ainvoke(input=gen_related_prompt, extra_body={"guided_json": RelatedSubjects.model_json_schema()})
    related_subjects = RelatedSubjects.model_validate_json(response.content)
    retrieved_docs = await wikipedia_retriever.abatch(
        related_subjects.topics, return_exceptions=True
    )
    all_docs = []
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        all_docs.extend(docs)
    formatted = format_docs(all_docs)
    persp_prompt = format_messages(gen_perspectives_prompt.invoke({"examples": formatted, "topic": topic}).to_messages())
    response = await fast_llm.ainvoke(input=persp_prompt, extra_body={"guided_json": Perspectives.model_json_schema()})
    perspectives = Perspectives.model_validate_json(response.content)
    return perspectives


def get_draft_outline(topic):
    prompt = format_messages(direct_gen_outline_prompt.invoke(topic).to_messages())
    response = fast_llm.invoke(input=prompt, extra_body={"guided_json": Outline.model_json_schema()})
    return Outline.model_validate_json(response.content)


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
        # clean_name = format_name(state["messages"][i].name)
        # state["messages"][i].name = clean_name
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


@tool
def search_engine(query: str, top_n: int = 5):
    """Search and return the top 5 results."""
    results = search_engine_wrapper.results(query, top_n)
    return [{"content": r["snippet"], "url": r["link"]} for r in results]


async def gen_answer(
    state: InterviewState,
    config: Optional[RunnableConfig] = None,
    name: str = "Subject_Matter_Expert",
    max_str_len: int = 15000,
):
    swapped_state = swap_roles(state, name)
    gen_queries_p = format_messages(gen_queries_prompt.invoke({"messages": swapped_state['messages']}).to_messages())
    response = await fast_llm.ainvoke(input=gen_queries_p, extra_body={"guided_json": Queries.model_json_schema()})
    queries = Queries.model_validate_json(response.content)
    query_results = await search_engine.abatch(
        queries.queries, config, return_exceptions=True
    )
    successful_results = [
        res for res in query_results if not isinstance(res, Exception)
    ]
    all_query_results = {
        res["url"]: res["content"] for results in successful_results for res in results
    }
    dumped = json.dumps(all_query_results)[:max_str_len]
    ai_message: AIMessage = response.content
    tool_id = response.id
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])
    gen_ans_prompt = format_messages(gen_answer_prompt.invoke({"messages": swapped_state['messages']}).to_messages())
    response = await fast_llm.ainvoke(input=gen_ans_prompt, extra_body={"guided_json": AnswerWithCitations.model_json_schema()})
    generated = AnswerWithCitations.model_validate_json(response.content)
    cited_urls = set(generated.cited_urls)
    cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
    formatted_message = AIMessage(name=name, content=generated.as_str)
    return {"messages": [formatted_message], "references": cited_references}


# ======================
# Refine Outline
# ======================

async def get_refined_outline(topic, old_outline, conversations):
    ref_outline_prompt = format_messages(refine_outline_prompt.invoke({
            "topic": topic,
            "old_outline": old_outline,
            "conversations": conversations,
        }).to_messages())
    response = await long_context_llm.ainvoke(input=ref_outline_prompt, extra_body={"guided_json": Outline.model_json_schema()})
    return Outline.model_validate_json(response.content)


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


async def write_sections(prompt):
    sec_writer_prompt = format_messages(prompt.to_messages())
    response = await long_context_llm.ainvoke(input=sec_writer_prompt, extra_body={"guided_json": WikiSection.model_json_schema()})
    return WikiSection.model_validate_json(response.content)


section_writer = (
    retrieve
    | section_writer_prompt
    | write_sections
)

writer = writer_prompt | long_context_llm | StrOutputParser()
