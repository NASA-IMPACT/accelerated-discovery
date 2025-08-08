from langchain_core.runnables import chain as as_runnable
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

import json

from akd.tools.search import SearxNGSearchToolInputSchema
from .structures import (RelatedSubjects,
                         InterviewState,
                         Perspectives,
                         AnswerWithCitations,
                         Queries)
from .prompts import (gen_answer_prompt,
                      gen_queries_prompt,
                      gen_related_topics_prompt,
                      gen_perspectives_prompt,
                      gen_qn_prompt)


# add comment

def format_conversation(interview_state):
    messages = interview_state["messages"]
    convo = "\n".join(f"{m.name}: {m.content}" for m in messages)
    return f'Conversation with {interview_state["editor"].name}\n\n' + convo

def format_name(name):
    print(name)
    return "".join([char if char.isalnum() or char in "_-" else "_" for char in name])


def format_doc(doc, max_length=1000):
    related = "- ".join(doc.metadata["categories"])
    return f"### {doc.metadata['title']}\n\nSummary: {doc.page_content}\n\nRelated\n{related}"[
        :max_length
    ]


def format_docs(docs):
    return "\n\n".join(format_doc(doc) for doc in docs)


# add comment

@as_runnable
async def survey_subjects(topic: str, llm, wikipedia_retriever):
    # Expand topics
    expand_chain = gen_related_topics_prompt | llm.with_structured_output(
            RelatedSubjects
    )
    # Generate perspectives
    gen_perspectives_chain = gen_perspectives_prompt | llm.with_structured_output(
        Perspectives
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
        # top n results
    all_docs = all_docs[:2]
    formatted = format_docs(all_docs)
    return await gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})


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
async def generate_question(state: InterviewState, llm):
    editor = state["editor"]
    print(f"{editor.name} is speaking now")
    gn_chain = (
        RunnableLambda(swap_roles).bind(name=editor.name)
        | gen_qn_prompt.partial(persona=editor.persona)
        | llm
        | RunnableLambda(tag_with_name).bind(name=editor.name)
    )
    result = await gn_chain.ainvoke(state)
    return {"messages": [result]}


@as_runnable
async def generate_answer(
        state: InterviewState,
        llm,
        search_tool,
        name: str = "Subject_Matter_Expert",
        max_str_len: int = 15000,
    ):
    gen_answer_chain = gen_answer_prompt | llm.with_structured_output(
        AnswerWithCitations, include_raw=True
    ).with_config(run_name="GenerateAnswer")
    gen_queries_chain = gen_queries_prompt | llm.with_structured_output(Queries, include_raw=True, method='function_calling')
    swapped_state = swap_roles(state, name)
    queries = await gen_queries_chain.ainvoke(swapped_state)
    query_results = await search_tool.arun(
        SearxNGSearchToolInputSchema(queries=queries["parsed"].queries, category=None)
    )
    formatted_query_results = {
        str(res.url): res.content for res in query_results.results
    }
    dumped = json.dumps(formatted_query_results)[:max_str_len]
    ai_message: AIMessage = queries["raw"]
    tool_id = queries["raw"].tool_calls[0]["id"]
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])
    generated = await gen_answer_chain.ainvoke(swapped_state)
    cited_urls = set(generated["parsed"].cited_urls)
    cited_references = {k: v for k, v in formatted_query_results.items() if k in cited_urls}
    cited_search_results = []
    for k, _ in cited_references.items():
        for res in query_results.results:
            if str(res.url) == k:
                cited_search_results.append(res)
                break
    formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
    return {"messages": [formatted_message], "references": cited_references, "search_results": cited_search_results}