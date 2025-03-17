from langgraph.graph import END, StateGraph, START
from langgraph.pregel import RetryPolicy
from langchain_core.documents import Document

from agent.state import ResearchState, InterviewState
from agent.tools import *
from agent.config import RETRY_ATTEMPTS, MAX_NUM_TURNS, NUM_EDITORS

def route_messages(state: InterviewState, name: str = "Subject_Matter_Expert"):
    messages = state["messages"]
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    if num_responses >= MAX_NUM_TURNS:
        return END
    last_question = messages[-2]
    if last_question.content.endswith("Thank you so much for your help!"):
        return END
    return "ask_question"

# Interview graph
builder = StateGraph(InterviewState)

builder.add_node("ask_question", generate_question, retry=RetryPolicy(max_attempts=RETRY_ATTEMPTS))
builder.add_node("answer_question", gen_answer, retry=RetryPolicy(max_attempts=RETRY_ATTEMPTS))
builder.add_conditional_edges("answer_question", route_messages)
builder.add_edge("ask_question", "answer_question")

builder.add_edge(START, "ask_question")
interview_graph = builder.compile(checkpointer=False).with_config(
    run_name="Conduct Interviews"
)


# Research nodes

async def initialize_research(state: ResearchState):
    topic = state["topic"]
    print(f"\n💬: {topic}\n")
    outline = get_draft_outline(topic)
    print(f"\n🤖: Here is a highlight of your article's initial outline.\n")
    print(f"{outline.as_str} ...")
    editors = await get_perspectives(topic)
    return {
        **state,
        "outline": outline,
        "editors": editors,
    }


async def conduct_interviews(state: ResearchState):
    topic = state["topic"]
    initial_states = []
    print(f"🤖: Here are your editors!")
    for editor in state["editors"].editors[:NUM_EDITORS]:
        initial_states.append(
        {
            "editor": editor,
            "messages": [
                AIMessage(
                    content=f"So you said you were writing an article on {topic}?",
                    name="Subject_Matter_Expert",
                )
            ],
        })
        print(f"👤: {editor.name} works at {editor.affiliation} as a {editor.role}. They {editor.description}.")
    # We call in to the sub-graph here to parallelize the interviews
    print(f"\n🤖: The interviews have started!")
    interview_results = await interview_graph.abatch(initial_states)
    print(f"\n🤖: Interview outcomes\n")
    for interview in interview_results:
        print("👥 Interview\n")
        i = 0
        messages = interview['messages']
        for message in messages:
            print(f"{message.name}: {message.content}")
            i = i + 1
            if i%2 == 0:
                print('\n')
    return {
        **state,
        "interview_results": interview_results,
    }


async def refine_outline(state: ResearchState):
    convos = "\n\n".join(
        [
            format_conversation(interview_state)
            for interview_state in state["interview_results"]
        ]
    )
    updated_outline = await get_refined_outline(topic=state["topic"], 
                                    old_outline=state["outline"].as_str,
                                    conversations=convos)
    print(f"\n🤖: Here is a highlight of your article's refined outline using the interviews for context.\n")
    print(f"{updated_outline.as_str} ...")
    return {**state, "outline": updated_outline}


async def index_references(state: ResearchState):
    print(f"\n🤖: Indexing references")
    all_docs = []
    for interview_state in state["interview_results"]:
        reference_docs = [
            Document(page_content=v, metadata={"source": k})
            for k, v in interview_state["references"].items()
        ]
        all_docs.extend(reference_docs)
    await vectorstore.aadd_documents(all_docs)
    return state


async def write_sections(state: ResearchState):
    outline = state["outline"]
    print(f"\n🤖: Writing each section")
    sections = await section_writer.abatch(
        [
            {
                "outline": outline.as_str,
                "section": section.section_title,
                "topic": state["topic"],
            }
            for section in outline.sections
        ]
    )
    return {
        **state,
        "sections": sections,
    }


async def write_article(state: ResearchState):
    topic = state["topic"]
    sections = state["sections"]
    print(f"\n🤖: Writing the article!")
    draft = "\n\n".join([section.as_str for section in sections])
    article = await writer.ainvoke({"topic": topic, "draft": draft})
    print(f"\n🤖: Done. Print your article below!")
    return {
        **state,
        "article": article,
    }
