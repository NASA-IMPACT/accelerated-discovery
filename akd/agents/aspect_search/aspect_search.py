from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig

from typing import Dict, List

from langgraph.pregel import RetryPolicy
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph, START
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

from pydantic import Field

from akd.agents.aspect_search.interview_utils import (generate_question, 
                                                      generate_answer,
                                                      survey_subjects)
from akd.agents.aspect_search.structures import (InterviewState,
                                                 update_references, 
                                                 update_search_results)
from akd.tools.search._base import SearchTool, SearchResultItem


class AspectSearchInputSchema(InputSchema):
    """Input schema for aspect search agent"""
    topic: str = Field(..., description="")


class AspectSearchOutputSchema(OutputSchema):
    """Output schema for aspect search agent"""
    search_results: List[SearchResultItem] = Field(None, description="")
    references: Dict = Field(None, description="")


class AspectSearchConfig(BaseAgentConfig):
    """Configuration for Aspect Search Agent"""
    retry_attempts: int = Field(default=3, description="")
    num_editors: int = Field(default=3, description="")
    max_turns: int = Field(default=3, description="")
    top_n_wiki_results: int = Field(default=3, description="")
    search_tool: SearchTool = Field(..., description="")
    category: str = Field(..., description="")
    max_results: int = Field(default=3, description="")


class AspectSearchAgent(BaseAgent):
    input_schema = AspectSearchInputSchema
    output_schema = AspectSearchOutputSchema
    config_schema = AspectSearchConfig

    def _post_init(
        self,
    ) -> None:
        super()._post_init()
        self.llm= ChatOpenAI(
             model=self.config.llm, temperature=self.config.llm, api_key=self.config.api_key
        )
        self.wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, 
                                                      top_k_results=self.config.top_n_wiki_results)
        builder = StateGraph(InterviewState)
        builder.add_node("ask_question", generate_question, retry=RetryPolicy(max_attempts=self.retry_attempts))
        builder.add_node("answer_question", generate_answer, retry=RetryPolicy(max_attempts=self.retry_attempts))
        builder.add_conditional_edges("answer_question", self._route_messages)
        builder.add_edge("ask_question", "answer_question")

        builder.add_edge(START, "ask_question")
        self.interview_graph = builder.compile(checkpointer=False).with_config(
            run_name="Conduct Interviews"
        )


    async def _get_perspectives(self, topic):
        return await survey_subjects.ainvoke(topic, self.llm, self.wikipedia_retriever)


    async def _conduct_interviews(self, topic: str):
        perspectives = await self._get_perspectives(topic)
        editors = perspectives.editors[:1]  
        initial_states = []
        print(f"🤖: Here are your editors!")
        for editor in editors:
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
        print(f"\n🤖: The interviews have started!")
        interview_results = await self.interview_graph.abatch(initial_states)
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
        return interview_results
    

    def _route_messages(self, state: InterviewState, name: str = "Subject_Matter_Expert"):
        messages = state["messages"]
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        if num_responses >= self.MAX_NUM_TURNS:
            return END
        last_question = messages[-2]
        if last_question.content.endswith("Thank you so much for your help!"):
            return END
        return "ask_question"


    async def get_response_async(
        self,
        params: AspectSearchInputSchema,
        **kwargs,
    ) -> AspectSearchOutputSchema:
        """
        Obtains a response from the language model asynchronously.

        Args:
            response_model (Optional[OutputSchema]):
                The schema for the response data. If not set,
                self.output_schema is used.

        Returns:
            OutputSchema: The response from the language model.
        """
        interview_results = await self._conduct_interviews(topic=params.topic)
        search_results = []
        references = {}
        for interview in interview_results:
            search_results = update_search_results(search_results, interview['search_results'])
            references = update_references(references, interview['references'])
        return AspectSearchOutputSchema(search_results=search_results, references=references)

    
    async def _arun(self, params: AspectSearchInputSchema, **kwargs) -> AspectSearchOutputSchema:
        return await self.get_response_async(params, **kwargs)
