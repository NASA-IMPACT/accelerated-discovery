import asyncio
from loguru import logger
from typing import Dict, List, Optional, Union

from akd._base import InputSchema, OutputSchema
from akd.agents._base import BaseAgent, BaseAgentConfig

from langgraph.pregel import RetryPolicy
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph, START
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

from pydantic import Field, ConfigDict

from akd.agents.aspect_search.interview_utils import (generate_question, 
                                                      generate_answer,
                                                      survey_subjects,
                                                      route_messages)
from akd.agents.aspect_search.structures import (InterviewState,
                                                 Perspectives,
                                                 update_references, 
                                                 update_search_results)
from akd.tools.search import SearchTool, SearchResultItem, SearxNGSearchTool


class AspectSearchInputSchema(InputSchema):
    """Input schema for aspect search agent"""
    topic: Union[str, List[str]] = Field(..., description="Topic to search for.")


class AspectSearchOutputSchema(OutputSchema):
    """Output schema for aspect search agent"""
    search_results: Union[List[SearchResultItem], List[List[SearchResultItem]]] = Field(None, description="List of search results returned by the search engine after the interviews.")
    references: Union[Dict[str, str], List[Dict[str, str]]] = Field(None, description="List of references.")
    perspectives: Union[Perspectives, List[Perspectives]] = Field(None, description="List of perspectives used to explore the topic.")


class AspectSearchConfig(BaseAgentConfig):
    """Configuration for Aspect Search Agent"""
    retry_attempts: Optional[int] = Field(default=3, 
                                description="Number of retry attempts.")
    num_editors: Optional[int] = Field(default=3, 
                             description="Number of editors to participate in the interview")
    max_turns: Optional[int] = Field(default=3, 
                           description="Maximum number of turns each interview runs for.")
    top_n_wiki_results: Optional[int] = Field(default=3, 
                                    description="Number of wiki results to return to generate perspectives")
    max_wiki_ctx_len: int = Field(default=1500,
                                 description="Maximum length of wiki content context for perspective generation.")
    search_tool: Optional[object] = Field(default=SearxNGSearchTool(), 
                                    description="Search tool to use.")
    category: Optional[str] = Field(default=None, 
                          description="Category for the search tool.")
    max_ctx_len: int = Field(default=15000,
                                 description="Maximum length of the search result context during interviews.")


class AspectSearchAgent(BaseAgent):
    input_schema = AspectSearchInputSchema
    output_schema = AspectSearchOutputSchema
    config_schema = AspectSearchConfig

    def _post_init(
        self,
    ) -> None:
        super()._post_init()

        self.llm = ChatOpenAI(
             model=self.config.model_name, temperature=self.config.temperature, api_key=self.config.api_key
        )

        self.wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, 
                                                      top_k_results=self.config.top_n_wiki_results)
        
        self.search_tool = self.config.search_tool

        builder = StateGraph(InterviewState)
        builder.add_node("ask_question", 
                         generate_question.bind(llm=self.llm), 
                         retry=RetryPolicy(max_attempts=self.config.retry_attempts))
        builder.add_node("answer_question", 
                         generate_answer.bind(llm=self.llm, 
                                              search_tool=self.search_tool, 
                                              search_category=self.config.category,
                                              max_context_len=self.config.max_ctx_len), 
                         retry=RetryPolicy(max_attempts=self.config.retry_attempts))
        builder.add_conditional_edges("answer_question", 
                                      route_messages.bind(max_turns=self.config.max_turns))
        builder.add_edge("ask_question", "answer_question")
        builder.add_edge(START, "ask_question")

        self.interview_graph = builder.compile(checkpointer=False).with_config(
            run_name="Conduct Interviews"
        )


    async def get_perspectives(self, topic: str) -> List[Perspectives]:
        """
        Retrieves structured perspectives on a topic using related Wikipedia content.

        Args:
            topic (str): The subject to analyze.

        Returns:
            List[Perspectives]: Structured viewpoints generated from Wikipedia information 
            related to the topic.
        """
        perspectives = await survey_subjects.ainvoke(topic, 
                                             llm=self.llm, 
                                             wikipedia_retriever=self.wikipedia_retriever,
                                             max_docs=self.config.top_n_wiki_results,
                                             max_wiki_ctx_len=self.config.max_wiki_ctx_len)
        return Perspectives(editors=perspectives.editors[:self.num_editors])


    async def _conduct_interviews(self, topic: str) -> List[Dict]:
        """
        Runs interviews with multiple editors on a topic, initializing conversations 
        and collecting their responses.

        Args:
            topic (str): The subject to explore.

        Returns:
            List[Dict]: Interview results containing exchanged messages for each editor.
        """
        perspectives = await self.get_perspectives(topic)
        editors = perspectives.editors
        initial_states = []
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
        if self.debug:
            logger.debug(f"🤖: Here are your editors!")
            for editor in editors:
                logger.debug(f"👤: {editor.name} works at {editor.affiliation} as a {editor.role}. They {editor.description}.")
            logger.debug(f"\n🤖: The interviews have started!")
        interview_results = await self.interview_graph.abatch(initial_states)
        if self.debug:
            logger.debug(f"\n🤖: Interview outcomes\n")
            for interview in interview_results:
                logger.debug("👥 Interview\n")
                messages = interview['messages']
                for message in messages:
                    logger.debug(f"{message.name}: {message.content}")
        search_results = []
        references = {}
        for interview in interview_results:
            search_results = update_search_results(search_results, interview['search_results'])
            references = update_references(references, interview['references'])
        return search_results, references, perspectives


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
        topics = [params.topic] if isinstance(params.topic, str) else params.topic
        results = await asyncio.gather(*(self._conduct_interviews(topic=topic) for topic in topics))
        search_results, references, perspectives = zip(*results)

        if isinstance(params.topic, str):
            return AspectSearchOutputSchema(
                search_results=search_results[0],
                references=references[0],
                perspectives=perspectives[0],
            )

        return AspectSearchOutputSchema(
            search_results=list(search_results),
            references=list(references),
            perspectives=list(perspectives),
        )

    
    async def _arun(self, params: AspectSearchInputSchema, **kwargs) -> AspectSearchOutputSchema:
        return await self.get_response_async(params, **kwargs)
