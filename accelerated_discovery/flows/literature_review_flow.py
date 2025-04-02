
## ADAM IMPORTS
from adam.abstractions.single_task_crew import get_single_task_crew
from adam.globals import memory_client
from memory_backend_client.api.default import (initialize_memory_db_initialize_memory_db_post,
                                               memorize_memorize_post)
    


## INTERNAL IMPORTS

from accelerated_discovery.tools.literature_search_utils import download_file, search_ads, search_serper_run
from accelerated_discovery.models.document import DocumentSearchResults
from accelerated_discovery.models.question_plan import QuestionDecompositionOutput 


import os
from rich import print
from pydantic import BaseModel, Field
from typing import Optional, List
from langgraph.graph import StateGraph, START, END




class LiteratureResearchState(BaseModel):
    main_question: Optional[str]=Field(None, description="The main research question that the literature review is going to address")
    output_path: Optional[str] = Field(None, description="The path of the folder where the crew output will be stored")
    memory_id: Optional[str] = Field(None, description="The id of the memory store that will be used to index the ingested documents")
    n_subquestions: Optional[int] = Field(7, description="The number of subquestions that will be generated from the original question")
    n_publications: Optional[int] = Field(50, description="The max number of reference document that will be returned as a result of the flow")
    decomposed_question: Optional[QuestionDecompositionOutput] = Field(None, description="The decomposition of the original question into subquestions")
    selected_sources:Optional[List[str]] =  Field(["papers"], description="""List of sources to be used for search: use "papers" for Google scholar search, "patents" to look up for patents, "papers+patents" to look up for both """)
    retrieved_documents: Optional[DocumentSearchResults] = Field(None,description="The unfiltered list of documents returned as a result of the original search")
    literature_review: Optional[str] = Field(None,description="The generated literature report survey")
    final_report:Optional[str] =  Field(None,description="The final report summarizing the findings and the generated output")
    
    

    async def initialize_variables(self, state):
        if state.output_path and not os.path.exists(state.output_path):
            os.mkdir(state.output_path)
        # if state.memory_id:
        #     initialize_memory_db_initialize_memory_db_post.sync(client=memory_client, memory_id=state.memory_id)
        
    
    async def  decompose_question_into_subquestions(self, state):
        
        crew= get_single_task_crew(output_file = os.path.join(state.output_path , "subquestions.json") if state.output_path else None,
   			output_pydantic=QuestionDecompositionOutput, verbose= True
            )
        decomposed_question = crew.kickoff({"task_description" : f""" You have been given the task of performing a literature review survey for the following research question:
"{state.main_question}" 
To this aim, you will have to run several search queries against external services such as Google Scholar and the Nasa Astrophysics Discovery system.
You task is to brake down the original research question into a set of subquestions, that could be either focusing on parts of the original research question or provide a relevant new perspective on the original question. 
For each subquestion you will also generate a search engine queries, composed by few short search phrases (not more than 3 words) using the solr syntax adopting OR for AND as needed.
For example , if the search question is  "Can integrating SAR and optical data improve the classification accuracy of small-scale oil palm plantations?" it can be decomposed into (oil palm plantations) AND (SAR optical data)
""",
						"expected_output" : """original questions and generated subquestions, and their related search queries.
Search query should be composed by two short search phrases, grouped by () and separated by AND e.g. "(key word one) AND (key word 2)" """
						})
        
        state.decomposed_question = decomposed_question.pydantic
        return state
   
    async def execute_subquestions(self, state):
        if state.decomposed_question:
            subquestions = state.decomposed_question.subquestions
            search_queries = [state.decomposed_question.main_search_query]
            for subquestion in subquestions:
                search_queries.append(subquestion.search_query)
            papers=[]
            patents = []
            
            # ## execute search
            if "papers" in state.selected_sources:
                for query in search_queries:
                    papers += search_serper_run(query,k=5,document_type = "scholar")
            if "patents" in state.selected_sources:
                for query in search_queries:
                    patents += search_serper_run(query,k=5,document_type = "patents")
        
            document_freq = {}

            seen_documents = set()

            filtered_documents = {}
           

            for document in papers:
                if document.url:
                    if document.url not in seen_documents:
                        seen_documents.add(document.url)
                        if "arxiv" in document.url:
                            document.url = document.url.replace("abs","pdf")
                        document.document_type = "paper"
                        filtered_documents[document.url] = document
                        document_freq[document.url] = 1
                    else: document_freq[document.url] = document_freq.get(document.url) +1
            for document in patents:
                if document.url:
                    if document.url not in seen_documents:
                        seen_documents.add(document.url)
                        if "arxiv" in document.url:
                            document.url = document.url.replace("abs","pdf")
                        document.document_type = "patent"
                        filtered_documents[document.url] = document
                        document_freq[document.url] = 1
                    else: document_freq[document.url] = document_freq.get(document.url) +1
                            
            final_documents = []
            for doc_id in sorted(document_freq.items(), key=lambda item: item[1], reverse=True)[:state.n_publications]:
                final_documents.append(filtered_documents[doc_id[0]])
          
            state.retrieved_documents = DocumentSearchResults(document_list=final_documents)
          
            return state

    
    async def download_documents(self, state):
        output_documents=DocumentSearchResults()
        if state.output_path and state.retrieved_documents:
            
            
            for document in state.retrieved_documents.document_list:    
                download_status = download_file(document.url, state.output_path)
                if download_status:
                    document.download_path = download_status[1]
                else:
                    document.download_path="ERROR"
                output_documents.document_list.append(document)
            state.retrieved_documents = output_documents
        return state
    
    async def update_agent_memory(self, state):
        for document in state.retrieved_documents.document_list:    
            path = document.download_path
            memorize_memorize_post.sync(client=memory_client,url_query = path, memory_id=state.memory_id)
        print(f"Memory {state.memory_id} updated with {len(state.retrieved_documents.document_list)} documents")
        return state
        
    
    
    async def write_literature_review(self, state):
        if state.retrieved_documents and len(state.retrieved_documents.document_list) > 0:
            task_description = f"""You have been given the task to perform a literature review 
on a main research question that you decomposed into research subquestions, as described by the following json
{state.decomposed_question.model_dump()} 
You also executed the above queries using your tools and look up on google for relevant papers and patents, reported below:
{state.retrieved_documents.model_dump()}
=====================================================================================
Your task is to write a literature review to address the following research question using only information from the retrieved documents, whenever relevant
{state.main_question} 
The literature review should contain citations only to the relevant retrieved documents.
You will include references at the bottom of the page to the documents actually mentioned in the literature review summary.
"""
            expected_output = f"""An extensive literature review paper about {state.main_question}.
The literature review is a paper composed by title, body, references to selected papers.
Use (author, date) citation style within the text. 
Generate a complete reference list in the end following same style, containing also the url of the paper whenever available""" 
            crew= get_single_task_crew(output_file = os.path.join(state.output_path , "literature_review.md"))
            state.literature_review = crew.kickoff({"task_description" : task_description ,
                            "expected_output" : expected_output}).raw
            return state
        

    
    async def wrap_data_product(self, state):
        crew= get_single_task_crew(output_file = os.path.join(state.output_path , "final_report.md") if state.output_path else None,
   			output_pydantic=QuestionDecompositionOutput,
            )
        state.final_report = crew.kickoff({"task_description" : f"""
{state.model_dump()}
Write a final report describing the work done so far to answer the main research question
Describe statistics on the generated lists of documents 
Summarize the main findings aggregated from both the literature report
Draft conclusions on whether you have enough material to answer the main research question and provide an answer if possible.
""",
						"expected_output" : """ Report in Markdown format """
						}).raw
        return state
        
    
    def initialize_flow(self):
        flow = StateGraph(LiteratureResearchState)
        flow.add_node(self.initialize_variables)
        flow.add_node(self.decompose_question_into_subquestions)
        flow.add_node(self.execute_subquestions)
        flow.add_node(self.download_documents)
        flow.add_node(self.write_literature_review)
        flow.add_node(self.update_agent_memory)
        flow.add_node(self.wrap_data_product)
        flow.add_edge(START, "initialize_variables")
        flow.add_edge("initialize_variables","decompose_question_into_subquestions")
        flow.add_edge("decompose_question_into_subquestions","execute_subquestions")
        flow.add_edge("execute_subquestions", "download_documents")
        flow.add_edge("download_documents", "write_literature_review")
        flow.add_edge("write_literature_review","update_agent_memory")
        flow.add_edge("update_agent_memory","wrap_data_product")
        flow.add_edge("wrap_data_product",END)
        self.__graph=flow.compile()

    async def run_flow(self):
        print(self.model_dump())
        result = await self.__graph.ainvoke(input=self.model_dump())
        return result
        
import asyncio
if __name__ == "__main__":


    input = LiteratureResearchState(main_question="Long term memory transformers",
                                    output_path = "/Users/gliozzo/Documents/Code/adam/data/nasa_impact/literature_review/Laggraph_Flow_try",
                                    n_subquestions = 3,
                                    selected_sources = ["papers", "patents"],
                                    n_publications = 20,
                                    )
   

    input.initialize_flow()
    print(asyncio.run(input.run_flow()))
