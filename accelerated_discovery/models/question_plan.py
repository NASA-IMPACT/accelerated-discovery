
from pydantic import BaseModel, Field
from typing import List

class SubQuestion(BaseModel):
    subquestion: str = Field(..., description="The natural language questions")
    search_query: str = Field(..., description="A search query to look up for relevant information about the question in a search engine. It should be composed by a list of optional terms, e.g first keyphrase OR second keyphsase OR third keyphrase")
    explanation: str = Field(..., description="The reason why you think this subquestion is relevant for the main question")
    def to_dict(self):
        return self.model_dump_json()
    
class QuestionDecompositionOutput(BaseModel):
    main_question: str = Field(..., description="The original research question that have been asked for")
    main_search_query: str = Field(..., description="A search query to look up for relevant information about the main question")
    subquestions: List[SubQuestion] = Field([], description="list of subquestions in which the main question should be decomposed in order for you to provide a full answer", max_length=7)
    def to_dict(self):
        return self.model_dump_json()

    
class LiteratureReviewPydanticUI(BaseModel):
    id:str = Field("default", description="The name of the specific run of the flow, used to identify the flow state")
    selected_sources: str =  Field("patents+papers", description="""List of sources to be used for search: use "papers" for Google scholar search, "patents" to look up for patents, "papers+patents" to look up for both """)
    main_question: str=Field(None, description="The main research question that the literature review is going to address")
    output_path: str = Field("/tmp/your_literature_review_folder", description="The path of the folder where the crew output will be stored")
    memory_id: str = Field(None, description="The id of the memory store that will be used to index the ingested documents")
    n_subquestions: int = Field(10, description="The number of subquestions that will be generated from the original question")
    n_publications: int = Field(10, description="The max number of reference document that will be returned as a result of the flow")
    # decomposed_question: QuestionDecompositionOutput = Field(None, description="The decomposition of the original question into subquestions")
    # retrieved_papers: DocumentSearchResults = Field(None,description="The unfiltered list of papers returned as a result of the original search")
    # selected_papers: DocumentSearchResults = Field(None,description="The filtered list of papers returned after filtering the original search")
    # literature_review: str = Field(None,description="The generated literature report survey")
    def to_dict(self):
        return self.model_dump_json()