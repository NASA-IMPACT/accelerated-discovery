from typing import Optional

from pydantic import BaseModel, Field
from typing import List

class Document(BaseModel):
    title:Optional[str] = Field(..., description="The title of the document")
    document_type: str = Field("paper", description="The type of the document. Either paper or patent")
    author:Optional[str]=Field(None,description="The author(s) of the document")
    url: Optional[str]=Field(None, description="The url from which the document can be downloaded")
    publication_reference: Optional[str] = Field(None, description="Reference to the publicaton")
    snippet: Optional[str]=Field(None, description="Snippet of text from the document that are relevant for the search terms")
    download_path: Optional[str]=Field(None,description="The location of the file where the provided url has been downloaded. NONE is not downloaded. ERROR if download was unsucessfully attempted")
    year:Optional[int]=Field(None,description="The year in which the paper has been published")
    doi: Optional[str]=Field(None, description="The Document Object Identifier (doi) of the Document")
    citation_count: Optional[int]=Field(None, description="The number of citations of the document")
    reason_for_selection: Optional[str]=Field(None,description="The reason why the document has been selected to be part of the list")
    
    def dict(self):
        return self.model_dump()
    
    def print_document(self):
        
        out = ""
        if self.document_type:
            out+= "document_type: "+ self.document_type + "\n"
        if self.doi:
            out+= "doi - Document Object Identifier: "+ self.doi + "\n"
        if self.title:
            out+= "title: "+ self.title + "\n"
        if self.author:
            out+= "author: "+ self.author + "\n"
        if self.snippet:
            out+= "snippet: "+ self.snippet + "\n"
        if self.url:
            out+= "url: "+ self.url + "\n"
        if self.download_path:
            out+= "download_path: "+ self.download_path + "\n"
        if self.year:
            out+= "year: "+ str(self.year) + "\n"
        if self.citation_count:
            out+= "citation_count: "+ str(self.citation_count) + "\n"
        if self.publication_reference:
            out+= "publication_reference: "+ str(self.publication_reference) + "\n"
        if self.reason_for_selection:
            out+= "reason_for_selection: "+ self.reason_for_selection + "\n"
        out += f""" {"-" * 80} """
        return out


class DocumentSearchResults(BaseModel):
    document_list:List[Document] = Field([], description="list of documents returned from search")