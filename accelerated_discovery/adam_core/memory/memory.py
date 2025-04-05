
from dotenv import load_dotenv
load_dotenv()
#from sentence_transformers import SentenceTransformer
#from adam.tools.nasa import get_arXiv_metadata_from_bibcode
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
import os
from fastapi import FastAPI
import uvicorn
#from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from 
from uuid import uuid4
import pandas as pd
from pymilvus import connections, utility
#from adam.tools.nasa import nasa_astrophysics_data_system_api 
import json
from accelerated_discovery.flows.literature_review_flow import LiteratureResearchState
global GLOBALS

GLOBALS ={
    "__VDB" : None,
    "MEMORY_ID" : None,
    "LAST_RESULTS" : [],
    "LAST_QUERY" : ""}

app = FastAPI(
    title="memory_backend",
    debug=True,
    description="API to access basic functionalities of ADAM",
)


# import time 
# from fastapi import  Request
# @app.middleware("http")
# async def add_process_time_header(request: Request, call_next):
#     start_time = time.perf_counter()
#     response = await call_next(request)
#     process_time = time.perf_counter() - start_time
#     response.headers["X-Process-Time"] = str(process_time)
#     return response

if not os.path.exists(os.environ["PERSISTANCE_PATH"]):
    os.mkdir(os.environ["PERSISTANCE_PATH"])

if not os.path.exists(os.path.join(os.environ["PERSISTANCE_PATH"],
                                            "memory_dbs")):
    os.mkdir(os.path.join(os.environ["PERSISTANCE_PATH"],
                                            "memory_dbs"))
db_path = os.path.join(os.environ["PERSISTANCE_PATH"],
                        "memory_dbs",
                        "MEMORY.db")
connections.connect(uri=db_path, alias="ADAM_MEMORY")


@app.post("/get_memory_dbs_list")
def get_memory_dbs_list() -> list[str]:
    available_memories = utility.list_collections(using="ADAM_MEMORY")
    return available_memories

from langchain_milvus import Milvus
@app.post("/initialize_memory_db")
def initialize_memory_db(memory_id: str = None) -> str:
    global GLOBALS
    if memory_id and memory_id != GLOBALS["MEMORY_ID"]:
            print(f"""Initializing memory {memory_id}. This might take a while...""")
            db_path = os.path.join(os.environ["PERSISTANCE_PATH"],
                                "memory_dbs",
                                "MEMORY.db")
            try: 
                index_params = {
                    "index_type": "IVF_FLAT",  # Supported type in local mode
                    "params": {"nlist": 128},  # Adjust nlist based on your dataset
                    "metric_type": "L2"       # Use L2 or IP based on your similarity metric
                }

                        
                GLOBALS["__VDB"] = Milvus(embedding_function= SentenceTransformerEmbeddings(
                                                model_name="all-MiniLM-L6-v2"),
                            connection_args={"uri": db_path},
                            index_params=index_params,
                            collection_name=memory_id)
                
                GLOBALS["MEMORY_ID"] = memory_id
                #print(memorize("https://github.ibm.com/nl2insights/adam/blob/main/README.md", memory_id = memory_id))
                return f"Vector db {memory_id} successfully connected"
            except: return f"Vector db {memory_id} cannot be connected"
    else: return f"Vector db {memory_id} successfully connected"



@app.post("/import_corpus")
def import_corpus(corpus_path: str, data_format:str = "NASA", memory_id: str = "default" ) -> str:
    global GLOBALS
    #if memory_id != "NASA_ADS":
    initialize_memory_db(memory_id)
    print(f"Adding documents from {corpus_path} to memory {memory_id}")
    import os
    if data_format =="NASA":
        i = 0    
        j = 0
        df = pd.read_csv(os.path.join(corpus_path , "open_corpus_ABSTRACTS.csv"))
        print(f"Inporting documents from {corpus_path}, this might take a while ...")
        for index, row in df.iterrows(): 
            file = os.path.join(corpus_path , row["path"][1:] , "abstract.txt")
            if os.path.exists(file):
                text = open(file).read()
                i+=1
                row_dic = row.to_dict()
                row_dic = {key: (value if pd.notna(value) else "[]") for key, value in row_dic.items()}
                document = Document(page_content=text,
                                    metadata=row_dic,
                                    collection_name=memory_id,
                                    connection_args={"corpus_path": corpus_path,
                                                    "data_format": data_format},)
                if i % 100 == 0:
                    print(i , " documents have been imported")
                try:
                    GLOBALS["__VDB"].add_documents(documents=[document], ids=[row_dic["bibcode"]])
                    j +=1     
                except: 
                    print(f"""WARNING: importing document {row_dic["bibcode"]} from {corpus_path}. Data format not recognized""")
        return f"{j} documents correctly imported"
    elif data_format =="FOLDER":
        import os
        
        def scan_directory_recursively(path):
            """Recursively scans the directory and yields file paths."""
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        yield from scan_directory_recursively(entry.path)
                    else:
                        yield entry.path
        try:
            
            for entry in scan_directory_recursively(corpus_path):
               
                try:                    
                            
                    print(memorize(url= entry, memory_id = memory_id))
                     
                except:
                    return f"Unable to import {entry}."
        except:
            return f"ERROR, corpus {corpus_path} has not been imported"
    elif data_format =="NASA_IMPACT":
        with open(corpus_path) as f:
            i =1
            for str_line in f:
                line = json.loads(str_line)
                memorize(memory_id= memory_id , url=corpus_path, raw_text = line["text"], id = line["_id"])
                if i % 100 == 0:
                    print(i , " documents have been imported")
                i+=1
        return f"{i} documents correctly imported"
    elif data_format =="LITERATURE_REVIEW":
        literature_research_state = json.load(open(os.path.join(corpus_path,"output_state.json")))
        literature_review_output = LiteratureResearchState(**literature_research_state)
        index = {}
        for line in open(os.path.join(corpus_path,"index.csv")):
            index[line.split("\t")[0]] = line.split("\t")[1]
        initialize_memory_db(memory_id)
        for filename in index.keys():
            memorize(memory_id= memory_id ,
                     url=os.path.join(corpus_path,filename),
                     id=index[filename]
            )
        
                     
            
        
        pass
    
    # else: return "Data Format {data_format} not recognized"
    #else: return "SPECIAL MEMORY SELECTED. Cannot import data. Use a regular memory instead"
   
   
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import List, Optional

class StringLoader(BaseLoader):
    def __init__(self, text: str, metadata: Optional[dict] = None):
        """Initializes the StringLoader with text and optional metadata."""
        self.text = text
        self.metadata = metadata or {}

    def load(self) -> List[Document]:
        """Loads the text into a LangChain Document."""
        return [
            Document(
                page_content=self.text,
                metadata=self.metadata
            )
        ]   
        
from io import StringIO
@app.post("/memorize")
def memorize(memory_id: str =None , url: str ="NO_SORCE_PROVIDED", raw_text:str=None, id:str =None) -> str:
    global GLOBALS
    initialize_memory_db(memory_id)

    # if memory_id == "NASA_ADS":
    #     return "SPECIAL MEMORY SELECTED. Cannot import data. Use a regular memory instead"
    # else:
    if raw_text:
       
        metadata = {"source": url,
                    "id" : id}
        

        loader = StringLoader(text=raw_text, metadata=metadata)
    else:
        loader = DoclingLoader(file_path=[url], chunker=HybridChunker())
    docs = loader.load()
    uuids = [str(uuid4()) for _ in range(len(docs))]
    GLOBALS["__VDB"].add_documents(documents=docs, ids=uuids)
    return f"Resource from {url} has been added to collection"

@app.post("/recall")
def recall(query: str = "", 
           k: int = 5, 
           memory_id:str=None,
           return_metadata: bool =False, 
           filters:str =None,
           add_arxive_metadata:bool =False) -> list:
    """Filters are strings in the form 'metadata_param == "value"', 
    where metadata_param is a key in the metadata dictionary. 
    "citation_count:[5 TO *]"
    "year > 2000"
    Default key is pk"""
    global GLOBALS
    # if memory_id=="NASA_ADS":
    #     # ("moon", k=10, return_metadata=True, sort= "year", fq = "citation_count:[11 TO *]"
    #     return nasa_astrophysics_data_system_api(query, k=k,sort=sort,
    #                                              return_metadata=return_metadata, 
    #                                              fq = filters)
    # else:
    if not GLOBALS["__VDB"]: initialize_memory_db(memory_id)
    if GLOBALS["__VDB"]:
        results = GLOBALS["__VDB"].similarity_search(query=query, k=k,expr=filters)
        if return_metadata:
            results_final = [(doc.page_content , doc.metadata) for doc in results]
        else: results_final = [doc.page_content for doc in results]
        if add_arxive_metadata and return_metadata==True:
            enriched_results = []
            for result in results_final:
                enriched_text= f"""bibcode: {result[1]["bibcode"]}\n{get_arXiv_metadata_from_bibcode(result[1]["bibcode"])}"""
                enriched_results.append((enriched_text , result[1]))
            results_final =enriched_results         
            
        GLOBALS["LAST_RESULTS"] = results_final
        GLOBALS["LAST_QUERY"] = query
        return results_final
    else: return []


@app.post("/get_globals")
def get_globals():
    global GLOBALS
    output={}
    for att in GLOBALS.keys():
        if not att.startswith("__"):
            output[att] = GLOBALS[att]    
    return output



if __name__ == "__main__":
    # Configure logging
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)

    initialize_memory_db()
    print("FastAPI Docs http://0.0.0.0:7816/docs")
    uvicorn.run(app, host="0.0.0.0", port=7816)
    