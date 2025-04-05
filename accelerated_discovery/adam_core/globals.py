import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import os
from dotenv import load_dotenv
#load_dotenv(dotenv_path="/home/adam/code/adam_production/.env")
load_dotenv()
from text_2sql_tool_server_client import Client as Text2sqlClient                                               
from adam_backend_client import Client as AdamClient
from data_ops_crew_backend_client import Client as WorkflowClient
from memory_backend_client import Client as MemoryClient
from enum import Enum
from typing import Any, Dict

os.environ["CREWAI_STORAGE_DIR"]=os.path.join(os.environ["AGENT_PERSISTANCE_PATH"], "crew_ai_persistance.db")
if not os.path.exists(os.environ["CREWAI_STORAGE_DIR"]):
    os.mkdir(os.environ["CREWAI_STORAGE_DIR"])
def state_saved_callback(flow_uuid: str, method_name: str, state_data: Dict[str, Any]):
    print(f"Callback Triggered: State saved for {flow_uuid} in {method_name}")

from adam.workflows.persistance import ADAMFlowPersistence
crewai_persistance = ADAMFlowPersistence(callback=state_saved_callback)

class WorkflowType(Enum):
  SchemaEnrichmentCrew = "SchemaEnrichmentCrew"
  Text2SQL = "Text2SQL"
  LiteratureReview = "LiteratureReview"
  NewCrew="NewCrew"

WORKFLOW_INPUT_PARAMETERS = {
  WorkflowType.Text2SQL.value : {"question": "What is the distribution of motorists versus non-motorists involved in crashes?", "database": "bigquery-public-data.nhtsa_traffic_fatalities", "datalake": "SPIDER2" },
  WorkflowType.SchemaEnrichmentCrew.value : { 
    "input_file": "/Users/gliozzo/Documents/Code/adam/data/SPIDER2_DB/bigquery-public-data.austin_bikeshare.json", 
    "output_path": "/Users/gliozzo/Documents/Code/adam/data/workflow_output_examples/schema_enrichment_example" },
  WorkflowType.LiteratureReview.value: {"research_issue" : "How effectively can machine learning techniques, integrating SAR, and optical data, differentiate oil palm plantations from natural forests?", 
                                  "number_of_publications": 20,
                                  "output_path" : "/tmp/literature_review_output",
                                  }
}


NL2INSIGHTS_DATASOURCE = "fp://cm05hr1bv000loahjz5l2u68w"
NL2INSIGHTS_DATASOURCES = ["fp://cm05hr1bv000loahjz5l2u68w"]
AGENT = None
AGENT_IDENTITY = "ADAM"
if not os.path.exists(os.environ["AGENT_PERSISTANCE_PATH"]):
    os.mkdir(os.path.join(os.environ["AGENT_PERSISTANCE_PATH"]))
CHECKPONTER_DB_CONN = sqlite3.connect(os.path.join(os.environ["AGENT_PERSISTANCE_PATH"],"checkpointers.db"), check_same_thread=False)
CHECKPOINTER_DB = SqliteSaver(CHECKPONTER_DB_CONN)
langchain_tools = ["WEB_SEARCH", "NASA_ASTROPHYSICS_DATA_SYSTEM_TOOL", "UNIX_SHELL", "PYTHON_REPL", "NL2INSIGHTS","SPIDER_TEXT2SQL","SPIDER_SCHEMA_LINKING","SQL_TOOLKIT","NASA_SCIENCE_DISCOVERY_ENGINE_TOOL"]
crew_ai_tools = ["FILE_READ", "ASK_QUESTION_TO_HUMAN"]


adam_client = AdamClient(base_url="http://0.0.0.0:7815/")
workflows_client = WorkflowClient(base_url="http://0.0.0.0:7814/")
text2sql_client = Text2sqlClient(base_url="http://0.0.0.0:7813/")
memory_client = MemoryClient(base_url="http://0.0.0.0:7816/")