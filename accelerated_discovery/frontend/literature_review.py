import streamlit as st
from accelerated_discovery.flows.literature_review_flow import LiteratureResearchState
import streamlit as st
import os
from globals import crewai_persistance
from accelerated_discovery.frontend.streamlit_utils import pydantic_to_markdown
from accelerated_discovery.flows.literature_review_flow import LiteratureResearchState
from accelerated_discovery.utils.file_system_utils import sanitize_filename
os.environ["STREAMLIT_SERVER_PORT"] = "8502"



st.header("📚 Literature Research Workflow")

with st.popover("ℹ️ **README: About This Workflow**"):


    st.markdown(
            """
            This **agentic workflow** is designed to **automate literature research processes** typically performed by researchers.
    The collected documentation is then ingested by an AI agent, that will help you understanding and applying that to your current research problems
    formulating hypotheses, writing paragraphs for your papers, providing references and brainstorming with you."""
        )
    # 🚀 Step-by-Step Process
    st.markdown(
        """
        🧩 **Step 1:** Decompose the main research issue into **sub-issues**.  
        🔍 **Step 2:** Query **Google Scholar** & **NASA ASD Portal** to gather relevant publications.  
        🏷️ **Step 3:** **Filter** retrieved publications based on **relevance** & provide explanations.  
        📥 **Step 4:** Download the **selected publications** for deeper analysis.  
        📜 **Step 5:** Generate a **BibTeX file** for citation management.  
        📝 **Step 6:** Write a structured **literature survey** summarizing key findings.  
        📦 **Step 7:** Package all generated data into a **Data Product folder** & deliver it to the user.  
        🧠 **Step 8:** Ingest papers into a **memory-augmented conversational agent** for **Retrieval-Augmented Generation (RAG)**.  
        🤖 **Step 9:** Interact with a **domain expert AI agent** for Q&A, brainstorming, & guided research assistance.  
        """
    )

#workflow_executions = {item['workflow_name'] for item in st.session_state.workflows_db.all() if 'workflow_name' in item}

# with st.sidebar:
#     flow_uiid = st.selectbox("Load Persisted Workflow", options=workflow_executions)

#     for thread in st.session_state.latest_treads:
#         with st.expander(thread):
#             st.write(thread)
#             st.write(crewai_persistance.load_state(thread))
#             # time.sleep(2)
import asyncio
with st.form("literature review search"):
    main_research_question = st.text_area("Enter here the description of the research question(s) you would like to explore")
    memory_id = st.text_input("Enter the name of the target memory", value = None)
    execute_button = st.form_submit_button("Execute Literature Review")


if execute_button:
    with st.spinner("Workflow running in background, hit refresh to update, you can continue to work on the app while running."):
    #with st.status("Flow is running in the background"):
        state = LiteratureResearchState(main_question=main_research_question, output_path = os.path.join("/tmp", memory_id) , memory_id = memory_id)
        state.initialize_flow()
        st.write(asyncio.run(state.run_flow()))
        # st.markdown(f"executing workflow with uidd {flow.flow_id}")
        # flow.kickoff({"main_question" : main_research_question})
            
# if col2.button("Refresh"):
#     st.session_state.latest_treads = crewai_persistance.get_states(last_n=3)
#     st.rerun()
# import time  
# with col3.popover("Optional Parameters"):
#     input = sp.pydantic_form(key="my_form", model=LiteratureReviewPydanticUI)
#     if input:
#         #with st.status("Flow is running in the background"):
#         flow.kickoff(input.model_dump())
#         time.sleep(5)
#         st.session_state.latest_treads = crewai_persistance.get_states(last_n=3)
#         st.rerun()
        

# # st.write(crewai_persistance.get_states())
# # if st.session_state.flow_uidd:
# #     st.write(crewai_persistance.load_state(st.session_state.flow_uidd))
# if len(st.session_state.latest_treads) > 0 : 
#     workflow_id = st.session_state.latest_treads[0]
#     #st.write(workflow_id)
#     workflow = crewai_persistance.load_state(workflow_id)
#     literature_research_pydantic = LiteratureResearchState(**workflow)
#     st.markdown("""##### Workflow Output""")

#     if workflow["literature_review"]:
        
#         st.markdown(workflow["literature_review"])
#         col1 , col2, col3 = st.columns(3)
        
#         with col1.popover("Show Intermediate Steps"): 
#             st.markdown(pydantic_to_markdown(literature_research_pydantic))
#         col2.markdown("##### Enter Name to save ->")
#         save = col3.text_input("Save Workflow",help="Workflow Name", value = "" ,label_visibility="collapsed")
#         if save != "":
#             st.session_state.workflows_db.insert({"workflow_id":st.session_state.latest_treads[0] , "workflow_name":save})
#             st.write(f"Workflow {st.session_state.latest_treads[0]} has been saved with name {save}")
#             st.rerun()
#     else: 
#         st.markdown("Workflow is in progress. You can inspect the intermediated steps by clicking the button above")
#         with st.popover("Show Intermediate Steps"): 
#             st.markdown(pydantic_to_markdown(literature_research_pydantic))
