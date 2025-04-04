import streamlit as st
from accelerated_discovery.flows.literature_review_flow import LiteratureResearchState
import streamlit as st
import os
from accelerated_discovery.flows.literature_review_flow import LiteratureResearchState
os.environ["STREAMLIT_SERVER_PORT"] = "8502"
from dotenv import load_dotenv
load_dotenv()
st.header("📚 Literature Research Workflow")

with st.popover("ℹ️ **README: About This Workflow**"):


    st.markdown(
            """
            This **agentic workflow** is designed to **automate literature research processes** typically performed by researchers.
    The collected documentation is then ingested by an AI agent, that will help you understanding and applying that to your current research problems
    formulating hypotheses, writing paragraphs for your papers, providing references and brainstorming with you."""
        )
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

import asyncio
with st.form("literature review search"):
    main_research_question = st.text_area("Enter here the description of the research question(s) you would like to explore")
    sources = st.multiselect("Sources",options=["papers", "patents"])
    n_publications = st.number_input("Number of Publications to be Analyzed",value=20)
    output_path = st.text_input("Select a folder to store the output", value = None)

    execute_button = st.form_submit_button("Execute Literature Review")


if execute_button:
    with st.spinner("Workflow running in background, hit refresh to update, you can continue to work on the app while running."):
        state = LiteratureResearchState(main_question=main_research_question, 
                                        output_path = output_path,
                                        selected_sources= sources)
        state.initialize_flow()
        result = asyncio.run(state.run_flow())
        st.write(result)
       