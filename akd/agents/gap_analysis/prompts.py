from langchain_core.prompts import ChatPromptTemplate


# =============================================================================
# Groups scientific paper section titles into hierarchical section clusters
# =============================================================================

section_grouper_inst = '''I have a structured list of section titles from a scientific paper, and I want you to group them into logical sections. Each group should represent a distinct part of the document, starting with the title, followed by related subsections if present. Here's an example:

Input Example:
['KnowledgeHub: An End-to-End Tool for Assisted Scientific Discovery',
'Abstract',
'1 Introduction',
'2 System Description',
'2.1 Document Ingestion',
'2.2 Annotation',
'2.3 Question Answering',
'3 Use-case: Knowledge Discovery for the Battery Domain',
'4 Conclusion',
'Ethical Statement',
'Acknowledgments',
'References']

Output Example:
[['KnowledgeHub: An End-to-End Tool for Assisted Scientific Discovery'], ['Abstract'], ['1 Introduction'], ['2 System Description', '2.1 Document Ingestion', '2.2 Annotation', '2.3 Question Answering'], ['3 Use-case: Knowledge Discovery for the Battery Domain'], ['4 Conclusion'], ['Ethical Statement'], ['Acknowledgments'], ['References']]

Make sure to preserve the hierarchy of sections and subsections. ONLY provide the output.
Now, group the following lists in the same way.'''


section_grouper_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", section_grouper_inst),
            ("user", "{input_sections}"),
        ]
    )


# =============================================================================
# Classifies research paper section titles into predefined key categories like 
# introduction, methodology, and conclusion.
# =============================================================================

section_classifier_inst = """You are an expert in text categorization. Your task is to group section titles from research papers into one of the following key sections: 

## Key Sections
`['introduction', 'related work or background', 'methodology', 'experiments, models and datasets', 'results and discussions', 'limitations', 'future work', 'conclusion', 'appendix', 'misc']`

---

## Instructions
1. Assign each section title to the most appropriate key section based on its content and context.
2. If a section title does not clearly fit into one of the predefined key sections, group it under **`misc`**.
3. Sometimes, there are sections between introduction/related work and experiments/discuss that may be related to methodology. Group accordingly.
4. If there are no sections under a particular key, leave the list empty.
5. Use the following example mappings as a guide to your decisions:

---

## Input
[List of section titles]

### Example Input
['Deep Convolutional Neural Networks for Palm Fruit Maturity Classification *', '1 Introduction', '2 Related Work', '3 Proposed Method', 'Background', '4 Experiments', '5 Limitations', '6 Discussion and Conclusion', '7 References']

---

## Output
A dictionary where each key is a key section, and the value is a list of section titles grouped under that key.
Only provide a JSON dictionary as output. Do not include any explanation or text outside of the dictionary.


### Example Output
'introduction': ['1 Introduction'],
'related work or background': ['2 Related Work', 'Background'],
'methodology': ['3 Proposed Method'],
'experiments, models and datasets': ['4 Experiments'],
'results and discussions': ['6 Discussion and Conclusion'],
'limitations': ['5 Limitations'],
'future work': [],
'conclusion': ['6 Discussion and Conclusion'],
'appendix': [],
'misc': ['Deep Convolutional Neural Networks for Palm Fruit Maturity Classification *', '7 References']
"""

section_classifier_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", section_classifier_inst),
            ("user", "{sections_to_group}"),
        ]
    )


# =============================================================================
# Selects relevant section triples from a paper’s relations to answer a given 
# query based on typical scientific structure.
# =============================================================================

traverse_relations_inst = """You are given a query and the relations of a scientific paper describing its sections:

For example:
[('paper', 'contains_section', 'related work or background'),
 ('paper', 'contains_section', 'results and discussions'),
 ('paper', 'contains_section', 'conclusion'),
 ('paper', 'contains_section', 'introduction'),
 ('paper', 'contains_section', 'methodology'),
 ('paper', 'authored_by', 'author')]

## Instructions
- Extract only the triples which will help answer the question.
- Do not generate any other information or text.
- Use your general knowledge of how scientific papers usually structure their information.
- Only include sections where you are reasonably certain the answer to the query would be found.
- If none of the sections are likely to contain the answer, return an empty list.
- Your output must be exactly a Python-style list of tuples and nothing else.

Examples:

Query: "Identify the main contributions of the paper"  
Relations:
[('paper', 'contains_section', 'related work or background'),
 ('paper', 'contains_section', 'results and discussions'),
 ('paper', 'contains_section', 'conclusion'),
 ('paper', 'contains_section', 'introduction'),
 ('paper', 'contains_section', 'methodology'),
 ('paper', 'authored_by', 'author')]
Output: [('paper', 'contains_section', 'introduction'), ('paper', 'contains_section', 'conclusion')]
"""

traverse_relations_input = "Query: {query}\nRelation: {relations}"

traverse_relations_prompt = ChatPromptTemplate.from_messages([
            ("system", traverse_relations_inst),
            ("user", traverse_relations_input)
        ]
    )


# =============================================================================
# Selects subsections likely to contain information relevant to answering a 
# specific research question.
# =============================================================================

select_subsection_inst = """You are given a list containing relations of the form (section_type relation subsection_title).
Select all relations that may provide direct, supporting, or contextual information to help answer the question.

## Instructions
- Do not generate any other information or text.
- Use your general knowledge of how scientific papers usually structure their information.
- Only include titles where you are reasonably certain the answer to the query would be found.
- If none of the relations are likely to contain the answer, return an empty list.
- Your output must be exactly a Python-style list.

## Example

Question: find all the datasets used in experiments on Knowledge-Graph Based Question Answering
List:
['experiments, models and datasets contains_subsection A. Experimental Setup',
 'experiments, models and datasets contains_subsection B. Baseline Models',
 'experiments, models and datasets contains_subsection C. Analysis of Fine-Tuned Models',
 'experiments, models and datasets contains_subsection D. Analysis of Zero- and Few-Shot Learning',
 'experiments, models and datasets contains_subsection E. Analysis of Model Performance Across Query Complexity and Characteristics']

Output: ['experiments, models and datasets contains_subsection A. Experimental Setup']
"""

select_subsection_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", select_subsection_inst),
            ("user", "Question: {query}\nList:\n{titles}\n")
        ]
    )


# =============================================================================
# Generates an answer to a question using only the content from a specific 
# document section.
# =============================================================================

gen_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the following question based on the provided content from a document section. Use only the information from the content and avoid making assumptions."),
        ("user", "Question: {query}\n\nSection Title: {section_title}\nSection Type: {node_type}\nSection Content: {section_content}\n\nAnswer:"),
    ]
)


# =============================================================================
# Generates a well-structured, cited summary answer to a query using information 
# from multiple source-based responses.
# =============================================================================

summarise_answer_inst = '''You are a helpful AI assistant skilled at crafting detailed, engaging, and well-structured answers. You excel at summarizing and extracting relevant information to generate accurate, clear, and well-cited answers.

Given a query and a dictionary where the key is a `source_id` and the value is an answer generated from the `source` for the query, your task is to provide answers that are:
- **Informative and relevant**: Thoroughly address the user's query using the data present in the sources.
- **Well-structured**: Present information concisely and logically.
- **Cited and credible**: Use inline citations with [number] notation to refer to the context source(s) for each fact or detail included.

### Formatting Instructions
- **Tone and Style**: Maintain a neutral, journalistic tone with engaging narrative flow.
- **Markdown Usage**: Format your response with Markdown for clarity. Use headings, subheadings, bold text, and italicized words when needed to enhance readability.
- **Length and Depth**: Avoid superficial responses and strive for depth without unnecessary repetition.

### Citation Requirements
- **Inline Citations**: Cite every single fact, statement, or sentence using [number] notation corresponding to the source from the provided `context`. For example:  
  - "The Eiffel Tower is one of the most visited landmarks in the world [1]."
- **Citation Syntax**: Use multiple sources for a single detail where applicable. For example:  
  - "Paris is a cultural hub, attracting millions of visitors annually [1][2]."
- **Source List**: Include a "Sources" section at the end of the response that lists each source in detail. Format the sources like this:
  - `[1] source_id_1`
  - `[2] source_id_2`

### General Instructions
- If any information is unsupported by the sources, clearly indicate the limitation.
- Ensure every fact in your response is linked to its respective source using the specified format.
- Organize the source list in numerical order for easy reference.
'''

summarise_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", summarise_answer_inst),
        ("user", "Query: {query}\nSources and context:\n{attributed_answer_list}"),
    ]
)

# =============================================================================
# Gap-to-Query map
# =============================================================================

knowledge_gap_query = """Does the literature demonstrate a lack of comprehensive understanding or up-to-date insights into the topic? \
Are there areas where foundational knowledge is missing, outdated, or fragmented? \
Does the research acknowledge uncertainties, ambiguities, or areas still poorly understood? \
Are researchers calling for further conceptual or descriptive exploration?"""

evidence_gap_query = """Is there a shortage of robust empirical evidence—such as experiments, trials, longitudinal studies, or real-world data—to support, challenge, or validate the core claims made in the literature? \
Are theoretical assertions made without sufficient quantitative or qualitative backing? \
Do review papers or authors explicitly note the need for more primary data collection or stronger empirical validation?"""

theoretical_gap_query = """Do existing theories fail to account for emerging or unexplained phenomena discussed in the literature? \
Are there inconsistencies between theoretical models and real-world observations? \
Are researchers using outdated frameworks? \
Is there a call for new conceptual models, paradigms, or revisions to current theories?"""

methodological_gap_query = """Are the methods used in existing research inadequate, inappropriate, or poorly aligned with the research questions posed? \
Do authors critique the limitations of current methods? \
Is there a need for innovation in research design, sampling, measurement, or analysis?"""

population_gap_query = """Are particular demographic, cultural, social, or identity-based groups underrepresented or entirely missing in the reviewed research? \
Do authors acknowledge this underrepresentation or suggest a need for more inclusive sampling?"""

geographical_gap_query = """Is the research concentrated in a limited set of countries, regions, or contexts, neglecting how the phenomenon may differ elsewhere? \
Are global or comparative perspectives missing? \
Do authors indicate that findings may not generalize beyond certain locations? \
Is there a call for more region-specific or cross-cultural research?"""

gap_query_map = {"knowledge": knowledge_gap_query,
                 "evidence": evidence_gap_query,
                 "theoretical": theoretical_gap_query,
                 "methodological": methodological_gap_query,
                 "population": population_gap_query,
                 "geographical": geographical_gap_query}