from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# =============================================================================
# Retrieves closely related Wikipedia pages for a given topic, helping to 
# identify relevant subjects and understand typical content.
# =============================================================================

gen_related_topics_inst = '''I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.
Please list the urls in separate lines.

Topic of interest: {topic}'''

gen_related_topics_prompt = ChatPromptTemplate.from_template(gen_related_topics_inst)


# =============================================================================
# Generates a list of hypothetical Wikipedia editors, each representing a 
# unique perspective to a topic, along with a description of their focus areas.
# =============================================================================

gen_perspectives_prompt_inst = '''You need to select a group of Wikipedia editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic. You can use other Wikipedia pages of related topics for inspiration. For each editor, add description of what they will focus on.
Give your answer in the following format: 1. short summary of editor 1:description\n2. short summary of editor 2: description\n...

Wiki page outlines of related topics for inspiration:
{examples}'''

gen_perspectives_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", gen_perspectives_prompt_inst),
        ("user", "Topic of interest: {topic}"),
    ]
)


# =============================================================================
# Generates a list of Google search queries that could help answer a question, 
# based on the conversation context if available.
# =============================================================================

gen_queries_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You want to answer the question using a search engine. What do you type in the search box?
            Write the queries you will use in the following format:- query 1\n- query 2\n..."""
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


# =============================================================================
# Asks focused questions from the perspective of a Wikipedia writer 
# to gather expert insights for an article.
# =============================================================================

gen_qn_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an experienced Wikipedia writer and want to edit a specific page. \
Besides your identity as a Wikipedia writer, you have a specific focus when researching the topic. \
Now, you are chatting with an expert to get information. Ask good questions to get more useful information.

When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.\
Please only ask one question at a time and don't ask what you have asked before.\
Your questions should be related to the topic you want to write.
Be comprehensive and curious, gaining as much unique insight from the expert as possible.\

Stay true to your specific perspective:

{persona}""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


# =============================================================================
# Generates detailed, well-cited answers to support a Wikipedia writer
#  with source URLs in footnotes.
# =============================================================================

gen_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert who can use information effectively. You are chatting with a Wikipedia writer who wants\
 to write a Wikipedia page on the topic you know. You have gathered the related information and will now use the information to form a response.

Make your response as informative as possible and make sure every sentence is supported by the gathered information.
Each response must be backed up by a citation from a reliable source, formatted as a footnote, reproducing the URLS after your response.""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)
